//
//  CT_ImageRegistration_C++.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/02/06.
//  Copyright (c) 2015 Nozomu Ishiguro. All rights reserved.
//

#include "CTXAFS.hpp"
#include "imageregistration_kernel_src_cl.hpp"

#define  REGMODE 0 //0:XY, 1:XY+rotation, 2:XY+scale, 3:XY+rotation+scale
#include "RegMode.hpp"


string kernel_preprocessor_def(string title, string pointertype,
                               string memobjectname,string pointername,
                               int energy_num, int angle_num){
    ostringstream OSS;
    
    OSS<<"#define "<<title<<"_DEF ";
    
    for (int i=0; i<energy_num; i++) {
        for (int j=0; j<angle_num; j++) {
            OSS << pointertype << memobjectname<<"_E"<<i<<"A"<<j<<", ";
        }
    }
    OSS  << pointertype <<" __local *"<< pointername << endl << endl;
    
    OSS<<"#define "<<title<<"_P ";
    for (int i=0; i<energy_num; i++) {
        for (int j=0; j<angle_num; j++) {
            OSS << pointername;
            OSS << "[" << i+j*energy_num <<"]=";
            OSS <<memobjectname<<"_E"<<i<<"A"<<j;
            if((i!=energy_num-1) | (j!=angle_num-1)) OSS<<"; ";
        }
    }
    OSS<<endl<<endl;
    
    return OSS.str();
}


int mt_output_thread(int startAngleNo, int EndAngleNo, int startEnergyNo, int endEnergyNo,
                     string output_dir, vector<vector<float*>> mt_outputs,int waittime){
    
    this_thread::sleep_for(chrono::seconds(waittime));
    ostringstream oss;
    for (int j=startAngleNo; j<=EndAngleNo; j++) {
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            string fileName_output= output_dir+ "/"+ EnumTagString(i)+ AnumTagString(j,"/mtr", ".raw");
            oss << "output file: " << fileName_output << "\n";
            outputRawFile_stream(fileName_output,mt_outputs[j-startAngleNo][i-startEnergyNo]);
        }
    }
    oss << "\n";
    cout << oss.str();
    for (int j=startAngleNo; j<=EndAngleNo; j++) {
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            delete [] mt_outputs[j-startAngleNo][i-startEnergyNo];
        }
    }
    return 0;
}

int imageReg_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                    cl::Buffer dark_buffer,
                    cl::Buffer I0_target_buffer,
                    vector<cl::Buffer> I0_sample_buffer,
                    int startAngleNo,int EndAngleNo,
                    int startEnergyNo, int endEnergyNo, int targetEnergyNo,
                    int dE, string fileName_base, string output_dir,bool last){
    try {
        time_t start_t,readfinish_t,end_t;
        time(&start_t);
        
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
        string devicename = device.getInfo<CL_DEVICE_NAME>();
        size_t WorkGroupSize = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), IMAGE_SIZE_X);
        int dA;
        cl_command_queue_properties device_queue_info;
        device.getInfo(CL_DEVICE_QUEUE_PROPERTIES, &device_queue_info);
        if ((device_queue_info & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0x0) {
            dA=1;
        }else{
            dA=EndAngleNo-startAngleNo+1;
        }
        size_t num_enqueue_repeat=(EndAngleNo-startAngleNo+1)/dA;
        if((EndAngleNo-startAngleNo+1)%dA>0) num_enqueue_repeat++;
        
        //vector declaration
        vector<float*> transpara;
        vector<cl::Buffer> It_target_buffers;
        vector<vector<cl::Buffer>> It_sample_buffers;
        vector<cl::Buffer> mt_target_buffers;
        vector<vector<cl::Buffer>> mt_sample_buffers;
        vector<vector<cl::Buffer>> mt_out_buffers;
        vector<cl::Buffer> transpara_buffers;
        vector<vector<cl::Kernel>> kernel_parallel;
        
        
        //kernel dimension declaration
        const cl::NDRange global_item_size(WorkGroupSize*dA,1,1);
        const cl::NDRange local_item_size(WorkGroupSize,1,1);
        
        
        //Energy loop setting
        vector<int> energyNo={startEnergyNo,endEnergyNo};
        vector<int> LoopStartenergyNo;
        LoopStartenergyNo.push_back(min(targetEnergyNo, endEnergyNo));
        LoopStartenergyNo.push_back(max(targetEnergyNo, startEnergyNo));
        
        if (startAngleNo==EndAngleNo) {
            cout << "Device: "<< devicename << ", angle: "<<startAngleNo<< ", processing It...\n\n";
        } else {
            cout << "Device: "<< devicename << ", angle: "<<startAngleNo<<"-"<<EndAngleNo<< "    processing It...\n\n";
        }
        
        //Loop process for data input
        /*target It data input*/
        float *It_img_target;
        It_img_target = new float[IMAGE_SIZE_M*(EndAngleNo-startAngleNo+1)];
        string fileName_It_target = fileName_base + EnumTagString(targetEnergyNo) + ".his";
        readHisFile_stream(fileName_It_target,startAngleNo,EndAngleNo,It_img_target,IMAGE_SIZE_M);
        
        /* Sample It data input */
        vector<float *>It_img;
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            It_img.push_back(new float[IMAGE_SIZE_M*(EndAngleNo-startAngleNo+1)]);
            string fileName_It = fileName_base + EnumTagString(i) + ".his";
            readHisFile_stream(fileName_It,startAngleNo,EndAngleNo,It_img[i-startEnergyNo],IMAGE_SIZE_M);
        }
        
        time(&readfinish_t);
        
        //create buffer, kernel, pointer, ostringstream vectors
        for (int j=startAngleNo; j<=EndAngleNo; j++) {
            It_target_buffers.push_back(cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL));
            
            mt_target_buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M, NULL, NULL));
            
            
            vector<cl::Buffer> It_sample_buffers_atAng;
            vector<cl::Buffer> mt_sample_buffers_atAng;
            vector<cl::Buffer> mt_out_buffers_atAng;
            for (int i=startEnergyNo; i<=endEnergyNo; i++) {
                It_sample_buffers_atAng.push_back(cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL));
                mt_sample_buffers_atAng.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M, NULL, NULL));
                mt_out_buffers_atAng.push_back(cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*IMAGE_SIZE_M, NULL, NULL));
            }
            It_sample_buffers.push_back(move(It_sample_buffers_atAng));
            mt_sample_buffers.push_back(move(mt_sample_buffers_atAng));
            mt_out_buffers.push_back(move(mt_out_buffers_atAng));
        }
        for (int j=startAngleNo; j<=EndAngleNo; j++){
            command_queue.enqueueWriteBuffer(It_target_buffers[j-startAngleNo], CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M, It_img_target+(j-startAngleNo)*IMAGE_SIZE_M);
            for (int i=startEnergyNo; i<=endEnergyNo; i++) {
                command_queue.enqueueWriteBuffer(It_sample_buffers[j-startAngleNo][i-startEnergyNo], CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M, It_img[i-startEnergyNo]+(j-startAngleNo)*IMAGE_SIZE_M);
            }
        }
        command_queue.finish();
        
        //delete input pointer
        delete [] It_img_target;
        for (int i=startEnergyNo; i<=endEnergyNo; i++){
            delete [] It_img[i-startEnergyNo];
        }
        
        for (int numk=0; numk<num_enqueue_repeat; numk++) {
            kernel_parallel.push_back(kernel);
            transpara_buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*transpara_num*dA*dE, NULL, NULL));
            transpara.push_back(new float[transpara_num*dA*dE]);
            
            //kernel[0] set args
            kernel_parallel[numk][0].setArg(0, dark_buffer);
            kernel_parallel[numk][0].setArg(1, I0_target_buffer);
            int t=2;
            for (int k=0; k<dA; k++) {
                kernel_parallel[numk][0].setArg(t, It_target_buffers[(numk*dA+k)%(EndAngleNo-startAngleNo+1)]);
                t++;
            }
            kernel_parallel[numk][0].setArg(t, cl::Local(sizeof(cl_float*)*dA));
            t++;
            for (int k=0; k<dA; k++) {
                kernel_parallel[numk][0].setArg(t, mt_target_buffers[(numk*dA+k)%(EndAngleNo-startAngleNo+1)]);
                t++;
            }
            kernel_parallel[numk][0].setArg(t, cl::Local(sizeof(cl_float*)*dA));
        }
        
        //mt_target transform
        for (int numk=0; numk<num_enqueue_repeat; numk++) {
            command_queue.enqueueNDRangeKernel(kernel_parallel[numk][0], cl::NullRange, global_item_size, local_item_size, NULL, NULL);
        }
        command_queue.finish();
        //delete [] It_img_target;
        
        //mt sample transform
        for (int i=startEnergyNo; i<=endEnergyNo; i+=dE) {
            //kernel[1] set args
            for (int numk=0; numk<num_enqueue_repeat; numk++) {
                //int angNum=numk*dA;
                
                kernel_parallel[numk][1].setArg(0, dark_buffer);
                int t=1;
                for (int l=0; l<dE; l++) {
                    kernel_parallel[numk][1].setArg(t, I0_sample_buffer[(i+l-startEnergyNo)%(endEnergyNo-startEnergyNo+1)]);
                    t++;
                }
                kernel_parallel[numk][1].setArg(t, cl::Local(sizeof(cl_float*)*dE));
                t++;
                for (int l=0; l<dE; l++) {
                    for (int k=0; k<dA; k++) {
                        kernel_parallel[numk][1].setArg(t, It_sample_buffers[(numk*dA+k)%(EndAngleNo-startAngleNo+1)][(i+l-startEnergyNo)%(endEnergyNo-startEnergyNo+1)]);
                        t++;
                    }
                }
                kernel_parallel[numk][1].setArg(t, cl::Local(sizeof(cl_float*)*dE*dA));
                t++;
                for (int l=0; l<dE; l++) {
                    for (int k=0; k<dA; k++) {
                        kernel_parallel[numk][1].setArg(t, mt_sample_buffers[(numk*dA+k)%(EndAngleNo-startAngleNo+1)][(i+l-startEnergyNo)%(endEnergyNo-startEnergyNo+1)]);
                        t++;
                    }
                }
                kernel_parallel[numk][1].setArg(t, cl::Local(sizeof(cl_float*)*dE*dA));
                if (i+dE>endEnergyNo) {
                    kernel_parallel[numk][1].setArg(t+1, (endEnergyNo-i+1));
                } else {
                    kernel_parallel[numk][1].setArg(t+1, dE);
                }
                
            }
            //execute
            for (int numk=0; numk<num_enqueue_repeat; numk++) {
                command_queue.enqueueNDRangeKernel(kernel_parallel[numk][1], NULL, global_item_size, local_item_size, NULL, NULL);
                //printf("    sample mt: %d\n",ret);
            }
            command_queue.finish();
            
            //delete [] It_img[i-startEnergyNo];
        }
        
        
        
        //Loop of OCL process
        //process when (sample Enegry No. == target Energy No.)
        if (targetEnergyNo>=startEnergyNo && targetEnergyNo<=endEnergyNo) {
            for (int j=startAngleNo; j<=EndAngleNo; j++) {
                command_queue.enqueueCopyBuffer(mt_target_buffers[j-startAngleNo], mt_out_buffers[j-startAngleNo][targetEnergyNo-startEnergyNo], 0, 0, IMAGE_SIZE_M);
                //printf("    target mt: %d\n",ret);
            }
            command_queue.finish();
            
            ostringstream oss;
            for (int j=startAngleNo; j<=EndAngleNo; j++) {
                oss << "Device: "<< devicename << ", angle: "<<j<< ", energy: "<<targetEnergyNo<<"\n";
                oss <<OSS_TARGET;
            }
            cout << oss.str();
        }
        
        for (int s=0; s<2; s++) {//1st cycle:i<targetEnergyNo, 2nd cycle:i>targetEnergyNo
            int di=(-1+2*s);
            if ((LoopStartenergyNo[s]+di)*di>energyNo[s]*di) {
                continue;
            }
            
            for (int numk=0; numk<num_enqueue_repeat; numk++) {
                for (int i=0; i<transpara_num*dA*dE; i++) {
                    transpara[numk][i]=0.0;
                }
                command_queue.enqueueWriteBuffer(transpara_buffers[numk], CL_TRUE, 0, sizeof(cl_float)*transpara_num*dA*dE, transpara[numk], NULL, NULL);
            }
            command_queue.finish();
            
            for (int i=LoopStartenergyNo[s]+di; i*di<=energyNo[s]*di; i+=(di*dE)) {
                for (int numk=0; numk<num_enqueue_repeat; numk++) {
                    int t=0;
                    for (int l=0; l<dE; l++) {
                        for (int k=0; k<dA; k++) {
                            kernel_parallel[numk][2].setArg(t, mt_sample_buffers[(numk*dA+k)%(EndAngleNo-startAngleNo+1)][(i+l-startEnergyNo)%(endEnergyNo-startEnergyNo+1)]);
                            t++;
                        }
                    }
                    kernel_parallel[numk][2].setArg(t, cl::Local(sizeof(cl_float*)*dA*dE));
                    t++;
                    for (int k=0; k<dA; k++) {
                        kernel_parallel[numk][2].setArg(t, mt_target_buffers[(numk*dA+k)%(EndAngleNo-startAngleNo+1)]);
                        t++;
                    }
                    kernel_parallel[numk][2].setArg(t, cl::Local(sizeof(cl_float*)*dA));
                    t++;
                    for (int l=0; l<dE; l++) {
                        for (int k=0; k<dA; k++) {
                            kernel_parallel[numk][2].setArg(t, mt_out_buffers[(numk*dA+k)%(EndAngleNo-startAngleNo+1)][(i+l-startEnergyNo)%(endEnergyNo-startEnergyNo+1)]);
                            t++;
                        }
                    }
                    kernel_parallel[numk][2].setArg(t, cl::Local(sizeof(cl_float*)*dA*dE));
                    kernel_parallel[numk][2].setArg(t+1, transpara_buffers[numk]);
                    kernel_parallel[numk][2].setArg(t+2, cl::Local(sizeof(cl_float)*transpara_num));
                    kernel_parallel[numk][2].setArg(t+3, cl::Local(sizeof(cl_float)*transpara_num));
                    kernel_parallel[numk][2].setArg(t+4, cl::Local(sizeof(cl_float)*reductpara_num));
                    kernel_parallel[numk][2].setArg(t+5, cl::Local(sizeof(cl_float)*WorkGroupSize*reductpara_num));
                    if (i+dE>endEnergyNo) {
                        kernel_parallel[numk][2].setArg(t+6, (endEnergyNo-i+1));
                    } else {
                        kernel_parallel[numk][2].setArg(t+6, dE);
                    }
                }
                
                for (int numk=0; numk<num_enqueue_repeat; numk++) {
                    command_queue.enqueueNDRangeKernel(kernel_parallel[numk][2], NULL, global_item_size, local_item_size, NULL, NULL);
                    //printf("        sample mt: %d\n", ret);
                }
                command_queue.finish();
                
                for (int numk=0; numk<num_enqueue_repeat; numk++) {
                    command_queue.enqueueReadBuffer(transpara_buffers[numk], CL_TRUE, 0, sizeof(cl_float)*transpara_num*dA*dE, transpara[numk], NULL, NULL);
                }
                command_queue.finish();
                ostringstream oss;
                for (int numk=0; numk<num_enqueue_repeat; numk++) {
                    for (int l=0; l<dE; l++) {
                        if ((i+di*l>endEnergyNo)|(i+di*l<0)) {
                            break;
                        }
                        for (int k=0; k<dA; k++) {
                            if (numk*dA+k>EndAngleNo) {
                                break;
                            }
                            oss << "Device: "<< devicename << ", angle: "<<startAngleNo+numk*dA+k<< ", energy: "<<i+di*l<<"\n";
                            OSS_SAMPLE(oss,transpara[numk]+transpara_num*(l+dE*k));
                        }
                    }
                }
                cout << oss.str();
            }
        }
        
        
        //Loop of data output
        vector<vector<float*>> mt_outputs;
        for (int j=startAngleNo; j<=EndAngleNo; j++) {
            vector<float*> mt_outputs_atAng;
            for (int i=startEnergyNo; i<=endEnergyNo; i++) {
                //cout << j << ","<<i<<endl;
                mt_outputs_atAng.push_back(new float[IMAGE_SIZE_M]);
                command_queue.enqueueReadBuffer(mt_out_buffers[j-startAngleNo][i-startEnergyNo], CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M, mt_outputs_atAng[i-startEnergyNo], NULL, NULL);
                //printf("            read mt buffer %d,%d: %d\n",j,i, ret);
            }
            command_queue.finish();
            mt_outputs.push_back(move(mt_outputs_atAng));
        }
        
        
        time(&end_t);
        int delta_t;
        if (last) {
            delta_t=0;
        }else{
            delta_t= difftime(readfinish_t,start_t)*2;
        }
        
        thread th_output(mt_output_thread,
                         startAngleNo,EndAngleNo,startEnergyNo,endEnergyNo,
                         output_dir,move(mt_outputs),delta_t);
        if(last) th_output.join();
        else th_output.detach();

    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
    
    
    return 0;
}




int imageRegistlation_ocl(string fileName_base, string output_dir,
                          int startEnergyNo, int endEnergyNo, int targetEnergyNo,
                          int startAngleNo, int endAngleNo,
                          vector<OCL_platform_device> plat_dev_list)
{
    cl_int ret;
    
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
        string fileName_output = output_dir + "/" + EnumTagString(i);
        MKDIR(fileName_output.c_str());
    }
    
    /*OpenCL　Plattform check*/
    vector<cl::Platform> platform_ids;
    cl::Platform::get(&platform_ids);
    
    vector<vector<cl::Device>> device_ids_USE;
    vector<cl::Platform> platform_ids_USE;
    vector<bool> platform_exists;
    for (int i=0; i<platform_ids.size(); i++) {
        vector<cl::Device> device_ids_plat_i;
        vector<cl::Device> device_id;
        platform_exists.push_back(false);
        platform_ids[i].getDevices(CL_DEVICE_TYPE_ALL, &device_id);
        for (int j=0; j<plat_dev_list.size(); j++) {
            if (plat_dev_list[j].platform_num==i) {
                device_ids_plat_i.push_back(device_id[plat_dev_list[j].device_num]);
                platform_exists[i]=true;
            }
        }
        if (platform_exists[i]) {
            device_ids_USE.push_back(device_ids_plat_i);
            platform_ids_USE.push_back(platform_ids[i]);
        }
        
    }
    
    vector<cl::Context> contexts;
    vector<vector<cl::Kernel>> kernels;
    vector<cl::CommandQueue> queues;
    vector<cl_command_queue_properties> queue_properties;
	vector<int> plat_if_of_dev;
    vector<int> dA,di,dE;
   
    //kernel source from cl file
	/*ifstream ifs("../../imageregistration_kernel_src.cl",ios::in);
	//ifstream ifs("../../opencl_device_info.cl", ios::in);
    if(!ifs) {
        cerr << "   Failed to load kernel \n\n" << endl;
        return -1;
    }
    istreambuf_iterator<char> it(ifs);
    istreambuf_iterator<char> last;
    string kernel_src(it,last);
    ifs.close();*/
    
    int t=1;
    for (int i=0; i<device_ids_USE.size(); i++) {
        
        
        /*OpenCL Context create*/
        cl_context_properties properties[]={CL_CONTEXT_PLATFORM, (cl_context_properties)(platform_ids_USE[i])(), 0};
        contexts.push_back(cl::Context(device_ids_USE[i], properties,NULL,NULL,&ret));
        
        
        
    
        for (int j=0; j<device_ids_USE[i].size(); j++) {
            cout<<"Device No. "<<t<<"\n";
            string platform_param;
            platform_ids_USE[i].getInfo(CL_PLATFORM_NAME, &platform_param);
            cout << "CL PLATFORM NAME: "<< platform_param<<"\n";
            platform_ids_USE[i].getInfo(CL_PLATFORM_VERSION, &platform_param);
            cout << "   "<<platform_param<<"\n";
            
            string device_pram;
            ret = device_ids_USE[i][j].getInfo(CL_DEVICE_NAME, &device_pram);
            cout << "CL DEVICE NAME: "<< device_pram<<"\n";
            
            /*Open CL command que create*/
            cl_command_queue_properties device_queue_info;
            ret = device_ids_USE[i][j].getInfo(CL_DEVICE_QUEUE_PROPERTIES, &device_queue_info);
            if ((device_queue_info & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0x0) {
                cout<<"OUT OF ORDER EXEC MODE ENABLE"<<endl<<endl;
                queue_properties.push_back(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
				//queue_properties.push_back(0);
                dA.push_back(1);
                di.push_back(1);
                dE.push_back(1);
            }else{
                cout<<"OUT OF ORDER EXEC MODE NOT ENABLE"<<endl<<endl;
                queue_properties.push_back(0);
                dA.push_back(4);
                di.push_back(4);
                dE.push_back(1);
            }
            
            
            //kernel sources
            string kernel_src0="";
            kernel_src0 += kernel_preprocessor_def("IT_TARGET_BUF","__global float *", "It_target","It_target_p",1,dA[t-1]);
            //cout<< kernel_src0;
            kernel_src0 += kernel_preprocessor_def("MT_TARGET_BUF","__global float *", "mt_target","mt_target_p",1,dA[t-1]);
            kernel_src0 += kernel_preprocessor_def("I0_SAMPLE_BUF","__global float *", "I0_sample","I0_sample_p",dE[t-1],1);
            kernel_src0 += kernel_preprocessor_def("IT_SAMPLE_BUF","__global float *", "It_sample","It_sample_p",dE[t-1],dA[t-1]);
            //cout<< kernel_src0;
            kernel_src0 += kernel_preprocessor_def("MT_SAMPLE_BUF","__global float *", "mt_sample","mt_sample_p",dE[t-1],dA[t-1]);
            kernel_src0 += kernel_preprocessor_def("MT_SAMPLE_OUTPUT_BUF","__global float *", "mt_sample_output","mt_sample_output_p",dE[t-1],dA[t-1]);
            //cout<< kernel_src0;
            kernel_src0 += kernel_src;
			//cout << kernel_src0;
            size_t kernel_code_size = kernel_src0.length();
            //OpenCL Program
            cl::Program::Sources source(1,std::make_pair(kernel_src0.c_str(),kernel_code_size));
            cl::Program program(contexts[i], source,&ret);
            //kernel build
            ret=program.build(device_ids_USE[i]);
            vector<cl::Kernel> kernels_plat;
            kernels_plat.push_back(cl::Kernel::Kernel(program,"mt_target_transform", &ret));//0
            //cout<<"kernel: "<<ret<<endl;
            kernels_plat.push_back(cl::Kernel::Kernel(program,"mt_sample_transform", &ret));//1
            kernels_plat.push_back(cl::Kernel::Kernel(program,"imageRegistration", &ret));//2
            kernels.push_back(kernels_plat);
            
            queues.push_back(cl::CommandQueue(contexts[i], device_ids_USE[i][j], queue_properties[i], &ret));
			plat_if_of_dev.push_back(i);
            t++;
        }
    }
    
   
    
    /* dark data input / OCL transfer*/
    cout << "Processing dark...\n";
    float *dark_img;
    dark_img = new float[IMAGE_SIZE_M];
    string fileName_dark = fileName_base+ "dark.his";
    readHisFile_stream(fileName_dark,1,30,dark_img,0);
    
    
    /* I0 data input */
    cout << "Processing I0...\n\n";
    vector<float*> I0_imgs;
    float *I0_img_target;
    I0_img_target = new float[IMAGE_SIZE_M];
    for (int i = startEnergyNo; i <= endEnergyNo; i++) {
        I0_imgs.push_back(new float[IMAGE_SIZE_M]);
        string fileName_I0 = fileName_base + EnumTagString(i) + "_I0.his";
        int readHis_err=readHisFile_stream(fileName_I0,1,20,I0_imgs[i-startEnergyNo],0);
        if (readHis_err<0) {
            endEnergyNo=i-1;
            cout <<"error at I0\n";
            break;
        }
    }
        
    /*target numTag settings*/
    string fileName_I0_target;
    fileName_I0_target =  fileName_base + EnumTagString(targetEnergyNo) + "_I0.his";
    readHisFile_stream(fileName_I0_target,1,20,I0_img_target,0);
    
    
    //transfer to opencl buffers
    vector<cl::Buffer> dark_buffers;
    vector<cl::Buffer> I0_target_buffers;
    vector<vector<cl::Buffer>> I0_sample_buffers;
    for (int i=0; i<queues.size(); i++){
        dark_buffers.push_back(cl::Buffer(queues[i].getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_ONLY, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL));
        
        I0_target_buffers.push_back(cl::Buffer(queues[i].getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_ONLY, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL));
        
        //sample I0 buffer
        vector<cl::Buffer> I0_sample_buffers_atE;
        for (int j=startEnergyNo; j<=endEnergyNo; j++) {
            I0_sample_buffers_atE.push_back(cl::Buffer(queues[i].getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_ONLY, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL));
        }
        I0_sample_buffers.push_back(move(I0_sample_buffers_atE));
        
        //write buffer
        queues[i].enqueueWriteBuffer(dark_buffers[i], CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M, dark_img);
        queues[i].enqueueWriteBuffer(I0_target_buffers[i], CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M, I0_img_target);
        for (int j=startEnergyNo; j<=endEnergyNo; j++) {
            queues[i].enqueueWriteBuffer(I0_sample_buffers[i][j-startEnergyNo], CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M, I0_imgs[j-startEnergyNo]);
        }
        queues[i].finish();
    }
    
    delete [] dark_img;
    delete [] I0_img_target;
    for (int j=startEnergyNo; j<=endEnergyNo; j++){
        delete [] I0_imgs[j-startEnergyNo];
    }
    
	vector<thread> th;
    //cout << "queue size: "<<queues.size()<<endl;
    //cout << "thread size: "<<th.size()<<endl;
    //printf("end angle: %d\n",endAngleNo);
    for (int i=startAngleNo; i<=endAngleNo;) {
        //printf("angle: %d\n\n",i);
		for (int j = 0; j<queues.size(); j++) {
            if (i<startAngleNo + queues.size()) {
                bool last = (i+di[j]-1>endAngleNo-queues.size());
                th.push_back(thread(imageReg_thread, queues[j], kernels[j],
                                    dark_buffers[j],
                                    I0_target_buffers[j],
                                    I0_sample_buffers[j],
                                    i,min(i+di[j]-1,endAngleNo),
                                    startEnergyNo,endEnergyNo,targetEnergyNo,
                                    dE[j],fileName_base, output_dir,last));
                this_thread::sleep_for(chrono::seconds((endEnergyNo-startEnergyNo+1)/2));
                i+=di[j];
				if (i > endAngleNo) break;
                else continue;
            }else if (th[j].joinable()) {
                bool last = (i+di[j]-1>endAngleNo-queues.size());
                th[j].join();
				th[j] = thread(imageReg_thread, queues[j], kernels[j],
                               dark_buffers[j],
                               I0_target_buffers[j],
                               I0_sample_buffers[j],
                               i, min(i+di[j]-1,endAngleNo),
                               startEnergyNo,endEnergyNo,targetEnergyNo,
                               dE[j],fileName_base, output_dir,last);
				i+=di[j];
				if (i > endAngleNo) break;
            } else{
                this_thread::sleep_for(chrono::milliseconds(100));
            }

			if (i > endAngleNo) break;
        }

        
    }
    
    for (int j=0; j<queues.size(); j++) {
        if (th[j].joinable()) th[j].join();
    }
    
    return 0;
}