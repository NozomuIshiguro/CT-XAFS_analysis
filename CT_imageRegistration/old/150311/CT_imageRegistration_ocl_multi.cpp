//
//  CT_ImageRegistration_C++.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/02/06.
//  Copyright (c) 2015 Nozomu Ishiguro. All rights reserved.
//

#include "CTXAFS.hpp"
//#include "imageregistration_kernel_src_cl.hpp"

#define  REGMODE 0 //0:XY, 1:XY+rotation, 2:XY+scale, 3:XY+rotation+scale
#include "RegMode.hpp"


string kernel_preprocessor_def(string title, string pointertype,
                               string memobjectname,string pointername,
                               int dA, int num_buffer){
    ostringstream OSS;
    
    OSS<<"#define "<<title<<"_DEF ";
    
    for (int i=0; i<num_buffer; i++) {
        OSS << pointertype << memobjectname<<"_A"<<i<<", ";
    }
    OSS  << pointertype <<" __local *"<< pointername << endl << endl;
    
    OSS<<"#define "<<title<<"_P ";
    int buffer_pnts=dA/num_buffer;
    if (dA%num_buffer>0) buffer_pnts++;
    for (int i=0; i<dA; i++) {
        OSS << pointername;
        OSS << "[" << i <<"]=&";
        OSS <<memobjectname<<"_A"<<(i/buffer_pnts);
        OSS<<"["<<IMAGE_SIZE_M*(i%buffer_pnts)<<"]";
        if(i!=dA-1) OSS<<"; ";
    }
    OSS<<endl<<endl;
    
    return OSS.str();
}


int mt_output_thread(int startAngleNo, int EndAngleNo, int startEnergyNo, int endEnergyNo,
                     string output_dir, vector<float*> mt_outputs,int waittime){
    
    this_thread::sleep_for(chrono::seconds(waittime));
    ostringstream oss;
    for (int j=startAngleNo; j<=EndAngleNo; j++) {
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            string fileName_output= output_dir+ "/"+ EnumTagString(i)+ AnumTagString(j,"/mtr", ".raw");
            oss << "output file: " << fileName_output << "\n";
            outputRawFile_stream(fileName_output,mt_outputs[i-startEnergyNo]+(j-startAngleNo)*IMAGE_SIZE_M);
        }
    }
    oss << "\n";
    cout << oss.str();
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
        delete [] mt_outputs[i-startEnergyNo];
    }
    return 0;
}


int imageReg_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                    vector<cl::Buffer> dark_buffers,
                    vector<cl::Buffer> I0_target_buffers,
                    vector<vector<cl::Buffer>> I0_sample_buffers,
                    int startAngleNo,int EndAngleNo,
                    int startEnergyNo, int endEnergyNo, int targetEnergyNo,
                    string fileName_base, string output_dir,bool last){
    try {
        time_t start_t,readfinish_t;
        time(&start_t);
        
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
        string devicename = device.getInfo<CL_DEVICE_NAME>();
        size_t WorkGroupSize = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), IMAGE_SIZE_X);
        const int dA=EndAngleNo-startAngleNo+1;
        const int num_buf=(int)dark_buffers.size();
        const int buf_pnts=dA/num_buf;
        
        //Loop process for data input
        /*target It data input*/
        float *It_img_target;
        It_img_target = new float[IMAGE_SIZE_M*dA];
        string fileName_It_target = fileName_base + EnumTagString(targetEnergyNo) + ".his";
        readHisFile_stream(fileName_It_target,startAngleNo,EndAngleNo,It_img_target,IMAGE_SIZE_M);
        
        /* Sample It data input */
        vector<float*>It_img;
        //vector<float*> mt_outputs;
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            It_img.push_back(new float[IMAGE_SIZE_M*dA]);
            //mt_outputs.push_back(new float[IMAGE_SIZE_M*dA]);
            string fileName_It = fileName_base + EnumTagString(i) + ".his";
            readHisFile_stream(fileName_It,startAngleNo,EndAngleNo,It_img[i-startEnergyNo],IMAGE_SIZE_M);
        }
        time(&readfinish_t);
        
        
        //Buffer declaration
        vector<cl::Buffer> It2mt_target_buffers;
        vector<vector<cl::Buffer>> It2mt_sample_buffers;
        cl::Buffer transpara_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*transpara_num*dA, 0, NULL);
        for (int k=0; k<num_buf; k++) {
            It2mt_target_buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*buf_pnts, 0, NULL));
            vector<cl::Buffer> It2mt_sample_buffers_atE;
            for (int i=startEnergyNo; i<=endEnergyNo; i++) {
                It2mt_sample_buffers_atE.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*buf_pnts, 0, NULL));
            }
            It2mt_sample_buffers.push_back(move(It2mt_sample_buffers_atE));
        }
        float *transpara;
        transpara = new float[transpara_num*dA];
        
        //kernel dimension declaration
        const cl::NDRange global_item_size(WorkGroupSize*dA,1,1);
        const cl::NDRange local_item_size(WorkGroupSize,1,1);
        
        
        //Energy loop setting
        vector<int> energyNo={startEnergyNo,endEnergyNo};
        vector<int> LoopStartenergyNo;
        LoopStartenergyNo.push_back(min(targetEnergyNo-1, endEnergyNo));
        LoopStartenergyNo.push_back(max(targetEnergyNo+1, startEnergyNo));
        
        
        if (startAngleNo==EndAngleNo) {
            cout << "Device: "<< devicename << ", angle: "<<startAngleNo<< ", processing It...\n\n";
        } else {
            cout << "Device: "<< devicename << ", angle: "<<startAngleNo<<"-"<<EndAngleNo<< "    processing It...\n\n";
        }
        
        
        //It_target dark subtraction
        for (int k=0; k<num_buf; k++) {
            command_queue.enqueueWriteBuffer(It2mt_target_buffers[k], CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M*buf_pnts, It_img_target+IMAGE_SIZE_M*buf_pnts*k,NULL,NULL);
            command_queue.finish();
        }
        int t=0;
        for (int k=0; k<num_buf; k++) {
            kernel[1].setArg(t, dark_buffers[k]);
            t++;
        }
        kernel[1].setArg(t, cl::Local(sizeof(cl_float*)*dA));
        t++;
        for (int k=0; k<num_buf; k++) {
            kernel[1].setArg(t, It2mt_target_buffers[k]);
            t++;
        }
        kernel[1].setArg(t, cl::Local(sizeof(cl_float*)*dA));
        command_queue.enqueueNDRangeKernel(kernel[1], cl::NullRange, global_item_size, local_item_size, NULL, NULL);
        delete [] It_img_target;
        command_queue.finish();
        
        
        //mt_target transform
        t=0;
        for (int k=0; k<num_buf; k++) {
            kernel[2].setArg(t, I0_target_buffers[k]);
            t++;
        }
        kernel[2].setArg(t, cl::Local(sizeof(cl_float*)*dA));
        t++;
        for (int k=0; k<num_buf; k++) {
            kernel[2].setArg(t, It2mt_target_buffers[k]);
            t++;
        }
        kernel[2].setArg(t, cl::Local(sizeof(cl_float*)*dA));
        command_queue.enqueueNDRangeKernel(kernel[2], cl::NullRange, global_item_size, local_item_size, NULL, NULL);
        command_queue.finish();
        
        
        
        //process when (sample Enegry No. == target Energy No.)
        if ((targetEnergyNo>=startEnergyNo)&(targetEnergyNo<=endEnergyNo)) {
            for (int k=0; k<num_buf; k++) {
                command_queue.enqueueCopyBuffer(It2mt_target_buffers[k], It2mt_sample_buffers[k][targetEnergyNo-startEnergyNo], 0, 0, sizeof(cl_float)*IMAGE_SIZE_M*buf_pnts);
                //command_queue.enqueueReadBuffer(It2mt_target_buffers[k], CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M*buf_pnts, mt_outputs[targetEnergyNo-startEnergyNo]+IMAGE_SIZE_M*buf_pnts*k, NULL, NULL);
                command_queue.finish();
            }
            
            ostringstream oss;
            for (int j=startAngleNo; j<=EndAngleNo; j++) {
                oss << "Device: "<< devicename << ", angle: "<<j<< ", energy: "<<targetEnergyNo<<"\n";
                oss <<OSS_TARGET;
            }
            cout << oss.str();
        }
        
        
        //transpara reset
        for (int i=0; i<transpara_num*dA; i++) {
            transpara[i]=0.0f;
        }
        command_queue.enqueueWriteBuffer(transpara_buffer, CL_TRUE, 0, sizeof(cl_float)*transpara_num*dA, transpara, NULL, NULL);
        command_queue.finish();
        
        for (int i=targetEnergyNo+1; i<=endEnergyNo; i++) {
            
            //It_sample dark subtraction
            for (int k=0; k<num_buf; k++) {
                command_queue.enqueueWriteBuffer(It2mt_sample_buffers[k][i-startEnergyNo], CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M*buf_pnts, It_img[i-startEnergyNo]+IMAGE_SIZE_M*buf_pnts*k);
                command_queue.finish();
            }
            t=0;
            for (int k=0; k<num_buf; k++) {
                kernel[4].setArg(t, dark_buffers[k]);
                t++;
            }
            kernel[4].setArg(t, cl::Local(sizeof(cl_float*)*dA));
            t++;
            for (int k=0; k<num_buf; k++) {
                kernel[4].setArg(t, It2mt_sample_buffers[k][i-startEnergyNo]);
                t++;
            }
            kernel[4].setArg(t, cl::Local(sizeof(cl_float*)*dA));
            command_queue.enqueueNDRangeKernel(kernel[4], NULL, global_item_size, local_item_size, NULL, NULL);
            command_queue.finish();
            
            
            //mt_sample transform
            t=0;
            for (int k=0; k<num_buf; k++) {
                kernel[5].setArg(t, I0_sample_buffers[k][i-startEnergyNo]);
                t++;
            }
            kernel[5].setArg(t, cl::Local(sizeof(cl_float*)*dA));
            t++;
            for (int k=0; k<num_buf; k++) {
                kernel[5].setArg(t, It2mt_sample_buffers[k][i-startEnergyNo]);
                t++;
            }
            kernel[5].setArg(t, cl::Local(sizeof(cl_float*)*dA));
            command_queue.enqueueNDRangeKernel(kernel[5], cl::NullRange, global_item_size, local_item_size, NULL, NULL);
            command_queue.finish();
            
            
            //Image registration
            t=0;
            for (int k=0; k<num_buf; k++) {
                kernel[6].setArg(t, It2mt_target_buffers[k]);
                t++;
            }
            kernel[6].setArg(t, cl::Local(sizeof(cl_float*)*dA));
            t++;
            for (int k=0; k<num_buf; k++) {
                kernel[6].setArg(t, It2mt_sample_buffers[k][i-startEnergyNo]);
                t++;
            }
            kernel[6].setArg(t, cl::Local(sizeof(cl_float*)*dA));
            kernel[6].setArg(t+1, transpara_buffer);
            kernel[6].setArg(t+2, cl::Local(sizeof(cl_float)*transpara_num));
            kernel[6].setArg(t+3, cl::Local(sizeof(cl_float)*transpara_num));
            kernel[6].setArg(t+4, cl::Local(sizeof(cl_float)*reductpara_num));
            kernel[6].setArg(t+5, cl::Local(sizeof(cl_float)*WorkGroupSize*reductpara_num));
            kernel[6].setArg(t+6, cl::Local(sizeof(cl_char)*IMAGE_SIZE_X));
            command_queue.enqueueNDRangeKernel(kernel[6], NULL, global_item_size, local_item_size, NULL, NULL);
            command_queue.finish();
            //transpara read buffer
            command_queue.enqueueReadBuffer(transpara_buffer, CL_TRUE, 0, sizeof(cl_float)*transpara_num*dA, transpara, NULL, NULL);
            command_queue.finish();
            
            
            ostringstream oss;
            for (int k=0; k<dA; k++) {
                if (startAngleNo+k>EndAngleNo) {
                    break;
                }
                oss << "Device: "<< devicename << ", angle: "<<startAngleNo+k<< ", energy: "<<i<<"\n";
                OSS_SAMPLE(oss,transpara+transpara_num*k);
            }
            cout << oss.str();
        }
        
        for (int i=targetEnergyNo; i>=startEnergyNo; i--) {
            
            //It_sample dark subtraction
            for (int k=0; k<num_buf; k++) {
                command_queue.enqueueWriteBuffer(It2mt_sample_buffers[k][i-startEnergyNo], CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M*buf_pnts, It_img[i-startEnergyNo]+IMAGE_SIZE_M*buf_pnts*k);
                command_queue.finish();
            }
            t=0;
            for (int k=0; k<num_buf; k++) {
                kernel[4].setArg(t, dark_buffers[k]);
                t++;
            }
            kernel[4].setArg(t, cl::Local(sizeof(cl_float*)*dA));
            t++;
            for (int k=0; k<num_buf; k++) {
                kernel[4].setArg(t, It2mt_sample_buffers[k][i-startEnergyNo]);
                t++;
            }
            kernel[4].setArg(t, cl::Local(sizeof(cl_float*)*dA));
            command_queue.enqueueNDRangeKernel(kernel[4], NULL, global_item_size, local_item_size, NULL, NULL);
            command_queue.finish();
            
            
            //mt_sample transform
            t=0;
            for (int k=0; k<num_buf; k++) {
                kernel[5].setArg(t, I0_sample_buffers[k][i-startEnergyNo]);
                t++;
            }
            kernel[5].setArg(t, cl::Local(sizeof(cl_float*)*dA));
            t++;
            for (int k=0; k<num_buf; k++) {
                kernel[5].setArg(t, It2mt_sample_buffers[k][i-startEnergyNo]);
                t++;
            }
            kernel[5].setArg(t, cl::Local(sizeof(cl_float*)*dA));
            command_queue.enqueueNDRangeKernel(kernel[5], cl::NullRange, global_item_size, local_item_size, NULL, NULL);
            command_queue.finish();
            
            
            //Image registration
            t=0;
            for (int k=0; k<num_buf; k++) {
                kernel[6].setArg(t, It2mt_target_buffers[k]);
                t++;
            }
            kernel[6].setArg(t, cl::Local(sizeof(cl_float*)*dA));
            t++;
            for (int k=0; k<num_buf; k++) {
                kernel[6].setArg(t, It2mt_sample_buffers[k][i-startEnergyNo]);
                t++;
            }
            kernel[6].setArg(t, cl::Local(sizeof(cl_float*)*dA));
            kernel[6].setArg(t+1, transpara_buffer);
            kernel[6].setArg(t+2, cl::Local(sizeof(cl_float)*transpara_num));
            kernel[6].setArg(t+3, cl::Local(sizeof(cl_float)*transpara_num));
            kernel[6].setArg(t+4, cl::Local(sizeof(cl_float)*reductpara_num));
            kernel[6].setArg(t+5, cl::Local(sizeof(cl_float)*WorkGroupSize*reductpara_num));
            kernel[6].setArg(t+6, cl::Local(sizeof(cl_char)*IMAGE_SIZE_X));
            command_queue.enqueueNDRangeKernel(kernel[6], NULL, global_item_size, local_item_size, NULL, NULL);
            command_queue.finish();
            //transpara read buffer
            command_queue.enqueueReadBuffer(transpara_buffer, CL_TRUE, 0, sizeof(cl_float)*transpara_num*dA, transpara, NULL, NULL);
            command_queue.finish();
            
            
            ostringstream oss;
            for (int k=0; k<dA; k++) {
                if (startAngleNo+k>EndAngleNo) {
                    break;
                }
                oss << "Device: "<< devicename << ", angle: "<<startAngleNo+k<< ", energy: "<<i<<"\n";
                OSS_SAMPLE(oss,transpara+transpara_num*k);
            }
            cout << oss.str();
        }
        
        //transpara reset
        for (int i=0; i<transpara_num*dA; i++) {
            transpara[i]=0.0f;
        }
        command_queue.enqueueWriteBuffer(transpara_buffer, CL_TRUE, 0, sizeof(cl_float)*transpara_num*dA, transpara, NULL, NULL);
        command_queue.finish();
        
        delete [] transpara;
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            //output
            for (int k=0; k<num_buf; k++) {
                command_queue.enqueueReadBuffer(It2mt_sample_buffers[k][i-startEnergyNo], CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M*buf_pnts, It_img[i-startEnergyNo]+IMAGE_SIZE_M*buf_pnts*k, NULL, NULL);
                command_queue.finish();
            }
        }
        
        int delta_t;
        if (last) {
            delta_t=0;
        }else{
            delta_t= difftime(readfinish_t,start_t)*1.5;
        }
        
        thread th_output(mt_output_thread,
                         startAngleNo,EndAngleNo,startEnergyNo,endEnergyNo,
                         output_dir,It_img,delta_t);
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
    vector<int> dA,num_buf;
   
    //kernel source from cl file
    ifstream ifs("./imageregistration_kernel_src.cl",ios::in);
    if(!ifs) {
        cerr << "   Failed to load kernel \n\n" << endl;
        return -1;
    }
    istreambuf_iterator<char> it(ifs);
    istreambuf_iterator<char> last;
    string kernel_src(it,last);
    ifs.close();
    
    int t=1;
    for (int i=0; i<device_ids_USE.size(); i++) {
        
        
        /*OpenCL Context create*/
        cl_context_properties properties[]={CL_CONTEXT_PLATFORM, (cl_context_properties)(platform_ids_USE[i])(), 0};
        contexts.push_back(cl::Context(device_ids_USE[i], properties,NULL,NULL,&ret));
        
    
        for (int j=0; j<device_ids_USE[i].size(); j++) {
            cout<<"Device No. "<<t<<endl;
            string platform_param;
            platform_ids_USE[i].getInfo(CL_PLATFORM_NAME, &platform_param);
            cout << "CL PLATFORM NAME: "<< platform_param<<endl;
            platform_ids_USE[i].getInfo(CL_PLATFORM_VERSION, &platform_param);
            cout << "   "<<platform_param<<endl;
            string device_pram;
            ret = device_ids_USE[i][j].getInfo(CL_DEVICE_NAME, &device_pram);
            cout << "CL DEVICE NAME: "<< device_pram<<endl;
            
            /*Open CL command que create*/
            queue_properties.push_back(0);
            cl_uint computeUnitSize;
            cl_ulong maxAllocSize, globalMemSize;
            float globalMemSafeLimit=0.5;
            ret = device_ids_USE[i][j].getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &computeUnitSize);
            ret = device_ids_USE[i][j].getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &maxAllocSize);
            ret = device_ids_USE[i][j].getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &globalMemSize);
            cl_uint dA_estimate = computeUnitSize;
            cl_uint num_buff_estimate = 1;
            do {
                cl_ulong imageRegGlobalMemSize = dA_estimate*sizeof(float)*(IMAGE_SIZE_M*2+transpara_num);
                //cout<<dA_estimate<<endl;
                if((float)imageRegGlobalMemSize<(float)globalMemSize*globalMemSafeLimit) break;
                else dA_estimate--;
            } while (dA_estimate>0);
            do {
                size_t imageBufferSize = dA_estimate*sizeof(float)*IMAGE_SIZE_M/num_buff_estimate;
                if (imageBufferSize < maxAllocSize) break;
                else num_buff_estimate++;
            } while (num_buff_estimate<=dA_estimate);
            globalMemSafeLimit*=100.0;
            dA.push_back(4);
            num_buf.push_back(1);
            //cout<<"Safty limit of VRAM: "<<globalMemSafeLimit <<"%"<<endl;
            cout<<"Number of working compute unit: "<<dA[t-1]<<endl;
            cout<<"Number of image buffer separation: "<<num_buf[t-1]<<endl<<endl;;
            
            
            
            //OpenCL Program
            string kernel_code="";
            kernel_code += kernel_preprocessor_def("DARK", "__global float *",
                                                   "dark","dark_p",dA[t-1],num_buf[t-1]);
            kernel_code += kernel_preprocessor_def("I", "__global float *",
                                                   "I","I_p",dA[t-1],num_buf[t-1]);
            kernel_code += kernel_preprocessor_def("I0", "__global float *",
                                                   "I0","I0_p",dA[t-1],num_buf[t-1]);
            kernel_code += kernel_preprocessor_def("IT2MT", "__global float *",
                                                   "It2mt","It2mt_p",dA[t-1],num_buf[t-1]);
            kernel_code += kernel_preprocessor_def("MT_TARGET", "__global float *",
                                                   "mt_target","mt_target_p",dA[t-1],num_buf[t-1]);
            kernel_code += kernel_preprocessor_def("MT_SAMPLE", "__global float *",
                                                   "mt_sample","mt_sample_p",dA[t-1],num_buf[t-1]);
            //cout << kernel_code<<endl;
            kernel_code += kernel_src;
            size_t kernel_code_size = kernel_code.length();
            cl::Program::Sources source(1,std::make_pair(kernel_code.c_str(),kernel_code_size));
            cl::Program program(contexts[i], source,&ret);
            //kernel build
            ret=program.build(device_ids_USE[i]);
            vector<cl::Kernel> kernels_plat;
            kernels_plat.push_back(cl::Kernel::Kernel(program,"dark_subtraction", &ret));//0
            kernels_plat.push_back(cl::Kernel::Kernel(program,"dark_subtraction", &ret));//1
            kernels_plat.push_back(cl::Kernel::Kernel(program,"mt_transform", &ret));//2
            kernels_plat.push_back(cl::Kernel::Kernel(program,"dark_subtraction", &ret));//3
            kernels_plat.push_back(cl::Kernel::Kernel(program,"dark_subtraction", &ret));//4
            kernels_plat.push_back(cl::Kernel::Kernel(program,"mt_transform", &ret));//5
            kernels_plat.push_back(cl::Kernel::Kernel(program,"imageRegistration", &ret));//6
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
    
    
    // I0 sample data input
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
    
        
    // I0 target data input
    string fileName_I0_target;
    fileName_I0_target =  fileName_base + EnumTagString(targetEnergyNo) + "_I0.his";
    readHisFile_stream(fileName_I0_target,1,20,I0_img_target,0);
    
    
    //transfer to opencl buffers
    vector<vector<cl::Buffer>> dark_buffers;
    vector<vector<cl::Buffer>> I0_target_buffers;
    vector<vector<vector<cl::Buffer>>> I0_sample_buffers;
    for (int i=0; i<queues.size(); i++){
        int buffer_pnts = dA[i]/num_buf[i];
        
        vector<cl::Buffer> dark_buffers_atQ;
        vector<cl::Buffer> I0_target_buffers_atQ;
        vector<vector<cl::Buffer>> I0_sample_buffers_atQ;
        for (int k=0; k<num_buf[i]; k++) {
            dark_buffers_atQ.push_back(cl::Buffer(queues[i].getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_ONLY, sizeof(cl_float)*IMAGE_SIZE_M*buffer_pnts, 0, NULL));
            
            I0_target_buffers_atQ.push_back(cl::Buffer(queues[i].getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*buffer_pnts, 0, NULL));
            
            vector<cl::Buffer> I0_sample_buffers_atE;
            for (int j=startEnergyNo; j<=endEnergyNo; j++) {
                I0_sample_buffers_atE.push_back(cl::Buffer(queues[i].getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*buffer_pnts, 0, NULL));
            }
            I0_sample_buffers_atQ.push_back(move(I0_sample_buffers_atE));
        }
        dark_buffers.push_back(move(dark_buffers_atQ));
        I0_target_buffers.push_back(move(I0_target_buffers_atQ));
        I0_sample_buffers.push_back(move(I0_sample_buffers_atQ));
        
        
        //write buffer
        for (int k=0; k<dA[i]; k++) {
            queues[i].enqueueWriteBuffer(dark_buffers[i][k/buffer_pnts], CL_TRUE, sizeof(cl_float)*IMAGE_SIZE_M*(k%buffer_pnts), sizeof(cl_float)*IMAGE_SIZE_M, dark_img);
            queues[i].finish();
            queues[i].enqueueWriteBuffer(I0_target_buffers[i][k/buffer_pnts], CL_TRUE, sizeof(cl_float)*IMAGE_SIZE_M*(k%buffer_pnts), sizeof(cl_float)*IMAGE_SIZE_M, I0_img_target);
            queues[i].finish();
            for (int j=startEnergyNo; j<=endEnergyNo; j++) {
                queues[i].enqueueWriteBuffer(I0_sample_buffers[i][k/buffer_pnts][j-startEnergyNo], CL_TRUE, sizeof(cl_float)*IMAGE_SIZE_M*(k%buffer_pnts), sizeof(cl_float)*IMAGE_SIZE_M, I0_imgs[j-startEnergyNo]);
                queues[i].finish();
            }
        }
        
        //kernel dimension declaration
        size_t WorkGroupSize = min((int)queues[i].getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), IMAGE_SIZE_X);
        const cl::NDRange global_item_size(WorkGroupSize*dA[i],1,1);
        const cl::NDRange local_item_size(WorkGroupSize,1,1);
        
        //I0_target dark subtraction
        int t=0;
        for (int k=0; k<num_buf[i]; k++) {
            kernels[i][0].setArg(t, dark_buffers[i][k]);
            t++;
        }
        kernels[i][0].setArg(t, cl::Local(sizeof(cl_float*)*dA[i]));
        t++;
        for (int k=0; k<num_buf[i]; k++) {
            kernels[i][0].setArg(t, I0_target_buffers[i][k]);
            t++;
        }
        kernels[i][0].setArg(t, cl::Local(sizeof(cl_float*)*dA[i]));
        queues[i].enqueueNDRangeKernel(kernels[i][0], cl::NullRange, global_item_size, local_item_size, NULL, NULL);
        queues[i].finish();
        
        //I0_sample dark subtraction
        for (int j=startEnergyNo; j<=endEnergyNo; j++) {
            int t=0;
            for (int k=0; k<num_buf[i]; k++) {
                kernels[i][3].setArg(t, dark_buffers[i][k]);
                t++;
            }
            kernels[i][3].setArg(t, cl::Local(sizeof(cl_float*)*dA[i]));
            t++;
            for (int k=0; k<num_buf[i]; k++) {
                kernels[i][3].setArg(t, I0_sample_buffers[i][k][j-startEnergyNo]);
                t++;
            }
            kernels[i][3].setArg(t, cl::Local(sizeof(cl_float*)*dA[i]));
            queues[i].enqueueNDRangeKernel(kernels[i][3], cl::NullRange, global_item_size, local_item_size, NULL, NULL);
            queues[i].finish();
        }
    }
    
    delete [] dark_img;
    delete [] I0_img_target;
    for (int j=startEnergyNo; j<=endEnergyNo; j++){
        delete [] I0_imgs[j-startEnergyNo];
    }
    
	vector<thread> th;
    //printf("end angle: %d\n",endAngleNo);
    for (int i=startAngleNo; i<=endAngleNo;) {
        //printf("angle: %d\n\n",i);
		for (int j = 0; j<queues.size(); j++) {
            
            
            
            if (i<startAngleNo + queues.size()) {
                bool last = (i+dA[j]-1>endAngleNo-queues.size());
                th.push_back(thread(imageReg_thread, queues[j], kernels[j],
                                    dark_buffers[j],
                                    I0_target_buffers[j],
                                    I0_sample_buffers[j],
                                    i,min(i+dA[j]-1,endAngleNo),
                                    startEnergyNo,endEnergyNo,targetEnergyNo,
                                    fileName_base,output_dir,last));
                this_thread::sleep_for(chrono::seconds((endEnergyNo-startEnergyNo+1)/2));
                i+=dA[j];
				if (i > endAngleNo) break;
                else continue;
            }else if (th[j].joinable()) {
                bool last = (i+dA[j]-1>endAngleNo-queues.size());
                th[j].join();
				th[j] = thread(imageReg_thread, queues[j], kernels[j],
                               dark_buffers[j],
                               I0_target_buffers[j],
                               I0_sample_buffers[j],
                               i, min(i+dA[j]-1,endAngleNo),
                               startEnergyNo,endEnergyNo,targetEnergyNo,
                               fileName_base,output_dir,last);
				i+=dA[j];
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