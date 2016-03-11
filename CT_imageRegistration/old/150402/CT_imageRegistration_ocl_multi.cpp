//
//  CT_ImageRegistration_C++.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/02/06.
//  Copyright (c) 2015 Nozomu Ishiguro. All rights reserved.
//

#include "CTXAFS.hpp"
//#include "imageregistration_kernel_src_cl.hpp"
//#include "XYshift_cl.hpp"
//#include "XYshift_rotation_cl.hpp"

#define  REGMODE 0 //0:XY, 1:XY+rotation, 2:XY+scale, 3:XY+rotation+scale
#include "RegMode.hpp"

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
                    float *dark_img,
                    float *I0_img_target,
                    vector<float*> I0_imgs,
                    int startAngleNo,int EndAngleNo, int num_buf,
                    int startEnergyNo, int endEnergyNo, int targetEnergyNo,
					string fileName_base, string output_dir, bool last){
    try {
        time_t start_t,readfinish_t;
        time(&start_t);
        
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
        string devicename = device.getInfo<CL_DEVICE_NAME>();
        size_t WorkGroupSize = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), IMAGE_SIZE_X);
        const int dA=EndAngleNo-startAngleNo+1;
        //const int num_buf=(int)dark_buffers.size();
        const int buf_pnts=dA/num_buf;
        
        //Loop process for data input
        /*target It data input*/
        float *It_img_target;
        It_img_target = new float[IMAGE_SIZE_M*dA];
        string fileName_It_target = fileName_base + EnumTagString(targetEnergyNo) + ".his";
        readHisFile_stream(fileName_It_target,startAngleNo,EndAngleNo,It_img_target,IMAGE_SIZE_M);
        
        /* Sample It data input */
        vector<float*>sample_img;
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            sample_img.push_back(new float[IMAGE_SIZE_M*dA]);
            string fileName_It = fileName_base + EnumTagString(i) + ".his";
            readHisFile_stream(fileName_It,startAngleNo,EndAngleNo,sample_img[i-startEnergyNo],IMAGE_SIZE_M);
        }
        time(&readfinish_t);
        
        //Buffer declaration
		vector<cl::Buffer> dark_buffers;
		vector<cl::Buffer> I0_target_buffers;
		vector<cl::Buffer> It2mt_target_buffers;
		vector<cl::Buffer> I0_sample_buffers;
		vector<cl::Buffer> It2mt_sample_buffers;
		vector<cl::Buffer> mt_output_buffers;
        cl::Buffer transpara_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*transpara_num*dA, 0, NULL);
        cl::Buffer transpara_err_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*transpara_num*dA, 0, NULL);
        for (int k=0; k<num_buf; k++) {
            dark_buffers.push_back(cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(cl_float)*IMAGE_SIZE_M*buf_pnts, 0, NULL));
			I0_target_buffers.push_back(cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(cl_float)*IMAGE_SIZE_M*buf_pnts, 0, NULL));
			It2mt_target_buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*buf_pnts, 0, NULL));
			I0_sample_buffers.push_back(cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(cl_float)*IMAGE_SIZE_M*buf_pnts, 0, NULL));
			It2mt_sample_buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*buf_pnts, 0, NULL));
			mt_output_buffers.push_back(cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*IMAGE_SIZE_M*buf_pnts, 0, NULL));
        }
        float *transpara;
        transpara = new float[transpara_num*dA];
        float *transpara_err;
        transpara_err = new float[transpara_num*dA];
        
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
        
		//write dark buffer
		for (int k = 0; k<dA; k++) {
			command_queue.enqueueWriteBuffer(dark_buffers[k / buf_pnts], CL_TRUE, sizeof(cl_float)*IMAGE_SIZE_M*(k%buf_pnts), sizeof(cl_float)*IMAGE_SIZE_M, dark_img, NULL, NULL);
			command_queue.finish();
		}

		//I0_target dark subtraction
		for (int k = 0; k<dA; k++) {
			command_queue.enqueueWriteBuffer(I0_target_buffers[k / buf_pnts], CL_TRUE, sizeof(cl_float)*IMAGE_SIZE_M*(k%buf_pnts), sizeof(cl_float)*IMAGE_SIZE_M, I0_img_target, NULL, NULL);
			command_queue.finish();
		}
		int t = 0;
		for (int k = 0; k<num_buf; k++) {
			kernel[0].setArg(t, dark_buffers[k]);
			t++;
		}
		//kernel[0].setArg(t, cl::Local(sizeof(cl_float*)*dA));
		//t++;
		for (int k = 0; k<num_buf; k++) {
			kernel[0].setArg(t, I0_target_buffers[k]);
			t++;
		}
		//kernel[0].setArg(t, cl::Local(sizeof(cl_float*)*dA));
		command_queue.enqueueNDRangeKernel(kernel[0], cl::NullRange, global_item_size, local_item_size, NULL, NULL);
		command_queue.finish();
        
        //It_target dark subtraction
        for (int k=0; k<num_buf; k++) {
            command_queue.enqueueWriteBuffer(It2mt_target_buffers[k], CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M*buf_pnts, It_img_target+IMAGE_SIZE_M*buf_pnts*k,NULL,NULL);
            command_queue.finish();
        }
        t=0;
        for (int k=0; k<num_buf; k++) {
            kernel[1].setArg(t, dark_buffers[k]);
            t++;
        }
        //kernel[1].setArg(t, cl::Local(sizeof(cl_float*)*dA));
        //t++;
        for (int k=0; k<num_buf; k++) {
            kernel[1].setArg(t, It2mt_target_buffers[k]);
            t++;
        }
        //kernel[1].setArg(t, cl::Local(sizeof(cl_float*)*dA));
        command_queue.enqueueNDRangeKernel(kernel[1], cl::NullRange, global_item_size, local_item_size, NULL, NULL);
        delete [] It_img_target;
        command_queue.finish();
        
        
        //mt_target transform
        t=0;
        for (int k=0; k<num_buf; k++) {
            kernel[2].setArg(t, I0_target_buffers[k]);
            t++;
        }
        //kernel[2].setArg(t, cl::Local(sizeof(cl_float*)*dA));
        //t++;
        for (int k=0; k<num_buf; k++) {
            kernel[2].setArg(t, It2mt_target_buffers[k]);
            t++;
        }
        //kernel[2].setArg(t, cl::Local(sizeof(cl_float*)*dA));
        command_queue.enqueueNDRangeKernel(kernel[2], cl::NullRange, global_item_size, local_item_size, NULL, NULL);
        command_queue.finish();
        
        
        
        //process when (sample Enegry No. == target Energy No.)
        if ((targetEnergyNo>=startEnergyNo)&(targetEnergyNo<=endEnergyNo)) {
            for (int k=0; k<num_buf; k++) {
				command_queue.enqueueReadBuffer(It2mt_target_buffers[k], CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M*buf_pnts, sample_img[targetEnergyNo - startEnergyNo] + IMAGE_SIZE_M*buf_pnts*k, NULL, NULL);
                command_queue.finish();
            }
            
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
            
            //transpara reset
            for (int i=0; i<transpara_num*dA; i++) {
                transpara[i]=0.0f;
            }
            command_queue.enqueueWriteBuffer(transpara_buffer, CL_TRUE, 0, sizeof(cl_float)*transpara_num*dA, transpara, NULL, NULL);
            command_queue.finish();
            
            for (int i=LoopStartenergyNo[s]+di; i*di<=energyNo[s]*di; i+=di) {
                
				//I0_sample dark subtraction
				for (int k = 0; k<dA; k++) {
					command_queue.enqueueWriteBuffer(I0_sample_buffers[k / buf_pnts], CL_TRUE, sizeof(cl_float)*IMAGE_SIZE_M*(k%buf_pnts), sizeof(cl_float)*IMAGE_SIZE_M, I0_imgs[i - startEnergyNo]);
					command_queue.finish();
				}
				t = 0;
				for (int k = 0; k<num_buf; k++) {
					kernel[3].setArg(t, dark_buffers[k]);
					t++;
				}
				//kernel[3].setArg(t, cl::Local(sizeof(cl_float*)*dA));
				//t++;
				for (int k = 0; k<num_buf; k++) {
					kernel[3].setArg(t, I0_sample_buffers[k]);
					t++;
				}
				//kernel[3].setArg(t, cl::Local(sizeof(cl_float*)*dA));
				command_queue.enqueueNDRangeKernel(kernel[3], NULL, global_item_size, local_item_size, NULL, NULL);
				command_queue.finish();
                
                //It_sample dark subtraction
                for (int k=0; k<num_buf; k++) {
                    command_queue.enqueueWriteBuffer(It2mt_sample_buffers[k], CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M*buf_pnts, sample_img[i-startEnergyNo]+IMAGE_SIZE_M*buf_pnts*k);
                    command_queue.finish();
                }
                t=0;
                for (int k=0; k<num_buf; k++) {
                    kernel[4].setArg(t, dark_buffers[k]);
                    t++;
                }
                //kernel[4].setArg(t, cl::Local(sizeof(cl_float*)*dA));
                //t++;
                for (int k=0; k<num_buf; k++) {
                    kernel[4].setArg(t, It2mt_sample_buffers[k]);
                    t++;
                }
                //kernel[4].setArg(t, cl::Local(sizeof(cl_float*)*dA));
                command_queue.enqueueNDRangeKernel(kernel[4], NULL, global_item_size, local_item_size, NULL, NULL);
                command_queue.finish();
                
                
                //mt_sample transform
                t=0;
                for (int k=0; k<num_buf; k++) {
                    kernel[5].setArg(t, I0_sample_buffers[k]);
                    t++;
                }
                //kernel[5].setArg(t, cl::Local(sizeof(cl_float*)*dA));
                //t++;
                for (int k=0; k<num_buf; k++) {
                    kernel[5].setArg(t, It2mt_sample_buffers[k]);
                    t++;
                }
                //kernel[5].setArg(t, cl::Local(sizeof(cl_float*)*dA));
                command_queue.enqueueNDRangeKernel(kernel[5], cl::NullRange, global_item_size, local_item_size, NULL, NULL);
                command_queue.finish();
                
                //Image registration
                t=0;
                for (int k=0; k<num_buf; k++) {
                    kernel[6].setArg(t, It2mt_target_buffers[k]);
                    t++;
                }
                //kernel[6].setArg(t, cl::Local(sizeof(cl_float*)*dA));
                //t++;
                for (int k=0; k<num_buf; k++) {
                    kernel[6].setArg(t, It2mt_sample_buffers[k]);
                    t++;
                }
                //kernel[6].setArg(t, cl::Local(sizeof(cl_float*)*dA));
				//t++;
				for (int k = 0; k<num_buf; k++) {
					kernel[6].setArg(t, mt_output_buffers[k]);
					t++;
				}
				//kernel[6].setArg(t, cl::Local(sizeof(cl_float*)*dA));
                kernel[6].setArg(t, transpara_buffer);
                kernel[6].setArg(t+1, transpara_err_buffer);
                kernel[6].setArg(t+2, cl::Local(sizeof(cl_float)*transpara_num));
                kernel[6].setArg(t+3, cl::Local(sizeof(cl_float)*transpara_num));
                kernel[6].setArg(t+4, cl::Local(sizeof(cl_float)*reductpara_num));
                kernel[6].setArg(t+5, cl::Local(sizeof(cl_float)*WorkGroupSize*reductpara_num));
				//kernel[6].setArg(t+5, loc_mem);
				kernel[6].setArg(t+6, cl::Local(sizeof(cl_char)*IMAGE_SIZE_X));
                command_queue.enqueueNDRangeKernel(kernel[6], NULL, global_item_size, local_item_size, NULL, NULL);
                command_queue.finish();
                //transpara read buffer
                command_queue.enqueueReadBuffer(transpara_buffer, CL_TRUE, 0, sizeof(cl_float)*transpara_num*dA, transpara, NULL, NULL);
                command_queue.enqueueReadBuffer(transpara_err_buffer, CL_TRUE, 0, sizeof(cl_float)*transpara_num*dA, transpara_err, NULL, NULL);
                command_queue.finish();
                
                ostringstream oss;
                for (int k=0; k<dA; k++) {
                    if (startAngleNo+k>EndAngleNo) {
                        break;
                    }
                    int precision_err[transpara_num];
                    for (int n=0; n<transpara_num; n++) {
                        if (log10(transpara_err[n])>0) {
                            precision_err[n]=1;
                        }else{
                            precision_err[n]=(floor(log10(abs(transpara[n])))
                                              -floor(log10(abs(transpara_err[n])))+1);
                        }
                    }
                    oss << "Device: "<< devicename << ", angle: "<<startAngleNo+k<< ", energy: "<<i<<"\n";
                    OSS_SAMPLE(oss,transpara+transpara_num*k, transpara_err+transpara_num*k,precision_err);
                }
                cout << oss.str();

				//read output buffer
				for (int k = 0; k<num_buf; k++) {
					command_queue.enqueueReadBuffer(mt_output_buffers[k], CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M*buf_pnts, sample_img[i - startEnergyNo] + IMAGE_SIZE_M*buf_pnts*k, NULL, NULL);
					command_queue.finish();
				}
				
            }
        }
        
        delete [] transpara;
        
        int delta_t;
        if (last) {
            delta_t=0;
        }else{
            delta_t= difftime(readfinish_t,start_t)*0.4;
        }
        
        thread th_output(mt_output_thread,
                         startAngleNo,EndAngleNo,startEnergyNo,endEnergyNo,
                         output_dir,sample_img,delta_t);
        if(last) th_output.join();
        else th_output.detach();

    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
    
    
    return 0;
}




int imageRegistlation_ocl(string fileName_base, input_parameter inp,
                          OCL_platform_device plat_dev_list)
{
    cl_int ret;
    
    int startEnergyNo=inp.getStartEnergyNo();
    int endEnergyNo=inp.getEndEnergyNo();
    int targetEnergyNo=inp.getTargetEnergyNo();
    int startAngleNo=inp.getStartAngleNo();
    int endAngleNo=inp.getEndAngleNo();
    
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
        string fileName_output = inp.getOutputDir() + "/" + EnumTagString(i);
        MKDIR(fileName_output.c_str());
    }
    
    
    vector<vector<cl::Kernel>> kernels;
    vector<cl::CommandQueue> queues;
    vector<int> dA,num_buf;
   
    //kernel source from cl file
	ifstream ifs("./imageregistration_kernel_src.cl", ios::in);
    if(!ifs) {
        cerr << "   Failed to load kernel \n" << endl;
        return -1;
    }
    istreambuf_iterator<char> it(ifs);
    istreambuf_iterator<char> last;
    string kernel_src(it,last);
    ifs.close();
    //kernel preprocessor source (XY) from cl file
    ifstream ifs2("./XYshift.cl",ios::in);
    if(!ifs2) {
        cerr << "   Failed to load kernel \n\n" << endl;
        return -1;
    }
    istreambuf_iterator<char> it2(ifs2);
    istreambuf_iterator<char> last2;
    string kernel_src_XY(it2,last2);
    ifs2.close();
    //kernel preprocessor source (XY) from cl file
    ifstream ifs3("./XYshift_rotation.cl",ios::in);
    if(!ifs3) {
        cerr << "   Failed to load kernel \n\n" << endl;
        return -1;
    }
    istreambuf_iterator<char> it3(ifs3);
    istreambuf_iterator<char> last3;
    string kernel_src_XYrot(it3,last3);
    ifs3.close();
    
    for (int i=0; i<plat_dev_list.size(); i++) {

        cout<<"Device No. "<<i<<endl;
        string platform_param;
        plat_dev_list.plat(i).getInfo(CL_PLATFORM_NAME, &platform_param);
        cout << "CL PLATFORM NAME: "<< platform_param<<endl;
        plat_dev_list.plat(i).getInfo(CL_PLATFORM_VERSION, &platform_param);
        cout << "   "<<platform_param<<endl;
        string device_pram;
        plat_dev_list.dev(i).getInfo(CL_DEVICE_NAME, &device_pram);
        cout << "CL DEVICE NAME: "<< device_pram<<endl;
            
        /*Open CL command que create*/
        cl_uint computeUnitSize;
        cl_ulong maxAllocSize, globalMemSize;
        float globalMemSafeLimit=0.5;
        ret = plat_dev_list.dev(i).getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &computeUnitSize);
        ret = plat_dev_list.dev(i).getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &maxAllocSize);
        ret = plat_dev_list.dev(i).getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &globalMemSize);
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
        dA.push_back(8);
        num_buf.push_back(1);
        //cout<<"Safty limit of VRAM: "<<globalMemSafeLimit <<"%"<<endl;
        cout<<"Number of working compute unit: "<<dA[i]<<endl;
        cout<<"Number of image buffer separation: "<<num_buf[i]<<endl<<endl;;
            
            
            
        //OpenCL Program
        string kernel_code="";
        kernel_code += kernel_preprocessor_def("DARK", "__global float *",
                                               "dark","dark_p",dA[i],num_buf[i],IMAGE_SIZE_M);
        kernel_code += kernel_preprocessor_def("I", "__global float *",
                                               "I","I_p",dA[i],num_buf[i],IMAGE_SIZE_M);
        kernel_code += kernel_preprocessor_def("I0", "__global float *",
                                                   "I0","I0_p",dA[i],num_buf[i],IMAGE_SIZE_M);
        kernel_code += kernel_preprocessor_def("IT2MT", "__global float *",
                                                   "It2mt","It2mt_p",dA[i],num_buf[i],IMAGE_SIZE_M);
        kernel_code += kernel_preprocessor_def("MT_TARGET", "__global float *",
                                                   "mt_target","mt_target_p",dA[i],num_buf[i],IMAGE_SIZE_M);
        kernel_code += kernel_preprocessor_def("MT_SAMPLE", "__global float *",
                                                   "mt_sample","mt_sample_p",dA[i],num_buf[i],IMAGE_SIZE_M);
		kernel_code += kernel_preprocessor_def("MT_OUTPUT", "__global float *",
			"mt_output", "mt_output_p", dA[i], num_buf[i], IMAGE_SIZE_M);
        switch (REGMODE) {
            case 0: //XY
                kernel_code += kernel_src_XY;
                break;
            
            case 1: //XY+rotation
                kernel_code += kernel_src_XYrot;
                break;
                
            case 2: //XY+scale
                
                break;
                
            case 3: //XY+rotation+scale
                
                break;
                
            default:
                break;
        }
        //cout << kernel_code<<endl;
		kernel_code += kernel_src;
        size_t kernel_code_size = kernel_code.length();
        cl::Program::Sources source(1,std::make_pair(kernel_code.c_str(),kernel_code_size));
        cl::Program program(plat_dev_list.context(i), source,&ret);
		//cout << kernel_code<<endl;
        //kernel build
        ret=program.build();
        vector<cl::Kernel> kernels_plat;
        kernels_plat.push_back(cl::Kernel::Kernel(program,"dark_subtraction", &ret));//0
        kernels_plat.push_back(cl::Kernel::Kernel(program,"dark_subtraction", &ret));//1
        kernels_plat.push_back(cl::Kernel::Kernel(program,"mt_transform", &ret));//2
        kernels_plat.push_back(cl::Kernel::Kernel(program,"dark_subtraction", &ret));//3
        kernels_plat.push_back(cl::Kernel::Kernel(program,"dark_subtraction", &ret));//4
        kernels_plat.push_back(cl::Kernel::Kernel(program,"mt_transform", &ret));//5
        kernels_plat.push_back(cl::Kernel::Kernel(program,"imageRegistration", &ret));//6
        kernels.push_back(kernels_plat);
        
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
    
    
	vector<thread> th;
    for (int i=startAngleNo; i<=endAngleNo;) {
		for (int j = 0; j<plat_dev_list.size(); j++) {
            
            
			if (th.size()<plat_dev_list.size()) {
				bool last = (i + dA[j] - 1>endAngleNo - plat_dev_list.size()*dA[j]);
                th.push_back(thread(imageReg_thread, plat_dev_list.queue(j), kernels[j],
                                    dark_img,
                                    I0_img_target,
                                    I0_imgs,
									i, min(i + dA[j] - 1, endAngleNo), num_buf[j],
                                    startEnergyNo,endEnergyNo,targetEnergyNo,
									fileName_base, inp.getOutputDir(), last));
                this_thread::sleep_for(chrono::seconds((int)((endEnergyNo-startEnergyNo+1)*dA[j]*0.02)));
				//this_thread::sleep_for(chrono::seconds(40));
                i+=dA[j];
				if (i > endAngleNo) break;
                else continue;
            }else if (th[j].joinable()) {
				bool last = (i + dA[j] - 1>endAngleNo - plat_dev_list.size()*dA[j]);
                th[j].join();
				th[j] = thread(imageReg_thread, plat_dev_list.queue(j), kernels[j],
								dark_img,
								I0_img_target,
								I0_imgs,
								i, min(i + dA[j] - 1, endAngleNo), num_buf[j],
                               startEnergyNo,endEnergyNo,targetEnergyNo,
							   fileName_base, inp.getOutputDir(), last);
				i+=dA[j];
				if (i > endAngleNo) break;
            } else{
                this_thread::sleep_for(chrono::milliseconds(100));
            }

			if (i > endAngleNo) break;
        }
    }
    
    for (int j=0; j<plat_dev_list.size(); j++) {
        if (th[j].joinable()) th[j].join();
    }

	delete[] dark_img;
	delete[] I0_img_target;
	for (int j = startEnergyNo; j <= endEnergyNo; j++){
		delete[] I0_imgs[j - startEnergyNo];
	}
    
    return 0;
}