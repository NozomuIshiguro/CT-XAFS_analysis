//
//  imaging2DXAFS_registration.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/05/28.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//


#include "imaging2DXAFS.hpp"

int mt_output_thread(int startLoopNo, int EndLoopNo, int startEnergyNo, int endEnergyNo,
                     string output_dir, vector<float*> mt_outputs, float *mt_merge_output,
                     vector<float*> p,vector<float*> p_err,
                     regMode regmode, int waittime){
    
    this_thread::sleep_for(chrono::seconds(waittime));
    
    string shift_dir=output_dir+ "/imageRegShift";
    MKDIR(shift_dir.c_str());
    
    const int p_num = regmode.get_p_num()+regmode.get_cp_num();
    
    ostringstream oss;
    for (int j=startLoopNo; j<=EndLoopNo; j++) {
        string fileName_output= shift_dir+ LnumTagString(j,"/shift", ".txt");
        ofstream ofs(fileName_output,ios::out|ios::trunc);
        ofs<<regmode.ofs_transpara();
        
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            string fileName_output= output_dir+ "/"+ LnumTagString(j,"","")+ EnumTagString(i,"/mtr", ".raw");
            oss << "output file: " << fileName_output << "\n";
            outputRawFile_stream(fileName_output,mt_outputs[j-startLoopNo]+(i-startEnergyNo)*IMAGE_SIZE_M,IMAGE_SIZE_M);
            
            for (int k=0; k<regmode.get_p_num(); k++) {
                ofs<<p[i-startEnergyNo][k+p_num*(j-startLoopNo)]<<"\t"
                <<p_err[i-startEnergyNo][k+p_num*(j-startLoopNo)]<<"\t";
            }
            ofs<<endl;
        }
        ofs.close();
    }
    if(EndLoopNo-startLoopNo>0){
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            string fileName_output = output_dir+ "/merge"+ EnumTagString(i,"/mtr", ".raw");
            oss << "output file: " << fileName_output << "\n";
            outputRawFile_stream(fileName_output,mt_merge_output+(i-startEnergyNo)*IMAGE_SIZE_M,IMAGE_SIZE_M);
        }
    }
    oss <<endl;
    cout << oss.str();
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
        delete [] p[i-startEnergyNo];
        delete [] p_err[i-startEnergyNo];
    }
    for(int j=startLoopNo; j<=EndLoopNo; j++){
        delete [] mt_outputs[j-startLoopNo];
    }
     delete [] mt_merge_output;
    return 0;
}


int imageReg_2D_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                       cl::Buffer dark_buffer,
                       cl::Buffer I0_target_buffer,
                       float* It_img_target,
                       int startLoopNo,int EndLoopNo, int targetLoopNo,
                       int startEnergyNo, int endEnergyNo, int targetEnergyNo,
                       string fileName_base, string output_dir, regMode regmode,
                       mask msk,int Num_trial, float lambda, bool last){
    try {
        
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
        string devicename = device.getInfo<CL_DEVICE_NAME>();
        size_t WorkGroupSize = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), IMAGE_SIZE_X);
        const int dLp=EndLoopNo-startLoopNo+1;
        const int dE =endEnergyNo-startEnergyNo+1;
        const int p_num = regmode.get_p_num()+regmode.get_cp_num();
        
        cl::ImageFormat format(CL_RG,CL_FLOAT);
        
        /* Sample I0, It data input */
        vector<float*>I0_sample_img;
        vector<float*>It2mt_sample_img;
        float *mt_merge_img;
        time_t start_t,readfinish_t;
        time(&start_t);
        mt_merge_img=new float[IMAGE_SIZE_M*dE];
        for (int i=startLoopNo; i<=EndLoopNo; i++) {
            I0_sample_img.push_back(new float[IMAGE_SIZE_M*dE]);
            It2mt_sample_img.push_back(new float[IMAGE_SIZE_M*dE]);
            string fileName_I0 = LnumTagString(i,fileName_base,"_I0.his");
            string fileName_It = LnumTagString(i,fileName_base,".his");
            readHisFile_stream(fileName_I0,startEnergyNo,endEnergyNo,I0_sample_img[i-startLoopNo],IMAGE_SIZE_M);
            readHisFile_stream(fileName_It,startEnergyNo,endEnergyNo,It2mt_sample_img[i-startLoopNo],IMAGE_SIZE_M);
        }
        vector<float*>p_vec;
        vector<float*>p_err_vec;
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            p_vec.push_back(new float[p_num*dLp]);
            p_err_vec.push_back(new float[p_num*dLp]);
        }
        time(&readfinish_t);
        
        //Buffer declaration
        cl::Buffer It2mt_target_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*dLp, 0, NULL);
        cl::Buffer I0_sample_buffer(context, CL_MEM_READ_ONLY, sizeof(cl_float)*IMAGE_SIZE_M*dLp, 0, NULL);
        cl::Buffer It2mt_sample_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*dLp, 0, NULL);
        vector<cl::Image2DArray> mt_target_image;
        vector<cl::Image2DArray> mt_sample_image;
        for (int i=0; i<4; i++) {
            int mergeN = 1<<i;
            mt_target_image.push_back(cl::Image2DArray(context, CL_MEM_READ_WRITE,format,dLp,
                                                       IMAGE_SIZE_X/mergeN,IMAGE_SIZE_Y/mergeN,
                                                       0,0,NULL,NULL));
            mt_sample_image.push_back(cl::Image2DArray(context, CL_MEM_READ_WRITE,format,dLp,
                                                       IMAGE_SIZE_X/mergeN,IMAGE_SIZE_Y/mergeN,
                                                       0,0,NULL,NULL));
        }
        cl::Buffer p_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*p_num*dLp, 0, NULL);
        cl::Buffer p_err_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*p_num*dLp, 0, NULL);
        cl::Buffer mt_merge_output_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        cl::Buffer lambda_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*dLp, 0, NULL);

        
        
        //kernel dimension declaration
        const cl::NDRange global_item_size(WorkGroupSize*dLp,IMAGE_SIZE_Y,1);
        const cl::NDRange global_item_size2(IMAGE_SIZE_X,IMAGE_SIZE_Y,1);
        const cl::NDRange local_item_size(WorkGroupSize,1,1);
        
        
        //Energy loop setting
        vector<int> LoopEndEnergyNo={startEnergyNo,endEnergyNo};
        vector<int> LoopStartenergyNo;
        LoopStartenergyNo.push_back(min(targetEnergyNo, endEnergyNo));
        LoopStartenergyNo.push_back(max(targetEnergyNo, startEnergyNo));
        
        
        if (startLoopNo==EndLoopNo) {
            cout << "Device: "<< devicename << ", loop: "<<startLoopNo<< ", processing I0 and It..."<<endl;
        } else {
            cout << "Device: "<< devicename << ", Loop: "<<startLoopNo<<"-"<<EndLoopNo<< "    processing I0 and It..."<<endl;
        }
        
        
        //write dark buffer
        
        //target mt conversion
        command_queue.enqueueWriteBuffer(It2mt_target_buffer, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M*dLp, It_img_target, NULL, NULL);
        kernel[0].setArg(0, dark_buffer);
        kernel[0].setArg(1, I0_target_buffer);
        kernel[0].setArg(2, It2mt_target_buffer);
        kernel[0].setArg(3, mt_target_image[0]);
        kernel[0].setArg(4, msk.refMask_shape);
        kernel[0].setArg(5, msk.refMask_x);
        kernel[0].setArg(6, msk.refMask_y);
        kernel[0].setArg(7, msk.refMask_width);
        kernel[0].setArg(8, msk.refMask_height);
        kernel[0].setArg(9, msk.refMask_angle);
        command_queue.enqueueNDRangeKernel(kernel[0], cl::NullRange, global_item_size, local_item_size, NULL, NULL);
        command_queue.finish();
        delete [] It_img_target;
        
        
        //It_target merged image create
        if (regmode.get_regModeNo()>=0) {
            for (int i=3; i>0; i--) {
                int mergeN = 1<<i;
                int localsize = min((int)WorkGroupSize,IMAGE_SIZE_X/mergeN);
                const cl::NDRange global_item_size_merge(localsize*dLp,1,1);
                const cl::NDRange local_item_size_merge(localsize,1,1);
                
                kernel[1].setArg(0, mt_target_image[0]);
                kernel[1].setArg(1, mt_target_image[i]);
                kernel[1].setArg(2, mergeN);
                command_queue.enqueueNDRangeKernel(kernel[1], cl::NullRange, global_item_size_merge, local_item_size_merge, NULL, NULL);
            }
            command_queue.finish();
        }
        
        
        for (int s=0; s<2; s++) {//1st cycle:i<targetEnergyNo, 2nd cycle:i>targetEnergyNo
            int di=(-1+2*s);
            if (startEnergyNo==endEnergyNo){
                if(startEnergyNo==targetEnergyNo) break;
            }else if ((LoopStartenergyNo[s]+di)*di>LoopEndEnergyNo[s]*di) {
                continue;
            }
            
            //transpara reset
            command_queue.enqueueFillBuffer(p_buffer, (cl_float)0.0, 0, sizeof(cl_float)*p_num*dLp,NULL,NULL);
            command_queue.finish();
            
            int ds = (LoopStartenergyNo[s]==targetEnergyNo) ? di:0;
            for (int i=LoopStartenergyNo[s]+ds; i*di<=LoopEndEnergyNo[s]*di; i+=di) {
                
                //sample mt conversion
                for (int k = 0; k<dLp; k++) {
                    command_queue.enqueueWriteBuffer(I0_sample_buffer, CL_TRUE, sizeof(cl_float)*IMAGE_SIZE_M*k, sizeof(cl_float)*IMAGE_SIZE_M, I0_sample_img[i-startEnergyNo], NULL, NULL);
                }
                command_queue.enqueueWriteBuffer(It2mt_sample_buffer, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M*dLp,It2mt_sample_img[i-startEnergyNo], NULL, NULL);
                kernel[0].setArg(0, dark_buffer);
                kernel[0].setArg(1, I0_sample_buffer);
                kernel[0].setArg(2, It2mt_sample_buffer);
                kernel[0].setArg(3, mt_sample_image[0]);
                kernel[0].setArg(4, msk.sampleMask_shape);
                kernel[0].setArg(5, msk.sampleMask_x);
                kernel[0].setArg(6, msk.sampleMask_y);
                kernel[0].setArg(7, msk.sampleMask_width);
                kernel[0].setArg(8, msk.sampleMask_height);
                kernel[0].setArg(9, msk.sampleMask_angle);
                command_queue.enqueueNDRangeKernel(kernel[0], cl::NullRange, global_item_size, local_item_size, NULL, NULL);
                command_queue.finish();
                
                
                if(regmode.get_regModeNo()>=0){
                    //It_sample merged image create
                    for (int i=3; i>0; i--) {
                        int mergeN = 1<<i;
                        int localsize = min((int)WorkGroupSize,IMAGE_SIZE_X/mergeN);
                        const cl::NDRange global_item_size_merge(localsize*dLp,1,1);
                        const cl::NDRange local_item_size_merge(localsize,1,1);
                        
                        kernel[1].setArg(0, mt_sample_image[0]);
                        kernel[1].setArg(1, mt_sample_image[i]);
                        kernel[1].setArg(2, mergeN);
                        command_queue.enqueueNDRangeKernel(kernel[1], cl::NullRange, global_item_size_merge, local_item_size_merge, NULL, NULL);
                        command_queue.finish();
                    }
                    
                    
                    //lambda_buffer reset
                    command_queue.enqueueFillBuffer(lambda_buffer, (cl_float)lambda, 0, sizeof(cl_float)*dLp,NULL,NULL);
                    command_queue.finish();
                    
                    
                    //Image registration
                    for (int i=0; i>=0; i--) {
                        int mergeN = 1<<i;
                        //cout<<mergeN<<endl;
                        int localsize = min((int)WorkGroupSize,IMAGE_SIZE_X/mergeN);
                        const cl::NDRange global_item_size_merge(localsize*dLp,1,1);
                        const cl::NDRange local_item_size_merge(localsize,1,1);
                        
                        kernel[2].setArg(0, mt_target_image[i]);
                        kernel[2].setArg(1, mt_sample_image[i]);
                        kernel[2].setArg(2, lambda_buffer);
                        kernel[2].setArg(3, p_buffer);
                        kernel[2].setArg(4, p_err_buffer);
                        kernel[2].setArg(5, cl::Local(sizeof(cl_float)*localsize));//locmem
                        kernel[2].setArg(6, mergeN);
                        kernel[2].setArg(7, 1.0f);
                        for (int trial=0; trial < Num_trial; trial++) {
                            command_queue.enqueueNDRangeKernel(kernel[2], NULL, global_item_size_merge, local_item_size_merge, NULL, NULL);
                            command_queue.finish();
                        }
                    }
                    
                    
                    //output image reg results to buffer
                    kernel[3].setArg(0, mt_sample_image[0]);
                    kernel[3].setArg(1, It2mt_sample_buffer);
                    kernel[3].setArg(2, p_buffer);
                    //kernel[3].setArg(3, cl::Local(sizeof(cl_float)*p_num));//p_loc
                    command_queue.enqueueNDRangeKernel(kernel[3], NULL, global_item_size, local_item_size, NULL, NULL);
                    command_queue.finish();
                    
                    
                    //transpara read buffer
                    command_queue.enqueueReadBuffer(p_buffer,CL_TRUE,0,sizeof(cl_float)*p_num*dLp,p_vec[i-startEnergyNo],NULL,NULL);
                    command_queue.enqueueReadBuffer(p_err_buffer,CL_TRUE,0,sizeof(cl_float)*p_num*dLp,p_err_vec[i-startEnergyNo],NULL,NULL);
                    command_queue.finish();
                }
                
                
                ostringstream oss;
                for (int k=0; k<dLp; k++) {
                    if (startLoopNo+k>EndLoopNo) {
                        break;
                    }
                    int *p_precision, *p_err_precision;
                    p_precision=new int[p_num];
                    p_err_precision=new int[p_num];
                    for (int n=0; n<p_num; n++) {
                        int a = floor(log10(abs(p_vec[i-startEnergyNo][n+k*p_num])));
                        int b = floor(log10(abs(p_err_vec[i-startEnergyNo][n+k*p_num])));
                        p_err_precision[n] = max(0,b)+1;
                        
                        
                        if (a>0) {
                            int c = floor(log10(pow(10,a+1)-0.5));
                            if(a>c) a++;
                            
                            p_precision[n] = a+1 - min(0,b);
                        }else if(a<b){
                            p_precision[n] = 1;
                        }else{
                            p_precision[n]= a - b + 1;
                        }
                    }
                    oss << "Device: "<< devicename << ", loop: "<<startLoopNo+k<< ", energy: "<<i<<endl;
                    oss << regmode.oss_sample(p_vec[i-startEnergyNo]+p_num*k,p_err_vec[i-startEnergyNo]+p_num*k,p_precision,p_err_precision);

                }
                cout << oss.str();
                
                //merge mt
                kernel[4].setArg(0, mt_target_image);
                kernel[4].setArg(1, mt_merge_output_buffer);
                command_queue.enqueueNDRangeKernel(kernel[7], NULL, global_item_size2, local_item_size, NULL, NULL);
                command_queue.finish();
                
                //read output buffer
                for (int k=0; k<dLp; k++) {
                    command_queue.enqueueReadBuffer(It2mt_sample_buffer, CL_TRUE, sizeof(cl_float)*IMAGE_SIZE_M*k, sizeof(cl_float)*IMAGE_SIZE_M, It2mt_sample_img[k]+IMAGE_SIZE_M*(i-startEnergyNo), NULL, NULL);
                    command_queue.finish();
                }
                command_queue.enqueueReadBuffer(mt_merge_output_buffer, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M, mt_merge_img+IMAGE_SIZE_M*(i-startEnergyNo), NULL, NULL);
                command_queue.finish();
            }
        }
        
        for (int i=startLoopNo; i<=EndLoopNo; i++) {
            delete [] I0_sample_img[i-startLoopNo];
        }
        
        int delta_t;
        if (last) {
            delta_t=0;
        }else{
            delta_t= 0;//min(60.0,difftime(readfinish_t,start_t)*0.4);
            //considering reading time of next cycle (max 60s)
        }
        
        thread th_output(mt_output_thread,
                         startLoopNo,EndLoopNo,startEnergyNo,endEnergyNo,
                         output_dir,move(It2mt_sample_img),move(mt_merge_img),
                         move(p_vec),move(p_err_vec),regmode,delta_t);
        if(last) th_output.join();
        else th_output.detach();
        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
    
    
    return 0;
}




int imageRegistlation_2D_ocl(string fileName_base, input_parameter inp,
                          OCL_platform_device plat_dev_list, regMode regmode)
{
    cl_int ret;
    
    const int startEnergyNo=inp.getStartEnergyNo();
    const int endEnergyNo=inp.getEndEnergyNo();
    const int targetEnergyNo=inp.getTargetEnergyNo();
    const int startLoopNo=inp.getStartAngleNo();
    const int endLoopNo=inp.getEndAngleNo();
    const int targetLoopNo=inp.getTargetAngleNo();
    
    
    for (int i=startLoopNo; i<=endLoopNo; i++) {
        string fileName_output = inp.getOutputDir() + "/" + LnumTagString(i,"","");
        MKDIR(fileName_output.c_str());
    }
    if(endLoopNo-startLoopNo>0){
        string fileName_output = inp.getOutputDir() + "/merged";
        MKDIR(fileName_output.c_str());
    }
    
    
    
    //OpenCL Program
    vector<vector<cl::Kernel>> kernels;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        cl::Program program=regmode.buildImageRegProgram(plat_dev_list.context(i));
        vector<cl::Kernel> kernels_plat;
        kernels_plat.push_back(cl::Kernel::Kernel(program,"mt_conversion", &ret));//0
        kernels_plat.push_back(cl::Kernel::Kernel(program,"merge", &ret));//1
        kernels_plat.push_back(cl::Kernel::Kernel(program,"imageRegistration", &ret));//2
        kernels_plat.push_back(cl::Kernel::Kernel(program,"output_imgReg_result", &ret));//3
        kernels_plat.push_back(cl::Kernel::Kernel(program,"merge_mt", &ret));//4
        kernels.push_back(kernels_plat);
    }
    
    
    //display OCL device
    int t=0;
    vector<int> dLp;
    for (int i=0; i<plat_dev_list.platsize(); i++) {
        for (int j=0; j<plat_dev_list.devsize(i); j++) {
            cout<<"Device No. "<<t+1<<endl;
            string platform_param;
            plat_dev_list.plat(i).getInfo(CL_PLATFORM_NAME, &platform_param);
            cout << "CL PLATFORM NAME: "<< platform_param<<endl;
            plat_dev_list.plat(i).getInfo(CL_PLATFORM_VERSION, &platform_param);
            cout << "   "<<platform_param<<endl;
            string device_pram;
            plat_dev_list.dev(i,j).getInfo(CL_DEVICE_NAME, &device_pram);
            cout << "CL DEVICE NAME: "<< device_pram<<endl;
            
            /*Open CL command que create*/
            dLp.push_back(endLoopNo-startLoopNo+1);
            cout<<"Number of working compute unit: "<<dLp[i]<<endl<<endl;
            t++;
        }
    }
    
    
    /* dark data input / OCL transfer*/
    cout << "Processing dark..."<<endl;
    float *dark_img;
    dark_img = new float[IMAGE_SIZE_M];
    string fileName_dark = fileName_base+ "dark.his";
    //cout<<fileName_dark<<endl;
    readHisFile_stream(fileName_dark,1,30,dark_img,0);
    
    
    // I0 target data input
    float *I0_img_target;
    I0_img_target = new float[IMAGE_SIZE_M];
    string fileName_I0_target;
    fileName_I0_target = LnumTagString(targetLoopNo,fileName_base,"_I0.his");
    //cout<<fileName_I0_target<<endl;
    readHisFile_stream(fileName_I0_target,targetEnergyNo,targetEnergyNo,I0_img_target,0);
    
    
    // It target data input
    float *It_img_target;
    It_img_target = new float[IMAGE_SIZE_M];
    string fileName_It_target;
    fileName_It_target = LnumTagString(targetLoopNo,fileName_base,".his");
    readHisFile_stream(fileName_It_target,targetEnergyNo,targetEnergyNo,It_img_target,0);
    
    
    //create dark, I0_target, I0_sample buffers
    vector<cl::Buffer> dark_buffers;
    vector<cl::Buffer> I0_target_buffers;
    for (int i=0; i<plat_dev_list.platsize(); i++) {
        cl::Buffer dark_buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        cl::Buffer I0_target_buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        cl::Buffer It_target_buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        
        //write buffers
        plat_dev_list.queue(i, 0).enqueueWriteBuffer(dark_buffer, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M, dark_img, NULL, NULL);
        plat_dev_list.queue(i, 0).enqueueWriteBuffer(I0_target_buffer, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M, I0_img_target, NULL, NULL);
        
        dark_buffers.push_back(dark_buffer);
        I0_target_buffers.push_back(I0_target_buffer);
    }
    delete[] dark_img;
    delete[] I0_img_target;
    

    
    
    //mask settings
    mask msk(inp);
    
    
    //queueを通し番号に変換
    vector<cl::CommandQueue> queues;
    vector<int> cotextID_OfQueue;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        for (int j=0; j<plat_dev_list.queuesize(i); j++) {
            queues.push_back(plat_dev_list.queue(i,j));
            cotextID_OfQueue.push_back(i);
        }
    }
    
    
    //start threads
    vector<thread> th;
    for (int i=startLoopNo; i<=endLoopNo;) {
        for (int j = 0; j<queues.size(); j++) {
            
            
            if (th.size()<queues.size()) {
                bool last = (i + dLp[j] - 1>endLoopNo - queues.size()*dLp[j]);
                th.push_back(thread(imageReg_2D_thread, queues[j], kernels[cotextID_OfQueue[j]],
                                    dark_buffers[cotextID_OfQueue[j]],
                                    I0_target_buffers[cotextID_OfQueue[j]],
                                    It_img_target,
                                    i, min(i + dLp[j] - 1, endLoopNo),targetLoopNo,
                                    startEnergyNo,endEnergyNo,targetEnergyNo,
                                    fileName_base, inp.getOutputDir(), regmode,
                                    msk,inp.getNumTrial(),inp.getLambda_t(),last));
                this_thread::sleep_for(chrono::seconds((int)((endLoopNo-startEnergyNo+1)*dLp[j]*0.02)));
                //this_thread::sleep_for(chrono::seconds(40));
                i+=dLp[j];
                if (i > endLoopNo) break;
                else continue;
            }else if (th[j].joinable()) {
                bool last = (i + dLp[j] - 1>endLoopNo - queues.size()*dLp[j]);
                th[j].join();
                th[j] = thread(imageReg_2D_thread, queues[j], kernels[cotextID_OfQueue[j]],
                               dark_buffers[cotextID_OfQueue[j]],
                               I0_target_buffers[cotextID_OfQueue[j]],
                               It_img_target,
                               i, min(i + dLp[j] - 1, endLoopNo),targetLoopNo,
                               startEnergyNo,endEnergyNo,targetEnergyNo,
                               fileName_base, inp.getOutputDir(), regmode,
                               msk, inp.getNumTrial(),inp.getLambda_t(),last);
                i+=dLp[j];
                if (i > endLoopNo) break;
            } else{
                this_thread::sleep_for(chrono::milliseconds(100));
            }
            
            if (i > endLoopNo) break;
        }
    }
    
    for (int j=0; j<queues.size(); j++) {
        if (th[j].joinable()) th[j].join();
    }
    delete[] It_img_target;
    
    
    return 0;
}