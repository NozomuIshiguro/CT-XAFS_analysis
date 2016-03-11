//
//  CT_ImageRegistration_C++.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/02/06.
//  Copyright (c) 2015 Nozomu Ishiguro. All rights reserved.
//

#include "CTXAFS.hpp"



int mt_output_thread(int startAngleNo, int EndAngleNo, int startEnergyNo, int endEnergyNo,
                     string output_dir, string output_base,
                     vector<float*> mt_outputs,
                     vector<float*> p,vector<float*> p_err,
                     regMode regmode, int waittime){

    this_thread::sleep_for(chrono::seconds(waittime));
    
    string shift_dir=output_dir+ "/imageRegShift";
    MKDIR(shift_dir.c_str());
    
    const int p_num = regmode.get_p_num()+regmode.get_cp_num();
    
    ostringstream oss;
    for (int j=startAngleNo; j<=EndAngleNo; j++) {
        string fileName_output= shift_dir+ AnumTagString(j,"/shift", ".txt");
        ofstream ofs(fileName_output,ios::out|ios::trunc);
        ofs<<regmode.ofs_transpara();
        
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            string fileName_output= output_dir+ "/" + EnumTagString(i,"","/") + AnumTagString(j,output_base, ".raw");
            oss << "output file: " << fileName_output <<endl;
            outputRawFile_stream(fileName_output,mt_outputs[i-startEnergyNo]+(j-startAngleNo)*IMAGE_SIZE_M,IMAGE_SIZE_M);
            
            for (int k=0; k<p_num; k++) {
                ofs.precision(7);
                ofs<<p[i-startEnergyNo][k+p_num*(j-startAngleNo)]<<"\t"
                <<p_err[i-startEnergyNo][k+p_num*(j-startAngleNo)]<<"\t";
            }
            ofs<<endl;
        }
        ofs.close();
    }
    oss << endl;
    cout << oss.str();
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
        delete [] mt_outputs[i-startEnergyNo];
        delete [] p[i-startEnergyNo];
        delete [] p_err[i-startEnergyNo];
    }
    return 0;
}


int imageReg_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                    cl::Buffer dark_buffer,
                    cl::Buffer I0_target_buffer,
                    vector<cl::Buffer> I0_sample_buffers,
                    int startAngleNo,int EndAngleNo,
                    int startEnergyNo, int endEnergyNo, int targetEnergyNo,
					string fileName_base, string output_dir, string output_base, regMode regmode,
                    mask msk,int Num_trial, float lambda, bool last){
    try {
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
        string devicename = device.getInfo<CL_DEVICE_NAME>();
		size_t WorkGroupSize = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), IMAGE_SIZE_X);
        const int dA=EndAngleNo-startAngleNo+1;
        const int p_num = regmode.get_p_num()+regmode.get_cp_num();
        
        //Loop process for data input
        time_t start_t,readfinish_t;
        time(&start_t);
        /*target It data input*/
        float *It_img_target;
        It_img_target = new float[IMAGE_SIZE_M*dA];
        string fileName_It_target = EnumTagString(targetEnergyNo,fileName_base,".his");
        readHisFile_stream(fileName_It_target,startAngleNo,EndAngleNo,It_img_target,IMAGE_SIZE_M);
        
        cl::ImageFormat format(CL_RG,CL_FLOAT);
        
        
        /* Sample It data input */
        vector<float*>sample_img;
        vector<float*>p_vec;
        vector<float*>p_err_vec;
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            sample_img.push_back(new float[IMAGE_SIZE_M*dA]);
            p_vec.push_back(new float[p_num*dA]);
            p_err_vec.push_back(new float[p_num*dA]);
            string fileName_It = EnumTagString(i,fileName_base,".his");
            readHisFile_stream(fileName_It,startAngleNo,EndAngleNo,sample_img[i-startEnergyNo],IMAGE_SIZE_M);
        }
        time(&readfinish_t);
        
        
        //Buffer declaration
		cl::Buffer It2mt_target_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*dA, 0, NULL);
		cl::Buffer It2mt_sample_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*dA, 0, NULL);
        vector<cl::Image2DArray> mt_target_image;
        vector<cl::Image2DArray> mt_sample_image;
        for (int i=0; i<4; i++) {
            int mergeN = 1<<i;
            mt_target_image.push_back(cl::Image2DArray(context, CL_MEM_READ_WRITE,format,dA,
                                                       IMAGE_SIZE_X/mergeN,IMAGE_SIZE_Y/mergeN,
                                                       0,0,NULL,NULL));
            mt_sample_image.push_back(cl::Image2DArray(context, CL_MEM_READ_WRITE,format,dA,
                                                       IMAGE_SIZE_X/mergeN,IMAGE_SIZE_Y/mergeN,
                                                       0,0,NULL,NULL));
        }
        cl::Buffer p_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*p_num*dA, 0, NULL);
        cl::Buffer p_err_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*p_num*dA, 0, NULL);
        cl::Buffer lambda_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*dA, 0, NULL);
        cl::Image2DArray mt_target_outputImg(context, CL_MEM_READ_WRITE,format,dA,IMAGE_SIZE_X,IMAGE_SIZE_Y,0,0,NULL,NULL);
        cl::Image2DArray mt_sample_outputImg(context, CL_MEM_READ_WRITE,format,dA,IMAGE_SIZE_X,IMAGE_SIZE_Y,0,0,NULL,NULL);
        
        
        //kernel dimension declaration
        const cl::NDRange global_item_size(WorkGroupSize*dA,IMAGE_SIZE_Y,1);
        const cl::NDRange local_item_size(WorkGroupSize,1,1);
        
        
        //Energy loop setting
        vector<int> LoopEndenergyNo={startEnergyNo,endEnergyNo};
        vector<int> LoopStartenergyNo;
        LoopStartenergyNo.push_back(min(targetEnergyNo, endEnergyNo));
        LoopStartenergyNo.push_back(max(targetEnergyNo, startEnergyNo));
        
        
        if (startAngleNo==EndAngleNo) {
            cout << "Device: "<< devicename << ", angle: "<<startAngleNo<< ", processing It..."<<endl<<endl;
        } else {
            cout << "Device: "<< devicename << ", angle: "<<startAngleNo<<"-"<<EndAngleNo<< "    processing It..."<<endl<<endl;
        }
        
        
        

		//target mt conversion
        command_queue.enqueueWriteBuffer(It2mt_target_buffer, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M*dA, It_img_target, NULL, NULL);
        kernel[0].setArg(0, dark_buffer);
        kernel[0].setArg(1, I0_target_buffer);
        kernel[0].setArg(2, It2mt_target_buffer);
        kernel[0].setArg(3, mt_target_image[0]);
        kernel[0].setArg(4, mt_target_outputImg);
        kernel[0].setArg(5, msk.refMask_shape);
        kernel[0].setArg(6, msk.refMask_x);
        kernel[0].setArg(7, msk.refMask_y);
        kernel[0].setArg(8, msk.refMask_width);
        kernel[0].setArg(9, msk.refMask_height);
        kernel[0].setArg(10, msk.refMask_angle);
        kernel[0].setArg(11, 0);
        command_queue.enqueueNDRangeKernel(kernel[0], cl::NullRange, global_item_size, local_item_size, NULL, NULL);
		command_queue.finish();
        delete [] It_img_target;
        
        
        //It_target merged image create
        if (regmode.get_regModeNo()>=0) {
            for (int i=3; i>0; i--) {
                int mergeN = 1<<i;
                int localsize = min((int)WorkGroupSize,IMAGE_SIZE_X/mergeN);
                const cl::NDRange global_item_size_merge(localsize*dA,1,1);
                const cl::NDRange local_item_size_merge(localsize,1,1);
                
                kernel[1].setArg(0, mt_target_image[0]);
                kernel[1].setArg(1, mt_target_image[i]);
                kernel[1].setArg(2, mergeN);
                command_queue.enqueueNDRangeKernel(kernel[1], cl::NullRange, global_item_size_merge, local_item_size_merge, NULL, NULL);
            }
            command_queue.finish();
        }
        
        
        //process when (sample Enegry No. == target Energy No.)
        if ((targetEnergyNo>=startEnergyNo)&(targetEnergyNo<=endEnergyNo)) {
            command_queue.enqueueReadBuffer(It2mt_target_buffer, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M*dA, sample_img[targetEnergyNo - startEnergyNo], NULL, NULL);
            command_queue.finish();
            
            ostringstream oss;
            for (int j=startAngleNo; j<=EndAngleNo; j++) {
                oss << "Device: "<< devicename << ", angle: "<<j<< ", energy: "<<targetEnergyNo<<endl;
                oss <<regmode.get_oss_target();
                for (int t=0; t<p_num; t++) {
                    p_vec[targetEnergyNo - startEnergyNo][t+p_num*(j-startAngleNo)]=0;
                    p_err_vec[targetEnergyNo - startEnergyNo][t+p_num*(j-startAngleNo)]=0;
                }
            }
            cout << oss.str();
        }
        
        
        for (int s=0; s<2; s++) {//1st cycle:i<targetEnergyNo, 2nd cycle:i>targetEnergyNo
            int di=(-1+2*s); //1st -1, 2nd +1
            if (startEnergyNo==endEnergyNo){
                if(startEnergyNo==targetEnergyNo) break;
            }else if ((LoopStartenergyNo[s]+di)*di>LoopEndenergyNo[s]*di) {
                continue;
            }
            
            
            //image reg parameter (p_buffer) reset
            command_queue.enqueueFillBuffer(p_buffer, (cl_float)0.0, 0, sizeof(cl_float)*p_num*dA,NULL,NULL);
            command_queue.finish();
            
            int ds = (LoopStartenergyNo[s]==targetEnergyNo) ? di:0;
            for (int i=LoopStartenergyNo[s]+ds; i*di<=LoopEndenergyNo[s]*di; i+=di) {

                //sample mt conversion
                command_queue.enqueueWriteBuffer(It2mt_sample_buffer, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M*dA, sample_img[i-startEnergyNo], NULL, NULL);
                kernel[0].setArg(0, dark_buffer);
                kernel[0].setArg(1, I0_sample_buffers[i-startEnergyNo]);
                kernel[0].setArg(2, It2mt_sample_buffer);
                kernel[0].setArg(3, mt_sample_image[0]);
                kernel[0].setArg(4, mt_sample_outputImg);
                kernel[0].setArg(5, msk.sampleMask_shape);
                kernel[0].setArg(6, msk.sampleMask_x);
                kernel[0].setArg(7, msk.sampleMask_y);
                kernel[0].setArg(8, msk.sampleMask_width);
                kernel[0].setArg(9, msk.sampleMask_height);
                kernel[0].setArg(10, msk.sampleMask_angle);
                kernel[0].setArg(11, 0);
                command_queue.enqueueNDRangeKernel(kernel[0], cl::NullRange, global_item_size, local_item_size, NULL, NULL);
                command_queue.finish();
                
        
                if (regmode.get_regModeNo()>=0) {
                    
                    //It_sample merged image create
                    for (int j=3; j>0; j--) {
                        int mergeN = 1<<j;
                        int localsize = min((int)WorkGroupSize,IMAGE_SIZE_X/mergeN);
                        const cl::NDRange global_item_size_merge(localsize*dA,1,1);
                        const cl::NDRange local_item_size_merge(localsize,1,1);
                        
                        kernel[1].setArg(0, mt_sample_image[0]);
                        kernel[1].setArg(1, mt_sample_image[j]);
                        kernel[1].setArg(2, mergeN);
                        command_queue.enqueueNDRangeKernel(kernel[1], cl::NullRange, global_item_size_merge, local_item_size_merge, NULL, NULL);
                        command_queue.finish();
                    }
                    
                    //lambda_buffer reset
                    command_queue.enqueueFillBuffer(lambda_buffer, (cl_float)lambda, 0, sizeof(cl_float)*dA,NULL,NULL);
                    command_queue.finish();
                    
                    //Image registration
                    for (int j=3; j>=0; j--) {
                        
                        int mergeN = 1<<j;
                        //cout<<mergeN<<endl;
                        int localsize = min((int)WorkGroupSize,IMAGE_SIZE_X/mergeN);
                        const cl::NDRange global_item_size_reg(localsize*dA,1,1);
                        const cl::NDRange local_item_size_reg(localsize,1,1);
                        
                        kernel[2].setArg(0, mt_target_image[j]);
                        kernel[2].setArg(1, mt_sample_image[j]);
                        kernel[2].setArg(2, lambda_buffer);
                        kernel[2].setArg(3, p_buffer);
                        kernel[2].setArg(4, p_err_buffer);
                        kernel[2].setArg(5, cl::Local(sizeof(cl_float)*localsize));//locmem
                        kernel[2].setArg(6, mergeN);
                        kernel[2].setArg(7, 1.0f);
                        
                        for (int trial=0; trial < Num_trial; trial++) {
                            command_queue.enqueueNDRangeKernel(kernel[2], NULL, global_item_size_reg, local_item_size_reg, NULL, NULL);
                            command_queue.finish();
                        }
                    }
                    
                    
                    //output image reg results to buffer
                    kernel[3].setArg(0, mt_sample_outputImg);
                    kernel[3].setArg(1, It2mt_sample_buffer);
                    kernel[3].setArg(2, p_buffer);
                    command_queue.enqueueNDRangeKernel(kernel[3], NULL, global_item_size, local_item_size, NULL, NULL);
                    command_queue.finish();
                    
                    
                    
                    //read p_buffer & p_err_buffer
                    command_queue.enqueueReadBuffer(p_buffer,CL_TRUE,0,sizeof(cl_float)*p_num*dA,p_vec[i-startEnergyNo],NULL,NULL);
                    command_queue.enqueueReadBuffer(p_err_buffer,CL_TRUE,0,sizeof(cl_float)*p_num*dA,p_err_vec[i-startEnergyNo],NULL,NULL);
                    command_queue.finish();
                }
                
                
                ostringstream oss;
                for (int k=0; k<dA; k++) {
                    if (startAngleNo+k>EndAngleNo) {
                        break;
                    }
                    int *p_precision, *p_err_precision;
                    p_precision=new int[p_num];
                    p_err_precision=new int[p_num];
                    for (int n=0; n<p_num; n++) {
                        int a = (int)floor(log10(abs(p_vec[i-startEnergyNo][n+k*p_num])));
                        int b = (int)floor(log10(abs(p_err_vec[i-startEnergyNo][n+k*p_num])));
                        p_err_precision[n] = max(0,b)+1;
                        
                        
                        if (a>0) {
                            int c = (int)floor(log10(pow(10,a+1)-0.5));
                            if(a>c) a++;
                            
                            p_precision[n] = a+1 - min(0,b);
                        }else if(a<b){
                            p_precision[n] = 1;
                        }else{
                            p_precision[n]= a - max(-3,b) + 1;
                        }
                    }
                    oss<<"Device: "<<devicename<<", angle: "<<startAngleNo+k<<", energy: "<<i<<endl;
                    oss << regmode.oss_sample(p_vec[i-startEnergyNo]+p_num*k, p_err_vec[i-startEnergyNo]+p_num*k,p_precision,p_err_precision);
                    
                }
                cout << oss.str();

				//read output buffer
                command_queue.enqueueReadBuffer(It2mt_sample_buffer, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M*dA, sample_img[i - startEnergyNo], NULL, NULL);
                command_queue.finish();
            }
        }
        
        int delta_t;
        if (last) {
            delta_t=0;
        }else{
            delta_t= 0;//max(60.0,difftime(readfinish_t,start_t)*0.4);
            //considering reading time of next cycle (max 60s)
        }
        
        thread th_output(mt_output_thread,
                         startAngleNo,EndAngleNo,startEnergyNo,endEnergyNo,
                         output_dir,output_base,
                         move(sample_img),move(p_vec),move(p_err_vec),regmode,delta_t);
        if(last) th_output.join();
        else th_output.detach();

    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
    
    
    return 0;
}




int imageRegistlation_ocl(string fileName_base, input_parameter inp,
                          OCL_platform_device plat_dev_list, regMode regmode)
{
    cl_int ret;
    
    int startEnergyNo=inp.getStartEnergyNo();
    int endEnergyNo=inp.getEndEnergyNo();
    int targetEnergyNo=inp.getTargetEnergyNo();
    int startAngleNo=inp.getStartAngleNo();
    int endAngleNo=inp.getEndAngleNo();
    
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
        string fileName_output = inp.getOutputDir() + "/" + EnumTagString(i,"","");
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
        kernels.push_back(kernels_plat);
    }
    
    
    //display OCL device
    int t=0;
    vector<int> dA;
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
            
            //working compute unit
            dA.push_back(4);
            cout<<"Number of working compute unit: "<<dA[t]<<endl<<endl;
            t++;
        }
    }
    
    
    // dark data input
    cout << "Reading dark file..."<<endl;
    float *dark_img;
    dark_img = new float[IMAGE_SIZE_M];
    string fileName_dark = fileName_base+ "dark.his";
    readHisFile_stream(fileName_dark,1,30,dark_img,0);
    
    
    // I0 sample data input
    cout << "Reading I0 files..."<<endl<<endl;
    vector<float*> I0_imgs;
    float *I0_img_target;
    I0_img_target = new float[IMAGE_SIZE_M];
    for (int i = startEnergyNo; i <= endEnergyNo; i++) {
        I0_imgs.push_back(new float[IMAGE_SIZE_M]);
        string fileName_I0 = EnumTagString(i,fileName_base,"_I0.his");
        int readHis_err=readHisFile_stream(fileName_I0,1,20,I0_imgs[i-startEnergyNo],0);
        if (readHis_err<0) {
            endEnergyNo=i-1;
            cout <<"No more I0 at Energy No. at over"<<i-1<<endl;
            break;
        }
    }
    
        
    // I0 target data input
    string fileName_I0_target;
    fileName_I0_target =  EnumTagString(targetEnergyNo,fileName_base,"_I0.his");
    readHisFile_stream(fileName_I0_target,1,20,I0_img_target,0);
    
    
    //create dark, I0_target, I0_sample buffers
    vector<cl::Buffer> dark_buffers;
    vector<cl::Buffer> I0_target_buffers;
    vector<vector<cl::Buffer>> I0_sample_buffers;
	for (int i = 0; i<plat_dev_list.contextsize(); i++) {
        cl::Buffer dark_buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        cl::Buffer I0_target_buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        vector<cl::Buffer> I0_sample_buffers_AtE;
        
        //write buffers
        plat_dev_list.queue(i, 0).enqueueWriteBuffer(dark_buffer, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M, dark_img, NULL, NULL);
        plat_dev_list.queue(i, 0).enqueueWriteBuffer(I0_target_buffer, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M, I0_img_target, NULL, NULL);
        for (int j = startEnergyNo; j <= endEnergyNo; j++) {
            cl::Buffer I0_sample_buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
            plat_dev_list.queue(i, 0).enqueueWriteBuffer(I0_sample_buffer, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M, I0_imgs[j-startEnergyNo], NULL, NULL);
            I0_sample_buffers_AtE.push_back(I0_sample_buffer);
        }
        
        
        dark_buffers.push_back(dark_buffer);
        I0_target_buffers.push_back(I0_target_buffer);
        I0_sample_buffers.push_back(I0_sample_buffers_AtE);
    }
    delete[] dark_img;
    delete[] I0_img_target;
    for (int j = startEnergyNo; j <= endEnergyNo; j++){
        delete[] I0_imgs[j - startEnergyNo];
    }
    
    
    //queueを通し番号に変換
    /*vector<cl::CommandQueue> queues;
    vector<int> cotextID_OfQueue;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        for (int j=0; j<plat_dev_list.queuesize(i); j++) {
            queues.push_back(plat_dev_list.queue(i,j));
            cotextID_OfQueue.push_back(i);
        }
    }*/
    
    
    //mask settings
    mask msk(inp);
    
    //start threads
	vector<thread> th;
    for (int i=startAngleNo; i<=endAngleNo;) {
		//for (int j = 0; j<queues.size(); j++) {
		for (int j = 0; j<plat_dev_list.contextsize(); j++) {
            
            if (th.size()<plat_dev_list.contextsize()) { //非共有context時
            bool last = (i + dA[j] - 1>endAngleNo - plat_dev_list.contextsize()*dA[j]);
			th.push_back(thread(imageReg_thread, plat_dev_list.queue(j, 0), kernels[j],
                                dark_buffers[j],
                                I0_target_buffers[j],
                                I0_sample_buffers[j],
                                i, min(i + dA[j] - 1, endAngleNo),
                                startEnergyNo,endEnergyNo,targetEnergyNo,
                                fileName_base, inp.getOutputDir(), inp.getOutputFileBase(),
                                regmode,msk,inp.getNumTrial(),inp.getLambda_t(),last));
			/*if (th.size()<queues.size()) { //共有context時
				bool last = (i + dA[j] - 1>endAngleNo - queues.size()*dA[j]);
                th.push_back(thread(imageReg_thread, queues[j], kernels[cotextID_OfQueue[j]],
                                    dark_buffers[cotextID_OfQueue[j]],
                                    I0_target_buffers[cotextID_OfQueue[j]],
                                    I0_sample_buffers[cotextID_OfQueue[j]],
									i, min(i + dA[j] - 1, endAngleNo),
                                    startEnergyNo,endEnergyNo,targetEnergyNo,
									fileName_base, inp.getOutputDir(), inp.getOutputFileBase(),
                                    regmode,msk,inp.getNumTrial(),inp.getLambda_t(),last));*/
                this_thread::sleep_for(chrono::seconds((int)((endEnergyNo-startEnergyNo+1)*dA[j]*0.02)));
				//this_thread::sleep_for(chrono::seconds(40));
                i+=dA[j];
				if (i > endAngleNo) break;
                else continue;
            }else if (th[j].joinable()) {
				/*bool last = (i + dA[j] - 1>endAngleNo - queues.size()*dA[j]);
                th[j].join();
				th[j] = thread(imageReg_thread, queues[j], kernels[cotextID_OfQueue[j]],//共有context時
                               dark_buffers[cotextID_OfQueue[j]],
                               I0_target_buffers[cotextID_OfQueue[j]],
                               I0_sample_buffers[cotextID_OfQueue[j]],
								i, min(i + dA[j] - 1, endAngleNo),
                               startEnergyNo,endEnergyNo,targetEnergyNo,
							   fileName_base, inp.getOutputDir(), inp.getOutputFileBase(),
                               regmode,msk,inp.getNumTrial(),inp.getLambda_t(),last);*/
				bool last = (i + dA[j] - 1>endAngleNo - plat_dev_list.contextsize()*dA[j]);//非共有context時
				th[j].join();
				th[j] = thread(imageReg_thread, plat_dev_list.queue(j, 0), kernels[j],
                                    dark_buffers[j],
                                    I0_target_buffers[j],
                                    I0_sample_buffers[j],
                                    i, min(i + dA[j] - 1, endAngleNo),
                                    startEnergyNo,endEnergyNo,targetEnergyNo,
                                    fileName_base, inp.getOutputDir(), inp.getOutputFileBase(),
                                    regmode,msk,inp.getNumTrial(),inp.getLambda_t(),last);
				i+=dA[j];
				if (i > endAngleNo) break;
            } else{
                this_thread::sleep_for(chrono::milliseconds(100));
            }

			if (i > endAngleNo) break;
        }
    }
    
	for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        if (th[j].joinable()) th[j].join();
    }

    
    return 0;
}