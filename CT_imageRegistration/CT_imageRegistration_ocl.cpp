﻿//
//  CT_ImageRegistration_C++.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/02/06.
//  Copyright (c) 2015 Nozomu Ishiguro. All rights reserved.
//

#include "CTXAFS.hpp"
#ifdef XANES_FIT
#include "XANES_fitting.hpp"
#include "XANES_fit_cl.hpp"
#include "LevenbergMarquardt_cl.hpp"
#endif

int mt_output_thread(int startAngleNo, int EndAngleNo,
                     input_parameter inp,
                     vector<float*> mt_outputs, vector<float*> p, vector<float*> p_err,
                     regMode regmode,int thread_id){
    
	//スレッドを待機/ロック
	m2.lock();
    string output_dir=inp.getOutputDir();
    string output_base=inp.getOutputFileBase();
    int startEnergyNo=inp.getStartEnergyNo();
    int endEnergyNo=inp.getEndEnergyNo();
	int imageSizeM = inp.getImageSizeM();
    
    string shift_dir=output_dir+ "/imageRegShift";
    MKDIR(shift_dir.c_str());
    
    const int p_num = regmode.get_p_num();
    //const int dA = EndAngleNo-startAngleNo+1;
    
    ostringstream oss;
    for (int j=startAngleNo; j<=EndAngleNo; j++) {
        string fileName_output= shift_dir+ AnumTagString(j,"/shift", ".txt");
        ofstream ofs(fileName_output,ios::out|ios::trunc);
        ofs<<regmode.ofs_transpara();
        
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            string fileName_output= output_dir+ "/" + EnumTagString(i,"","/") + AnumTagString(j,output_base, ".raw");
            oss << "output file: " << fileName_output <<endl;
            outputRawFile_stream(fileName_output,mt_outputs[i-startEnergyNo]+(j-startAngleNo)*imageSizeM,imageSizeM);
            
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
    //スレッドをアンロック
    m2.unlock();
    
    
    //delete [] mt_outputs_pointer;
	for (int i = startEnergyNo; i <= endEnergyNo; i++) {
		delete [] mt_outputs[i - startEnergyNo];
        delete [] p[i - startEnergyNo];
        delete [] p_err[i - startEnergyNo];
	}


    return 0;
}

int mt_conversion(cl::CommandQueue queue,cl::Kernel kernel,
                  cl::Buffer dark_buffer,cl::Buffer I0_buffer,
                  cl::Buffer mt_buffer,cl::Image2DArray mt_image,cl::Image2DArray mt_outputImg,
                  const cl::NDRange global_item_size,const cl::NDRange local_item_size,
                  unsigned short *It_pointer,int dA, mask msk, bool refBool, int imageSizeM){
    
	cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
    
	cl::Buffer It_buffer(context, CL_MEM_READ_ONLY, sizeof(cl_ushort)*(imageSizeM+32)*dA, 0, NULL);
	queue.enqueueWriteBuffer(It_buffer, CL_TRUE, 0, sizeof(cl_ushort)*(imageSizeM+32)*dA, It_pointer, NULL, NULL);
	kernel.setArg(0, dark_buffer);
    kernel.setArg(1, I0_buffer);
    kernel.setArg(2, It_buffer);
    kernel.setArg(3, mt_buffer);
    kernel.setArg(4, mt_image);
    kernel.setArg(5, mt_outputImg);
    if (refBool) {
        kernel.setArg(6, msk.refMask_shape);
        kernel.setArg(7, msk.refMask_x);
        kernel.setArg(8, msk.refMask_y);
        kernel.setArg(9, msk.refMask_width);
        kernel.setArg(10, msk.refMask_height);
        kernel.setArg(11, msk.refMask_angle);
    }else{
        kernel.setArg(6, msk.sampleMask_shape);
        kernel.setArg(7, msk.sampleMask_x);
        kernel.setArg(8, msk.sampleMask_y);
        kernel.setArg(9, msk.sampleMask_width);
        kernel.setArg(10, msk.sampleMask_height);
        kernel.setArg(11, msk.sampleMask_angle);
    }
    kernel.setArg(12, 0);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_item_size, local_item_size, NULL, NULL);
    queue.finish();
    
    return 0;
}

int imageReg_thread(cl::CommandQueue command_queue, CL_objects CLO,
                    unsigned short *It_img_target,vector<unsigned short*> It_img_sample,
                    int startAngleNo,int EndAngleNo,
					input_parameter inp, regMode regmode, mask msk, int thread_id){
	string errorArert;
	try {
		errorArert = "initialize";
		int imageSizeX = inp.getImageSizeX();
        int imageSizeY = inp.getImageSizeY();
        int imageSizeM = inp.getImageSizeM();
        
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
        string devicename = device.getInfo<CL_DEVICE_NAME>();
		size_t WorkGroupSize = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(),imageSizeX);
        const int dA=EndAngleNo-startAngleNo+1;
        const int p_num = regmode.get_p_num();
        cl::ImageFormat format(CL_RG,CL_FLOAT);
        int startEnergyNo=inp.getStartEnergyNo();
        int endEnergyNo=inp.getEndEnergyNo();
        int targetEnergyNo=inp.getTargetEnergyNo();
        float lambda=inp.getLambda_t();
        int Num_trial=inp.getNumTrial();
        
        
        // p_vec, p_err_vec, mt_sample
        vector<float*>p_vec;
        vector<float*>p_err_vec;
        vector<float*>mt_sample_img;
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            p_vec.push_back(new float[p_num*dA]);
            p_err_vec.push_back(new float[p_num*dA]);
            mt_sample_img.push_back(new float[imageSizeM*dA]);
		}
        
        
        //Buffer declaration
		errorArert="Buffer declaration";
        cl::Buffer dark_buffer=CLO.dark_buffer;
        cl::Buffer I0_target_buffer=CLO.I0_target_buffer;
        vector<cl::Buffer> I0_sample_buffers=CLO.I0_sample_buffers;
		cl::Buffer mt_target_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*imageSizeM*dA, 0, NULL);
		cl::Buffer mt_sample_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*imageSizeM*dA, 0, NULL);
        vector<cl::Image2DArray> mt_target_image;
        vector<cl::Image2DArray> mt_sample_image;
        for (int i=0; i<4; i++) {
            int mergeN = 1<<i;
            mt_target_image.push_back(cl::Image2DArray(context, CL_MEM_READ_WRITE,format,dA,
                                                       imageSizeX/mergeN,imageSizeY/mergeN,
                                                       0,0,NULL,NULL));
            mt_sample_image.push_back(cl::Image2DArray(context, CL_MEM_READ_WRITE,format,dA,
                                                       imageSizeX/mergeN,imageSizeY/mergeN,
                                                       0,0,NULL,NULL));
        }
        cl::Image2DArray mt_target_outputImg(context, CL_MEM_READ_WRITE,format,dA,imageSizeX,imageSizeY,0,0,NULL,NULL);
        cl::Image2DArray mt_sample_outputImg(context, CL_MEM_READ_WRITE,format,dA,imageSizeX,imageSizeY,0,0,NULL,NULL);
        cl::Buffer p_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*p_num*dA, 0, NULL);
        cl::Buffer dp_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*p_num*dA, 0, NULL);
        cl::Buffer p_cnd_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*p_num*dA, 0, NULL);
        cl::Buffer p_err_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*p_num*dA, 0, NULL);
        cl::Buffer p_fix_buffer(context, CL_MEM_READ_ONLY, sizeof(cl_char)*p_num, 0, NULL);
        cl::Buffer p_target_buffer(context, CL_MEM_READ_ONLY, sizeof(cl_float)*p_num*dA, 0, NULL);
        cl::Buffer lambda_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dA, 0, NULL);
        cl::Buffer nyu_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dA, 0, NULL);
        cl::Buffer dL_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dA, 0, NULL);
        cl::Buffer rho_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dA, 0, NULL);
        cl::Buffer dF2old_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dA, 0, NULL);
        cl::Buffer dF2new_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dA, 0, NULL);
        cl::Buffer tJdF_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*p_num*dA, 0, NULL);
        cl::Buffer tJJ_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*p_num*(p_num+1)/2*dA, 0, NULL);
        cl::Buffer mean_sample_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dA, 0, NULL);
        cl::Buffer mean_target_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dA, 0, NULL);
        
        
        
        //kernel dimension declaration
        const cl::NDRange global_item_size1(WorkGroupSize*dA,imageSizeY,1);
        const cl::NDRange local_item_size1(WorkGroupSize,1,1);
        const cl::NDRange global_item_size2(dA,1,1);
        const cl::NDRange local_item_size2(1,1,1);
        const cl::NDRange global_item_size3(dA,1,p_num);
        const cl::NDRange local_item_size3(1,1,1);
        
        
        //Energy loop setting
        vector<int> LoopEndenergyNo={startEnergyNo,endEnergyNo};
        vector<int> LoopStartenergyNo;
        LoopStartenergyNo.push_back(min(targetEnergyNo, endEnergyNo));
        LoopStartenergyNo.push_back(max(targetEnergyNo, startEnergyNo));        
        

		//target mt conversion
		errorArert = "target mt conversion";
        cl::Kernel kernel_mt = CLO.getKernel("mt_conversion");
        mt_conversion(command_queue,kernel_mt,dark_buffer,I0_target_buffer,
                      mt_target_buffer,mt_target_image[0], mt_target_outputImg,
                      global_item_size1,local_item_size1,It_img_target,dA,msk,true,imageSizeM);
        delete [] It_img_target;
        
        
        //estimate mean of mt_target
        cl::Kernel kernel_mean = CLO.getKernel("estimateImgMean");
        kernel_mean.setArg(0, mt_target_image[0]);
        kernel_mean.setArg(1, mean_target_buffer);
        kernel_mean.setArg(2, cl::Local(sizeof(cl_float)*WorkGroupSize));
        command_queue.enqueueNDRangeKernel(kernel_mean, cl::NullRange, global_item_size1, local_item_size1, NULL, NULL);
        command_queue.finish();
        
        
        //target image reg parameter (p_buffer) initialize
		errorArert = "target parameter initialize";
        for (int k=0; k<dA; k++) {
            for (int p=0; p<p_num; p++) {
                command_queue.enqueueWriteBuffer(p_target_buffer,CL_FALSE,sizeof(cl_float)*(k+p*dA), sizeof(cl_float),&regmode.p_ini[p],NULL,NULL);
            }
            command_queue.finish();
        }
        
        
        
        //set p_fix
		errorArert = "set p_fix";
        command_queue.enqueueWriteBuffer(p_fix_buffer, CL_TRUE, 0, sizeof(cl_char)*p_num,regmode.p_fix,NULL,NULL);
		command_queue.finish();
        
        
        //It_target merged image create
		errorArert = "target mt image merge";
        cl::Kernel kernel_merge = CLO.getKernel("merge");
        if (regmode.get_regModeNo()>=0) {
            for (int i=1; i<4; i++) {
                int mergeN = 1<<i;
                int localsize = min((int)WorkGroupSize,imageSizeX/mergeN);
                const cl::NDRange global_item_size_merge(localsize*dA,1,1);
                const cl::NDRange local_item_size_merge(localsize,1,1);
                
                kernel_merge.setArg(0, mt_target_image[0]);
                kernel_merge.setArg(1, mt_target_image[i]);
                kernel_merge.setArg(2, mergeN);
                command_queue.enqueueNDRangeKernel(kernel_merge, cl::NullRange, global_item_size_merge, local_item_size_merge, NULL, NULL);
                command_queue.finish();
            }
		}
		
        
        
        //process when (sample Enegry No. == target Energy No.)
		errorArert = "imagereg at target Energy No";
        cl::Kernel kernel_output = CLO.getKernel("output_imgReg_result");
        if ((targetEnergyNo>=startEnergyNo)&(targetEnergyNo<=endEnergyNo)) {
            if (regmode.get_regModeNo()>=0) {
                //kernel setArgs of outputing image reg results to buffer
                kernel_output.setArg(0, mt_target_outputImg);
                kernel_output.setArg(1, mt_target_buffer);
                kernel_output.setArg(2, p_target_buffer);
                
                //output image reg results to buffer
                command_queue.enqueueNDRangeKernel(kernel_output, NULL, global_item_size1, local_item_size1, NULL, NULL);
                command_queue.finish();
            }
                
            command_queue.enqueueReadBuffer(mt_target_buffer, CL_TRUE, 0, sizeof(cl_float)*imageSizeM*dA, mt_sample_img[targetEnergyNo-startEnergyNo], NULL, NULL);
            command_queue.finish();
            
            //move target image reg parameter from GPU to memory
            for (int k=0; k<dA; k++) {
                for (int p=0; p<p_num; p++) {
                    command_queue.enqueueReadBuffer(p_target_buffer, CL_FALSE, sizeof(cl_float)*(k+p*dA),sizeof(cl_float),&p_vec[targetEnergyNo-startEnergyNo][p+p_num*k],NULL,NULL);
                }
            }
			command_queue.finish();
			ostringstream oss;
            for (int j=startAngleNo; j<=EndAngleNo; j++) {
                for (int t=0; t<p_num; t++) {
                    //p_vec[targetEnergyNo - startEnergyNo][t+p_num*(j-startAngleNo)]=0;
                    p_err_vec[targetEnergyNo - startEnergyNo][t+p_num*(j-startAngleNo)]=0.0f;
                }
                
                oss << "Device("<<thread_id+1<<"): "<< devicename << ", angle: "<<j<< ", energy: "<<targetEnergyNo<<endl;
                oss <<regmode.get_oss_target();
            }
            cout << oss.str();
        }
        

        //kernel setArgs of It_sample merged image create
		errorArert = "kernel setting";
        kernel_merge.setArg(0, mt_sample_image[0]);
        //kernel setArgs of ImageReg1
        cl::Kernel kernel_imgReg1 = CLO.getKernel("imageReg1");
        kernel_imgReg1.setArg(2, p_buffer);
        kernel_imgReg1.setArg(3, p_target_buffer);
        kernel_imgReg1.setArg(4, p_fix_buffer);
        kernel_imgReg1.setArg(5, dF2old_buffer);
        kernel_imgReg1.setArg(6, tJdF_buffer);
        kernel_imgReg1.setArg(7, tJJ_buffer);
        kernel_imgReg1.setArg(10, mean_target_buffer);
        kernel_imgReg1.setArg(11, mean_sample_buffer);
        //kernel setArgs of LM
		errorArert = "kernel setting LM";
        cl::Kernel kernel_LM = CLO.getKernel("LevenbergMarquardt");
        kernel_LM.setArg(0, tJdF_buffer);
        kernel_LM.setArg(1, tJJ_buffer);
        kernel_LM.setArg(2, dp_buffer);
        kernel_LM.setArg(3, lambda_buffer);
        kernel_LM.setArg(4, p_fix_buffer);
        //kernel setArgs of dL estimation
		errorArert = "kernel setting dL" ;
        cl::Kernel kernel_dL = CLO.getKernel("estimate_dL");
        kernel_dL.setArg(0, dp_buffer);
        kernel_dL.setArg(1, tJJ_buffer);
        kernel_dL.setArg(2, tJdF_buffer);
        kernel_dL.setArg(3, lambda_buffer);
        kernel_dL.setArg(4, dL_buffer);
        //kernel setArgs of new parameter cnd
		errorArert = "kernel setting cnd";
        cl::Kernel kernel_cnd = CLO.getKernel("updatePara");
        kernel_cnd.setArg(0, dp_buffer);
        kernel_cnd.setArg(1, p_cnd_buffer);
        kernel_cnd.setArg(2, (cl_int)0);
        kernel_cnd.setArg(3, (cl_int)0);
        //kernel setArgs of ImageReg2
		errorArert = "kernel setting imageReg2";
        cl::Kernel kernel_imgReg2 = CLO.getKernel("imageReg2");
        kernel_imgReg2.setArg(2, p_cnd_buffer);
        kernel_imgReg2.setArg(3, p_target_buffer);
        kernel_imgReg2.setArg(4, dF2new_buffer);
        kernel_imgReg2.setArg(7, mean_target_buffer);
        kernel_imgReg2.setArg(8, mean_sample_buffer);
        //evaluate cnd
		errorArert = "kernel setting eval";
        cl::Kernel kernel_eval = CLO.getKernel("evaluateUpdateCandidate");
        kernel_eval.setArg(0, tJdF_buffer);
        kernel_eval.setArg(1, tJJ_buffer);
        kernel_eval.setArg(2, lambda_buffer);
        kernel_eval.setArg(3, nyu_buffer);
        kernel_eval.setArg(4, dF2old_buffer);
        kernel_eval.setArg(5, dF2new_buffer);
        kernel_eval.setArg(6, dL_buffer);
        kernel_eval.setArg(7, rho_buffer);
        //update to new para or hold to old para
		errorArert = "kernel setting UorH";
        cl::Kernel kernel_UorH = CLO.getKernel("updateOrHold");
        kernel_UorH.setArg(0, p_buffer);
        kernel_UorH.setArg(1, p_cnd_buffer);
        kernel_UorH.setArg(2, rho_buffer);
		//kernel setArgs of parameter error estimation
        cl::Kernel kernel_error = CLO.getKernel("estimateParaError");
		errorArert = "kernel setting error create";
        kernel_error.setArg(0, p_err_buffer);
        kernel_error.setArg(1, tJJ_buffer);
        //kernel setArgs of outputing image reg results to buffer
		errorArert = "kernel setting output";
        kernel_output.setArg(0, mt_sample_outputImg);
        kernel_output.setArg(1, mt_sample_buffer);
        kernel_output.setArg(2, p_buffer);
        //estimate mt_sample mean
        kernel_mean.setArg(0, mt_sample_image[0]);
        kernel_mean.setArg(1, mean_sample_buffer);
        
        
        for (int s=0; s<2; s++) {//1st cycle:i<targetEnergyNo, 2nd cycle:i>targetEnergyNo
            int di=(-1+2*s); //1st -1, 2nd +1
            if (startEnergyNo==endEnergyNo){
                if(startEnergyNo==targetEnergyNo) break;
            }else if ((LoopStartenergyNo[s]+di)*di>LoopEndenergyNo[s]*di) {
                continue;
            }
            
            //image reg parameter (p_buffer) initialize
			errorArert = "parameter initialize";
            for (int k=0; k<dA; k++) {
                for (int p=0; p<p_num; p++) {
                    command_queue.enqueueWriteBuffer(p_buffer,CL_FALSE,sizeof(cl_float)*(k+p*dA), sizeof(cl_float),&regmode.p_ini[p],NULL,NULL);
                }
                command_queue.finish();
            }
            
            int ds = (LoopStartenergyNo[s]==targetEnergyNo) ? di:0;
            for (int i=LoopStartenergyNo[s]+ds; i*di<=LoopEndenergyNo[s]*di; i+=di) {

                //sample mt conversion
				errorArert = "mt conversion";
                mt_conversion(command_queue,kernel_mt,dark_buffer,I0_sample_buffers[i-startEnergyNo],mt_sample_buffer,mt_sample_image[0], mt_sample_outputImg,global_item_size1,local_item_size1,It_img_sample[i-startEnergyNo],dA,msk,true,imageSizeM);
                
                
                //estimate mean of mt_sample
                command_queue.enqueueNDRangeKernel(kernel_mean, cl::NullRange, global_item_size1, local_item_size1, NULL, NULL);
                command_queue.finish();
                
        
                if (regmode.get_regModeNo()>=0) {
                    
                    //It_sample merged image create
					errorArert = "mt merge";
                    for (int j=1; j<4; j++) {
                        unsigned int mergeN = 1<<j;
                        int localsize = min((unsigned int)WorkGroupSize,imageSizeX/mergeN);
                        const cl::NDRange global_item_size_merge(localsize*dA,1,1);
                        const cl::NDRange local_item_size_merge(localsize,1,1);
                        kernel_merge.setArg(1, mt_sample_image[j]);
                        kernel_merge.setArg(2, mergeN);
                        command_queue.enqueueNDRangeKernel(kernel_merge, cl::NullRange, global_item_size_merge, local_item_size_merge, NULL, NULL);
                        command_queue.finish();
                    }
                    
                    
                    //lambda_buffer reset
                    command_queue.enqueueFillBuffer(lambda_buffer, (cl_float)lambda, 0, sizeof(cl_float)*dA,NULL,NULL);
                    command_queue.finish();
                    
                    
                    //Image registration
                    for (int j=3; j>=0; j--) {
                        unsigned int mergeN = 1<<j;
                        //cout<<mergeN<<endl;
						errorArert = "imagereg setting";
                        int localsize = min((int)WorkGroupSize,imageSizeX/(int)mergeN);
                        const cl::NDRange global_item_size_reg(localsize*dA,1,1);
                        const cl::NDRange local_item_size_reg(localsize,1,1);
                        kernel_imgReg1.setArg(0, mt_target_image[j]);
                        kernel_imgReg1.setArg(1, mt_sample_image[j]);
                        kernel_imgReg1.setArg(8, (cl_uint)mergeN);
                        kernel_imgReg1.setArg(9, cl::Local(sizeof(cl_float)*localsize));//locmem
                        kernel_imgReg2.setArg(0, mt_target_image[j]);
                        kernel_imgReg2.setArg(1, mt_sample_image[j]);
                        kernel_imgReg2.setArg(5, (cl_uint)mergeN);
                        kernel_imgReg2.setArg(6, cl::Local(sizeof(cl_float)*localsize));//locmem
                        for (int trial=0; trial < Num_trial; trial++) {
                            //imageReg1:estimate dF2(old), tJJ, tJdF
							errorArert = "imagereg1";
                            command_queue.enqueueNDRangeKernel(kernel_imgReg1, NULL, global_item_size_reg, local_item_size_reg, NULL, NULL);
                            command_queue.finish();
							/*float* tJJ;
							tJJ = new float[p_num*(p_num + 1) / 2 * dA];
							command_queue.enqueueReadBuffer(tJJ_buffer, CL_TRUE, 0, sizeof(cl_float)*p_num*(p_num+1)/2*dA, tJJ);
							cout << "tJJ" << endl;
							for (int p = 0; p < p_num*(p_num + 1) / 2; p++) {
								cout << tJJ[p*dA] << endl;
							}
							float* tJdF;
							tJdF = new float[p_num*dA];
							command_queue.enqueueReadBuffer(tJdF_buffer, CL_TRUE, 0, sizeof(cl_float)*p_num*dA, tJdF);
							cout << "tJdF" << endl;
							for (int p = 0; p < p_num; p++) {
								cout << tJdF[p*dA] << endl;
							}*/
                            
                            //LM
							errorArert = "LM";
                            command_queue.enqueueNDRangeKernel(kernel_LM, NULL, global_item_size2, local_item_size2, NULL, NULL);
                            command_queue.finish();
							/*float* dp;
							dp = new float[p_num*dA];
							command_queue.enqueueReadBuffer(dp_buffer, CL_TRUE, 0, sizeof(cl_float)*p_num*dA, dp);
							cout << "dp" << endl;
							for (int p = 0; p < p_num; p++) {
								cout << dp[p*dA] << endl;
							}*/
                            
                            //dL estimation
							errorArert = "dL";
                            command_queue.enqueueNDRangeKernel(kernel_dL, NULL, global_item_size2, local_item_size2, NULL, NULL);
                            command_queue.finish();
                            
                            //estimate parameter cnd
							errorArert = "cnd";
                            command_queue.enqueueCopyBuffer(p_buffer, p_cnd_buffer, 0, 0, sizeof(cl_float)*p_num*dA);
                            command_queue.enqueueNDRangeKernel(kernel_cnd, NULL, global_item_size3, local_item_size3, NULL, NULL);
                            command_queue.finish();
                            
                            //imageReg2:estimate dF2(new)
							errorArert = "imagereg2";
                            command_queue.enqueueNDRangeKernel(kernel_imgReg2, NULL, global_item_size_reg, local_item_size_reg, NULL, NULL);
                            command_queue.finish();
                            
                            //evaluate cnd
							errorArert = "eval cnd";
                            command_queue.enqueueNDRangeKernel(kernel_eval, NULL, global_item_size2, local_item_size2, NULL, NULL);
                            command_queue.finish();
                            
                            //update to new para or hold to old para
							errorArert = "update or hold";
                            command_queue.enqueueNDRangeKernel(kernel_UorH, NULL, global_item_size3, local_item_size3, NULL, NULL);
                            command_queue.finish();
                        }
                    }

					//error create
					errorArert = "error estimation";
                    command_queue.enqueueNDRangeKernel(kernel_error, NULL, global_item_size3, local_item_size3, NULL, NULL);
                    command_queue.finish();
                    
                    
                    //output image reg results to buffer
					errorArert = "output";
                    command_queue.enqueueNDRangeKernel(kernel_output, NULL, global_item_size1, local_item_size1, NULL, NULL);
                    command_queue.finish();
                    
                    
                    //read p_buffer & p_err_buffer from GPU to memory
                    for (int k=0; k<dA; k++) {
                        for (int p=0; p<p_num; p++) {
                            command_queue.enqueueReadBuffer(p_buffer, CL_FALSE, sizeof(cl_float)*(k+p*dA),sizeof(cl_float),&p_vec[i-startEnergyNo][p+k*p_num],NULL,NULL);
                            command_queue.enqueueReadBuffer(p_err_buffer, CL_FALSE, sizeof(cl_float)*(p+p_num*k),sizeof(cl_float),&p_err_vec[i-startEnergyNo][p+p_num*k],NULL,NULL);
                        }
                    }
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
                        
                        if(regmode.p_fix[n]==0.0f) p_precision[n]=3;
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
					oss << "Device(" << thread_id+1 << "): " << devicename << ", angle: " << startAngleNo + k << ", energy: " << i << endl;
                    oss << regmode.oss_sample(p_vec[i-startEnergyNo]+p_num*k, p_err_vec[i-startEnergyNo]+p_num*k,p_precision,p_err_precision);
                }
                cout << oss.str();

				//read output buffer
                command_queue.enqueueReadBuffer(mt_sample_buffer, CL_TRUE, 0, sizeof(cl_float)*imageSizeM*dA, mt_sample_img[i - startEnergyNo], NULL, NULL);
                command_queue.finish();
            }
        }
        
		for (int i = startEnergyNo; i <= endEnergyNo; i++) {
			delete[] It_img_sample[i - startEnergyNo];
		}
        
        
        
#ifdef XANES_FIT //XANES fitting batch
        for (int j=startAngleNo; j<=EndAngleNo; j++) {
            XANES_fit_thread(command_queue,CLO.program,
                             CLO.fiteq,j,thread_id,inp,
							 CLO.energy_buffer, CLO.C_matrix_buffer, CLO.D_vector_buffer,
                             CLO.freeFix_buffer,CLO.refSpectra,CLO.funcMode_buffer,
							 mt_sample_img,(j-startAngleNo)*imageSizeM);
        }
#endif
        //output imageReg
        if (inp.getImgRegOutput()) {
            output_th[thread_id].join();
            output_th[thread_id]=thread(mt_output_thread,
                                        startAngleNo,EndAngleNo,inp,
                                        move(mt_sample_img),move(p_vec),move(p_err_vec),regmode,thread_id);
		}
		else {
			for (int i = startEnergyNo; i <= endEnergyNo; i++) {
				delete[] mt_sample_img[i - startEnergyNo];
                delete [] p_vec[i - startEnergyNo];
                delete [] p_err_vec[i - startEnergyNo];
			}
            
        }
		

        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ") " <<errorArert<< endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(ret.err());
    }
    
    return 0;
}

int data_input_thread(int thread_id, cl::CommandQueue command_queue, CL_objects CLO,
                      int startAngleNo,int EndAngleNo,
                      string fileName_base, input_parameter inp,
                      regMode regmode, mask msk){


	//スレッドを待機/ロック
	m1.lock();
    
    int startEnergyNo=inp.getStartEnergyNo();
    int endEnergyNo=inp.getEndEnergyNo();
    int targetEnergyNo=inp.getTargetEnergyNo();
	int imageSizeM = inp.getImageSizeM();
    
	ostringstream oss;
	if (startAngleNo == EndAngleNo) {
		oss << "(" << thread_id + 1 << ") reading It files of angle " << startAngleNo << endl << endl;
	}
	else {
		oss << "(" << thread_id + 1 << ") reading It files of angle " << startAngleNo << "-" << EndAngleNo << endl << endl;
	}
	cout << oss.str();

    const int dA=EndAngleNo-startAngleNo+1;
    /*target It data input*/
    unsigned short *It_img_target;
    It_img_target = new unsigned short[(imageSizeM+32)*dA];
    string fileName_It_target = EnumTagString(targetEnergyNo,fileName_base,".his");
    readHisFile_stream(fileName_It_target,startAngleNo,EndAngleNo,It_img_target,imageSizeM);
    /* Sample It data input */
    vector<unsigned short*>It_img_sample;
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
        It_img_sample.push_back(new unsigned short[(imageSizeM+32)*dA]);
        string fileName_It = EnumTagString(i,fileName_base,".his");
        readHisFile_stream(fileName_It,startAngleNo,EndAngleNo,It_img_sample[i-startEnergyNo],imageSizeM);
    }
    
	//スレッドをアンロック
	m1.unlock();

    //image_reg
    imageReg_th[thread_id].join();
    imageReg_th[thread_id] = thread(imageReg_thread,
                                    command_queue, CLO,
                                    move(It_img_target),move(It_img_sample),
                                    startAngleNo, EndAngleNo,
                                    inp,regmode,msk,thread_id);

    return 0;
}

//dummy thread
static int dummy(){
    return 0;
}

int mergeRawhisBuffers(cl::CommandQueue queue, cl::Kernel kernel,
                       const cl::NDRange global_item_size,const cl::NDRange local_item_size,
                       unsigned short *img, cl::Buffer img_buffer,int mergeN,int imageSizeM){
    
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
    
    cl::Buffer rawhis_buffer(context, CL_MEM_READ_ONLY, sizeof(cl_ushort)*(imageSizeM+32)*mergeN, 0, NULL);
    
    queue.enqueueWriteBuffer(rawhis_buffer, CL_TRUE, 0, sizeof(cl_ushort)*(imageSizeM+32)*mergeN, img, NULL, NULL);
    kernel.setArg(0, rawhis_buffer);
    kernel.setArg(1, img_buffer);
    kernel.setArg(2, mergeN);
    queue.enqueueNDRangeKernel(kernel, NULL, global_item_size,local_item_size, NULL, NULL);
	queue.finish();
    
    return 0;
}


int imageRegistlation_ocl(string fileName_base, input_parameter inp,
                          OCL_platform_device plat_dev_list, regMode regmode)
{
    int startEnergyNo=inp.getStartEnergyNo();
    int endEnergyNo=inp.getEndEnergyNo();
    int targetEnergyNo=inp.getTargetEnergyNo();
    int startAngleNo=inp.getStartAngleNo();
    int endAngleNo=inp.getEndAngleNo();
    int imageSizeX = inp.getImageSizeX();
    int imageSizeY = inp.getImageSizeY();
	int imageSizeM = inp.getImageSizeM();
    
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
        string fileName_output = inp.getOutputDir() + "/" + EnumTagString(i,"","");
        MKDIR(fileName_output.c_str());
    }
    
    //OpenCL objects class
    vector<CL_objects> CLO;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        CL_objects CLO_contx;
        CLO.push_back(CLO_contx);
    }

	int scanN = inp.getScanN();
    
#ifdef XANES_FIT
    float startEnergy=inp.getStartEnergy();
    float endEnergy=inp.getEndEnergy();
    float E0=inp.getE0();
    
    //create output dir (fitting)
    //fiteq = fitting_eq::fitting_eq(inp);
	fitting_eq fiteq(inp);
    for (int i=0; i<fiteq.ParaSize(); i++) {
        char buffer=fiteq.freefix_para()[i];
        if (atoi(&buffer)==1) {
            string fileName_output = inp.getFittingOutputDir() + "/"+fiteq.param_name(i);
            MKDIR(fileName_output.c_str());
        }
    }
    
    
    //energy file input & processing
    ifstream energy_ifs(inp.getEnergyFilePath(),ios::in);
    if (!energy_ifs.is_open()) {
        cout << "energy file not found" << endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(-1);
    }
    vector<float> energy;
    int i=startEnergyNo, fittingStartEnergyNo=0, fittingEndEnergyNo=0;
    do {
        float a;
        energy_ifs>>a;
        if (energy_ifs.eof()) break;
        cout<<i<<": "<<a;
        if ((a>=startEnergy)&(a<=endEnergy)) {
            energy.push_back(a-E0);
            cout<<" <- fitting range";
            fittingEndEnergyNo = i;
        } else if(a<startEnergy) {
            fittingStartEnergyNo = i+1;
        }
        cout<<endl;
        i++;
    } while (!energy_ifs.eof()||i>endEnergyNo);
    int num_energy=fittingEndEnergyNo-fittingStartEnergyNo+1;
    inp.setFittingStartEnergyNo(fittingStartEnergyNo);
    inp.setFittingEndEnergyNo(fittingEndEnergyNo);
    cout << "energy num for fitting: "<<num_energy<<endl<<endl;
    
    //kernel program source
    /*ifstream ifs("./XANES_fit.cl", ios::in);
     if(!ifs) {
     cerr << "   Failed to load kernel \n" << endl;
     return -1;
     }
     istreambuf_iterator<char> it(ifs);
     istreambuf_iterator<char> last;
     string kernel_src(it,last);
     ifs.close();*/
    
    
    //OpenCL Program
#endif
    
    //OpenCL Program
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        cl::Program program=regmode.buildImageRegProgram(plat_dev_list.context(i),imageSizeX,imageSizeY);
        CLO[i].addKernel(program,"mt_conversion");
        CLO[i].addKernel(program,"merge");
        CLO[i].addKernel(program,"imageReg1"); //estimate dF2(old), tJJ tJdF;
        CLO[i].addKernel(program,"LevenbergMarquardt");
        CLO[i].addKernel(program,"estimate_dL");
        CLO[i].addKernel(program,"updatePara");
        CLO[i].addKernel(program,"imageReg2"); //estimate dF2(new)
		CLO[i].addKernel(program, "evaluateUpdateCandidate");
        CLO[i].addKernel(program,"updateOrHold");
		CLO[i].addKernel(program, "estimateParaError");
        CLO[i].addKernel(program,"output_imgReg_result");
        CLO[i].addKernel(program,"merge_rawhisdata");
        
#ifdef XANES_FIT
        cl::Program::Sources source;
#if defined (OCL120)
        source.push_back(std::make_pair(kernel_fit_src.c_str(),kernel_fit_src.length()));
        source.push_back(std::make_pair(kernel_LM_src.c_str(),kernel_LM_src.length()));
#else
        source.push_back(kernel_fit_src);
        source.push_back(kernel_LM_src);
#endif
        cl::Program program_fit(plat_dev_list.context(i), source);
        //kernel build
		string option = "";
#ifdef DEBUG
        option += "-D DEBUG";
#endif
		option += kernel_preprocessor_nums(fiteq,inp);
        string GPUvendor =  plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0].getInfo<CL_DEVICE_VENDOR>();
        if(GPUvendor == "nvidia"){
            option += "-cl-nv-maxrregcount=64";
            //option += " -cl-nv-verbose -Werror";
        }
		else if (GPUvendor.find("NVIDIA Corporation") == 0) {
			option += " -cl-nv-maxrregcount=64";
		}
        program_fit.build(option.c_str());
        CLO[i].program = program_fit;
		CLO[i].fiteq = fiteq;
#endif
    }
    
    
    //display OCL device
    int t=0;
    vector<int> dA;
    vector<int> maxWorkSize;
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
            size_t device_pram_size[3]={0,0,0};
            plat_dev_list.dev(i,j).getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &device_pram_size);
            dA.push_back(min(inp.getNumParallel(),(int)device_pram_size[0]));
            cout<<"Number of working compute unit: "<<dA[t]<<"/"<<device_pram_size[0]<<endl<<endl;
            maxWorkSize.push_back((int)min((int)plat_dev_list.dev(i,j).getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), imageSizeX));
            t++;
        }
    }
    
    
    // dark data input
    cout << "Reading dark file..."<<endl;
    unsigned short *dark_img;
    dark_img = new unsigned short[(imageSizeM+32)*scanN];
    string fileName_dark = fileName_base+ "dark.his";
    readHisFile_stream(fileName_dark,1,scanN,dark_img,imageSizeM);
    
    
    // I0 sample data input
    cout << "Reading I0 files..."<<endl<<endl;
    vector<unsigned short*> I0_imgs;
    for (int i = startEnergyNo; i <= endEnergyNo; i++) {
        I0_imgs.push_back(new unsigned short[(imageSizeM+32)*scanN]);
        string fileName_I0 = EnumTagString(i,fileName_base,"_I0.his");
        int readHis_err=readHisFile_stream(fileName_I0,1,scanN,I0_imgs[i-startEnergyNo],imageSizeM);
        if (readHis_err<0) {
            endEnergyNo=i-1;
            cout <<"No more I0 at Energy No. at over"<<i-1<<endl;
            break;
        }
    }
    
        
    // I0 target data input
    unsigned short *I0_img_target;
    I0_img_target = new unsigned short[(imageSizeM+32)*scanN];
    string fileName_I0_target;
    fileName_I0_target =  EnumTagString(targetEnergyNo,fileName_base,"_I0.his");
    readHisFile_stream(fileName_I0_target,1,scanN,I0_img_target,imageSizeM);
    
    
    //create dark, I0_target, I0_sample buffers
    const int p_num = regmode.get_p_num();
	for (int i = 0; i<plat_dev_list.contextsize(); i++) {
        CLO[i].dark_buffer=cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
        CLO[i].I0_target_buffer=cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
        CLO[i].p_freefix_buffer=cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float)*p_num*dA[i], 0, NULL);
        
        const cl::NDRange global_item_size(imageSizeX,imageSizeY,1);
        const cl::NDRange local_item_size(maxWorkSize[i],1,1);
        
        cl::Kernel kernel_mergeHis = CLO[i].getKernel("merge_rawhisdata");
        //merge rawhis buffers to dark_buffer
        mergeRawhisBuffers(plat_dev_list.queue(i, 0), kernel_mergeHis, global_item_size, local_item_size, dark_img, CLO[i].dark_buffer,scanN,imageSizeM);
        
        //merge rawhis buffers to I0_target_buffer
        mergeRawhisBuffers(plat_dev_list.queue(i, 0), kernel_mergeHis, global_item_size, local_item_size, I0_img_target, CLO[i].I0_target_buffer,scanN,imageSizeM);
        
        //merge rawhis buffers to I0_sample_buffer
        for (int j = startEnergyNo; j <= endEnergyNo; j++) {
            CLO[i].I0_sample_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*imageSizeM, 0, NULL));
            mergeRawhisBuffers(plat_dev_list.queue(i, 0), kernel_mergeHis, global_item_size, local_item_size, I0_imgs[j-startEnergyNo], CLO[i].I0_sample_buffers[j-startEnergyNo],scanN,imageSizeM);
        }
#ifdef XANES_FIT
        cl::ImageFormat format(CL_R, CL_FLOAT);
		int paramsize = (int)fiteq.ParaSize();
        int contrainsize = (int)fiteq.constrain_size;
		CLO[i].energy_buffer=cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*num_energy, 0, NULL);
		CLO[i].C_matrix_buffer=cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*paramsize*contrainsize, 0, NULL);
		CLO[i].D_vector_buffer=cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*contrainsize, 0, NULL);
		CLO[i].freeFix_buffer=cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_char)*paramsize, 0, NULL);
        CLO[i].funcMode_buffer = cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_int)*fiteq.numFunc, 0, NULL);
        CLO[i].refSpectra = cl::Image1DArray(plat_dev_list.context(i), CL_MEM_READ_WRITE,format,fiteq.numLCF,IMAGE_SIZE_E, 0, NULL);
        

        //write buffers
        plat_dev_list.queue(i,0).enqueueWriteBuffer(CLO[i].energy_buffer, CL_FALSE, 0, sizeof(cl_float)*num_energy, &energy[0], NULL, NULL);
        for (int j = 0; j < contrainsize; j++) {
            plat_dev_list.queue(i, 0).enqueueWriteBuffer(CLO[i].C_matrix_buffer, CL_FALSE, sizeof(cl_float)*j*paramsize, sizeof(cl_float)*paramsize, &(fiteq.C_matrix[j][0]), NULL, NULL);
        }
        plat_dev_list.queue(i,0).enqueueWriteBuffer(CLO[i].D_vector_buffer, CL_FALSE, 0, sizeof(cl_float)*contrainsize, &fiteq.D_vector[0], NULL, NULL);
        plat_dev_list.queue(i,0).enqueueWriteBuffer(CLO[i].freeFix_buffer, CL_FALSE, 0, sizeof(cl_char)*paramsize, fiteq.freefix_para(), NULL, NULL);
        plat_dev_list.queue(i,0).enqueueWriteBuffer(CLO[i].funcMode_buffer, CL_FALSE, 0, sizeof(cl_int)*fiteq.numFunc, &(fiteq.funcmode[0]), NULL, NULL);
        plat_dev_list.queue(i, 0).finish();
        
        //write buffers for LCF
        cl::Kernel kernel_LCFstd(CLO[i].program,"redimension_refSpecta");
        cl::size_t<3> origin;
        cl::size_t<3> region;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        region[1] = 1;
        region[2] = 1;
        for (int j=0; j<fiteq.numLCF; j++) {
            cl::Image1D refSpectraRaw(plat_dev_list.context(i), CL_MEM_READ_ONLY,format, fiteq.LCFstd_size[j], 0, NULL);
            cl::Buffer refSpectraRawE(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*fiteq.LCFstd_size[j], 0, NULL);
            
            region[0] = fiteq.LCFstd_size[j];
            plat_dev_list.queue(i,0).enqueueWriteImage(refSpectraRaw,CL_FALSE,origin,region,0,0,&(fiteq.LCFstd_mt[j][0]));
            plat_dev_list.queue(i,0).enqueueWriteBuffer(refSpectraRawE, CL_FALSE,0,sizeof(cl_float)*fiteq.LCFstd_size[j], &(fiteq.LCFstd_E[j][0]));
            plat_dev_list.queue(i, 0).finish();
            
            kernel_LCFstd.setArg(0, CLO[i].refSpectra);
            kernel_LCFstd.setArg(1, refSpectraRaw);
            kernel_LCFstd.setArg(2, refSpectraRawE);
            kernel_LCFstd.setArg(3, (cl_int)fiteq.LCFstd_size[j]);
            kernel_LCFstd.setArg(4, (cl_int)j);
            
            plat_dev_list.queue(i,0).enqueueNDRangeKernel(kernel_LCFstd, NULL, global_item_size, local_item_size, NULL, NULL);
            plat_dev_list.queue(i,0).finish();
        }
#endif
    }
    delete[] dark_img;
    delete[] I0_img_target;
	for (int i = startEnergyNo; i <= endEnergyNo; i++) {
		delete[] I0_imgs[i-startEnergyNo];
	}
    
    //mask settings
    mask msk(inp);
    
    //start threads
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        input_th.push_back(thread(dummy));
        imageReg_th.push_back(thread(dummy));
		output_th.push_back(thread(dummy));
#ifdef XANES_FIT
        output_th_fit.push_back(thread(dummy));
#endif
    }
    for (int i=startAngleNo; i<=endAngleNo;) {
		for (int j = 0; j<plat_dev_list.contextsize(); j++) {
            if (input_th[j].joinable()) {
				input_th[j].join();
				input_th[j] = thread(data_input_thread,j,plat_dev_list.queue(j,0),CLO[j],
                                     i, min(i + dA[j] - 1, endAngleNo),
                                     fileName_base,inp,regmode,msk);
				i+=dA[j];
				if (i > endAngleNo) break;
            } else{
                this_thread::sleep_for(chrono::milliseconds(100));
            }

			if (i > endAngleNo) break;
        }
    }
    
	for (int j = 0; j<plat_dev_list.contextsize(); j++) {
		input_th[j].join();
	}
	for (int j = 0; j<plat_dev_list.contextsize(); j++) {
		imageReg_th[j].join();
	}
	for (int j = 0; j<plat_dev_list.contextsize(); j++) {
		output_th[j].join();
	}
#ifdef XANES_FIT
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        output_th_fit[j].join();
    }
#endif
    
    return 0;
}
