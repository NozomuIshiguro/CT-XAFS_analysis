//
//  imageRegCore.cpp
//  CT-XAFS_analysis
//
//  Created by Nozomu Ishiguro on 2017/08/28.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#include "imageReg.hpp"


int imageRegistration(cl::CommandQueue command_queue, CL_objects CLO,
                      vector<cl::Image2DArray> mt_target_image,
                      vector<cl::Image2DArray> mt_sample_image,
                      vector<cl::Image2DArray> weight_image,
                      cl::Image2DArray mt_sample_outputImg, cl::Buffer mt_sample_buffer,
                      cl::Buffer p_buffer, cl::Buffer p_target_buffer, cl::Buffer p_fix_buffer,
                      cl::Buffer p_cnd_buffer, cl::Buffer p_err_buffer,
                      cl::Buffer dF2old_buffer,cl::Buffer dF2new_buffer,cl::Buffer dF_buffer,
                      cl::Buffer tJJ_buffer, cl::Buffer tJdF_buffer, cl::Buffer dev_buffer,
                      cl::Buffer dp_buffer, cl::Buffer lambda_buffer, cl::Buffer dL_buffer,
                      cl::Buffer nyu_buffer, cl::Buffer rho_buffer,
                      vector<cl::Buffer> dF2X,vector<cl::Buffer> dFX, vector<cl::Buffer> tJJX,
                      vector<cl::Buffer> tJdFX, vector<cl::Buffer> devX,
                      int mergeLevel, int imageSizeX, int imageSizeY, int p_num, int dZ, float CI,
                      int num_trial, float lambda){
    
    cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
    cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
    size_t maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    
    size_t workGroupSize;
    //kernel dimension declaration
    //for L-M processes
    const cl::NDRange global_item_size2(dZ,1,1);
    const cl::NDRange local_item_size2(1,1,1);
    //for paramerter processes
    const cl::NDRange global_item_size3(dZ,1,p_num+2);
    const cl::NDRange local_item_size3(1,1,1);
    //for output
    workGroupSize = min(maxWorkGroupSize, (size_t)imageSizeX);
    const cl::NDRange global_item_size5(imageSizeX, imageSizeY, dZ);
    const cl::NDRange local_item_size5(workGroupSize, 1, 1);
    //for weight theshold
    int localsizeRX = min((int)maxWorkGroupSize, imageSizeX / (int)(1 << mergeLevel) / 2);
    int localsizeRY = min((int)maxWorkGroupSize, imageSizeY / (int)(1 << mergeLevel) / 2);
    const cl::NDRange global_item_size_RX(localsizeRX, imageSizeY / (1 << mergeLevel), dZ);
    const cl::NDRange local_item_size_RX(localsizeRX, 1, 1);
    const cl::NDRange global_item_size_RY(localsizeRY, dZ, 1);
    const cl::NDRange local_item_size_RY(localsizeRY, 1, 1);
    cl::size_t<3> origin, region;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    region[0] = imageSizeX / (1 << mergeLevel);
    region[1] = imageSizeY / (1 << mergeLevel);
    region[2] = dZ;
    cl_float4 ini = { 1.0f,0.0f,0.0f,0.0f };
    
    
    //kernel setArgs of ImageReg1
    cl::Kernel kernel_imgReg1X = CLO.getKernel("imageReg1X");
    kernel_imgReg1X.setArg(3, dF2new_buffer);
    kernel_imgReg1X.setArg(4, dF_buffer);
    kernel_imgReg1X.setArg(5, (cl_float)CI);
    kernel_imgReg1X.setArg(6, p_buffer);
    kernel_imgReg1X.setArg(7, p_target_buffer);
    kernel_imgReg1X.setArg(8, p_fix_buffer);
    cl::Kernel kernel_imgReg1Y = CLO.getKernel("imageReg1Y");
    kernel_imgReg1Y.setArg(4, dF2old_buffer);
    kernel_imgReg1Y.setArg(5, tJdF_buffer);
    kernel_imgReg1Y.setArg(6, tJJ_buffer);
    kernel_imgReg1Y.setArg(7, dev_buffer);
    //kernel setArgs of LM
    cl::Kernel kernel_LM = CLO.getKernel("LevenbergMarquardt");
    kernel_LM.setArg(0, tJdF_buffer);
    kernel_LM.setArg(1, tJJ_buffer);
    kernel_LM.setArg(2, dp_buffer);
    kernel_LM.setArg(3, lambda_buffer);
    kernel_LM.setArg(4, p_fix_buffer);
    //kernel setArgs of dL estimation
    cl::Kernel kernel_dL = CLO.getKernel("estimate_dL");
    kernel_dL.setArg(0, dp_buffer);
    kernel_dL.setArg(1, tJJ_buffer);
    kernel_dL.setArg(2, tJdF_buffer);
    kernel_dL.setArg(3, lambda_buffer);
    kernel_dL.setArg(4, dL_buffer);
    //kernel setArgs of new parameter cnd
    cl::Kernel kernel_cnd = CLO.getKernel("updatePara");
    kernel_cnd.setArg(0, dp_buffer);
    kernel_cnd.setArg(1, p_cnd_buffer);
    kernel_cnd.setArg(2, (cl_int)0);
    kernel_cnd.setArg(3, (cl_int)0);
    //kernel setArgs of ImageReg2
    cl::Kernel kernel_imgReg2X = CLO.getKernel("imageReg2X");
    kernel_imgReg2X.setArg(0, mt_target_image[mergeLevel]);
    kernel_imgReg2X.setArg(1, mt_sample_image[mergeLevel]);
    kernel_imgReg2X.setArg(2, weight_image[mergeLevel]);
    kernel_imgReg2X.setArg(3, p_cnd_buffer);
    kernel_imgReg2X.setArg(4, p_target_buffer);
    kernel_imgReg2X.setArg(5, dF2X[mergeLevel]);
    kernel_imgReg2X.setArg(6, dFX[mergeLevel]);
    kernel_imgReg2X.setArg(7, devX[mergeLevel]);
    kernel_imgReg2X.setArg(8, (cl_int)(1<<mergeLevel));
    kernel_imgReg2X.setArg(9, cl::Local(sizeof(cl_float)*localsizeRX*2));//locmem
    cl::Kernel kernel_imgReg2Y = CLO.getKernel("imageReg2Y");
    kernel_imgReg2Y.setArg(0, dF2X[mergeLevel]);
    kernel_imgReg2Y.setArg(1, dFX[mergeLevel]);
    kernel_imgReg2Y.setArg(2, devX[mergeLevel]);
    kernel_imgReg2Y.setArg(3, dF2new_buffer);
    kernel_imgReg2Y.setArg(4, dF_buffer);
    kernel_imgReg2Y.setArg(5, (cl_int)(1<<mergeLevel));
    kernel_imgReg2Y.setArg(6, cl::Local(sizeof(cl_float)*localsizeRY*2));
    //evaluate cnd
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
    cl::Kernel kernel_UorH = CLO.getKernel("updateOrHold");
    kernel_UorH.setArg(0, p_buffer);
    kernel_UorH.setArg(1, p_cnd_buffer);
    kernel_UorH.setArg(2, rho_buffer);
    //kernel setArgs of parameter error estimation
    cl::Kernel kernel_error = CLO.getKernel("estimateParaError");
    kernel_error.setArg(0, p_err_buffer);
    kernel_error.setArg(1, tJdF_buffer);
    kernel_error.setArg(2, tJJ_buffer);
    kernel_error.setArg(3, dev_buffer);
    //kernel setArgs of outputing image reg results to buffer
    cl::Kernel kernel_output = CLO.getKernel("output_imgReg_result");
    kernel_output.setArg(0, mt_sample_outputImg);
    kernel_output.setArg(1, mt_sample_buffer);
    kernel_output.setArg(2, p_buffer);
    
    
    //lambda_buffer reset
    command_queue.enqueueFillBuffer(lambda_buffer, (cl_float)lambda, 0, sizeof(cl_float)*dZ,NULL,NULL);
    command_queue.finish();
    
    
    
    //itinialize weight
    command_queue.enqueueFillImage(weight_image[mergeLevel], ini, origin, region);
    command_queue.finish();
    command_queue.enqueueNDRangeKernel(kernel_imgReg2X, NULL, global_item_size_RX, local_item_size_RX, NULL, NULL);
    command_queue.finish();
    command_queue.enqueueNDRangeKernel(kernel_imgReg2Y, NULL, global_item_size_RY, local_item_size_RY, NULL, NULL);
    command_queue.finish();
	/*float* dF2i;
	float* dFi;
	dFi = new float[dZ];
	dF2i = new float[dZ];
	command_queue.enqueueReadBuffer(dF2new_buffer, CL_TRUE, 0, sizeof(cl_float)*dZ, dF2i);
	command_queue.enqueueReadBuffer(dF_buffer, CL_TRUE, 0, sizeof(cl_float)*dZ, dFi);
	for (int i = 0; i < dZ; i++) {
		cout << sqrt(max(0.0f,dF2i[i] - dFi[i] * dFi[i]))*CI << endl;
	}*/
    
    
    //Image registration
    for (int j=mergeLevel; j>=0; j--) {
        unsigned int mergeN = 1<<j;
        //cout<<mergeN<<endl;
        int localsize1 = min((int)maxWorkGroupSize,imageSizeX/(int)mergeN/2);
        int localsize2 = min((int)maxWorkGroupSize,imageSizeY/(int)mergeN/2);
        const cl::NDRange global_item_size_reg1(localsize1,imageSizeY/mergeN,dZ);
        const cl::NDRange local_item_size_reg1(localsize1,1,1);
        const cl::NDRange global_item_size_reg2(localsize2,dZ,1);
        const cl::NDRange local_item_size_reg2(localsize2,1,1);
        int localsize = min((unsigned int)maxWorkGroupSize,imageSizeX/mergeN);
        const cl::NDRange global_item_size_merge(imageSizeX/mergeN,imageSizeY/mergeN,dZ);
        const cl::NDRange local_item_size_merge(localsize,1,1);
        kernel_imgReg1X.setArg(0, mt_target_image[j]);
        kernel_imgReg1X.setArg(1, mt_sample_image[j]);
        kernel_imgReg1X.setArg(2, weight_image[j]);
        kernel_imgReg1X.setArg(9, dF2X[j]);
        kernel_imgReg1X.setArg(10, tJdFX[j]);
        kernel_imgReg1X.setArg(11, tJJX[j]);
        kernel_imgReg1X.setArg(12, devX[j]);
        kernel_imgReg1X.setArg(13, (cl_int)mergeN);
        kernel_imgReg1X.setArg(14, cl::Local(sizeof(cl_float)*localsize1*2));//locmem
        kernel_imgReg1X.setArg(15, (cl_int)(8/mergeN));
        kernel_imgReg1Y.setArg(0, dF2X[j]);
        kernel_imgReg1Y.setArg(1, tJdFX[j]);
        kernel_imgReg1Y.setArg(2, tJJX[j]);
        kernel_imgReg1Y.setArg(3, devX[j]);
        kernel_imgReg1Y.setArg(8, (cl_int)mergeN);
        kernel_imgReg1Y.setArg(9, cl::Local(sizeof(cl_float)*localsize2*2));
        kernel_imgReg2X.setArg(0, mt_target_image[j]);
        kernel_imgReg2X.setArg(1, mt_sample_image[j]);
        kernel_imgReg2X.setArg(2, weight_image[j]);
        kernel_imgReg2X.setArg(5, dF2X[j]);
        kernel_imgReg2X.setArg(6, dFX[j]);
        kernel_imgReg2X.setArg(7, devX[j]);
        kernel_imgReg2X.setArg(8, (cl_int)mergeN);
        kernel_imgReg2X.setArg(9, cl::Local(sizeof(cl_float)*localsize1*2));//locmem
        kernel_imgReg2Y.setArg(0, dF2X[j]);
        kernel_imgReg2Y.setArg(1, dFX[j]);
        kernel_imgReg2Y.setArg(2, devX[j]);
        kernel_imgReg2Y.setArg(5, (cl_int)mergeN);
        kernel_imgReg2Y.setArg(6, cl::Local(sizeof(cl_float)*localsize2*2));
        for (int trial=0; trial < num_trial; trial++) {
            //imageReg1:estimate dF2(old), tJJ, tJdF
            command_queue.enqueueNDRangeKernel(kernel_imgReg1X, NULL, global_item_size_reg1, local_item_size_reg1, NULL, NULL);
            command_queue.finish();
            command_queue.enqueueNDRangeKernel(kernel_imgReg1Y, NULL, global_item_size_reg2, local_item_size_reg2, NULL, NULL);
            command_queue.finish();
            
            
            //LM
            command_queue.enqueueNDRangeKernel(kernel_LM, NULL, global_item_size2, local_item_size2, NULL, NULL);
            command_queue.finish();
            
            
            //dL estimation
            command_queue.enqueueNDRangeKernel(kernel_dL, NULL, global_item_size2, local_item_size2, NULL, NULL);
            command_queue.finish();
            
            //estimate parameter cnd
            command_queue.enqueueCopyBuffer(p_buffer, p_cnd_buffer, 0, 0, sizeof(cl_float)*(p_num+2)*dZ);
            command_queue.enqueueNDRangeKernel(kernel_cnd, NULL, global_item_size3, local_item_size3, NULL, NULL);
            command_queue.finish();
            
            
            //imageReg2:estimate dF2(new)
            command_queue.enqueueNDRangeKernel(kernel_imgReg2X, NULL, global_item_size_reg1, local_item_size_reg1, NULL, NULL);
            command_queue.finish();
            command_queue.enqueueNDRangeKernel(kernel_imgReg2Y, NULL, global_item_size_reg2, local_item_size_reg2, NULL, NULL);
            command_queue.finish();
			/*float* dF2;
			float* dF;
			dF = new float[dZ];
			dF2 = new float[dZ];
			command_queue.enqueueReadBuffer(dF2new_buffer, CL_TRUE, 0, sizeof(cl_float)*dZ, dF2);
			command_queue.enqueueReadBuffer(dF_buffer, CL_TRUE, 0, sizeof(cl_float)*dZ, dF);
			for (int i = 0; i < dZ; i++) {
				cout << sqrt(max(0.0f, dF2i[i] - dFi[i] * dFi[i]))*CI <<endl;
			}*/

            
            //evaluate cnd
            command_queue.enqueueNDRangeKernel(kernel_eval, NULL, global_item_size2, local_item_size2, NULL, NULL);
            command_queue.finish();
            
            //update to new para or hold to old para
            command_queue.enqueueNDRangeKernel(kernel_UorH, NULL, global_item_size3, local_item_size3, NULL, NULL);
            command_queue.finish();
        }
    }
    
    //error create
    command_queue.enqueueNDRangeKernel(kernel_error, NULL, global_item_size3, local_item_size3, NULL, NULL);
    command_queue.finish();
    
    
    //output image reg results to buffer
    command_queue.enqueueNDRangeKernel(kernel_output, NULL, global_item_size5, local_item_size5, NULL, NULL);
    command_queue.finish();
    
    
    return 0;
}
