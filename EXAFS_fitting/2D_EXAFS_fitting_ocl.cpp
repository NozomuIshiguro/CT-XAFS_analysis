//
//  2D_EXAFS_fitting_ocl.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2016/01/04.
//  Copyright © 2016年 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"

int EXAFS_Rfit(cl::CommandQueue command_queue, vector<cl::Kernel> kernels,
               size_t xsize, size_t ysize){
    try {
        
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        string devicename = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>();
        size_t maxWorkGroupSize = min(xsize,command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
        const cl::NDRange local_item_size(maxWorkGroupSize,1,1);
        cl::NDRange global_item_size(xsize,ysize,1);
        
        int num_shell=1;
        
        //chi data buffers
        cl::Buffer chi_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float2)*1024*xsize*ysize, 0, NULL);
        cl::Buffer FTchi_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float2)*1024*xsize*ysize, 0, NULL);
        //chi fit buffers
        cl::Buffer chiFit_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float2)*1024*xsize*ysize, 0, NULL);
        cl::Buffer FTchiFit_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float2)*1024*xsize*ysize, 0, NULL);
        //Jacobian fit buffers
        cl::Buffer chiJacob_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float2)*1024*xsize*ysize*(num_shell*4+1), 0, NULL);
        cl::Buffer FTchiJacob_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float2)*1024*xsize*ysize*(num_shell*4+1), 0, NULL);
        //spinfactor buffer
        cl::Buffer w_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float2)*1024, 0, NULL);
        //FEFF calculation buffers
        vector<float> Reff_vecs;
        vector<cl::Buffer> mag_buffers;
        vector<cl::Buffer> phase_buffers;
        vector<cl::Buffer> redFactor_buffers;
        vector<cl::Buffer> lambdaFEFF_buffers;
        vector<cl::Buffer> real_2phc_buffers;
        vector<cl::Buffer> real_p_buffers;
        for (int i=0; i<num_shell; i++) {
            mag_buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*1024, 0, NULL));
            phase_buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*1024, 0, NULL));
            redFactor_buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*1024, 0, NULL));
            lambdaFEFF_buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*1024, 0, NULL));
            real_2phc_buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*1024, 0, NULL));
            real_p_buffers.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*1024, 0, NULL));
        }
        //fit parameter buffers
        cl::Buffer fit_para_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*(num_shell*4+1)*xsize*ysize, 0, NULL);
        
        
        //set kernel (spinFact)
        kernels[0].setArg(0, w_buffer);
        kernels[0].setArg(1, 1024);
        //spinFact
        const cl::NDRange global_item_size_SF(1024,1,1);
        command_queue.enqueueNDRangeKernel(kernels[0], NULL, global_item_size_SF, local_item_size, NULL, NULL);
        command_queue.finish();
        
        
        //set kernel (kWeightIMGarray)
        int kweight = 3;
        kernels[4].setArg(0, chi_buffer);
        kernels[4].setArg(1, kweight);
        kernels[4].setArg(2, 0.05f);
        //kWeightIMGarray
        const cl::NDRange global_item_size_EXAFS(xsize,ysize,401);//zsize=401: k=20まで計算
        command_queue.enqueueNDRangeKernel(kernels[4], NULL, global_item_size_EXAFS, local_item_size, NULL, NULL);
        command_queue.finish();
        
        
        //set kernel (hanningWindowFuncIMGarray)
        float kmin = 3.0f;
        float kmax = 15.0f;
        kernels[5].setArg(0, chi_buffer);
        kernels[5].setArg(1, kmin);
        kernels[5].setArg(2, kmax);
        kernels[5].setArg(3, 1.0f);
        kernels[5].setArg(4, 0.05f);
        //bitReverseIMGarray
        command_queue.enqueueNDRangeKernel(kernels[5], NULL, global_item_size_EXAFS, local_item_size, NULL, NULL);
        command_queue.finish();
        
        
        //set kernel (bitReverseIMGarray)
        kernels[1].setArg(0, chi_buffer);
        kernels[1].setArg(1, FTchi_buffer);
        kernels[1].setArg(2, 10);
        //bitReverseIMGarray
        command_queue.enqueueNDRangeKernel(kernels[1], NULL, global_item_size, local_item_size, NULL, NULL);
        command_queue.finish();
        
        
        //set kernel (butterflyIMGarray)
        kernels[2].setArg(0, FTchi_buffer);
        kernels[2].setArg(1, w_buffer);
        kernels[2].setArg(2, 1024);
        //butterflyIMGarray
        for (int iter=1; iter<10; iter++) {
            kernels[2].setArg(3, iter);
            command_queue.enqueueNDRangeKernel(kernels[2], NULL, global_item_size, local_item_size, NULL, NULL);
            command_queue.finish();
        }
        
        
        //set kernel (FFTnormMGarray)
        kernels[3].setArg(0, FTchi_buffer);
        kernels[3].setArg(1, 0.05f);
        kernels[3].setArg(2, 1024);
        command_queue.enqueueNDRangeKernel(kernels[3], NULL, global_item_size, local_item_size, NULL, NULL);
        command_queue.finish();
        
        
        //set kernel (EXAFSoscillationIMGarray)
        kernels[6].setArg(0, chiFit_buffer);
        kernels[6].setArg(1, fit_para_buffer);
        kernels[6].setArg(2, chiJacob_buffer);
        kernels[6].setArg(11, 0.05f);
        //butterflyIMGarray
        for (int i=0; i<num_shell; i++) {
            kernels[6].setArg(3, mag_buffers[i]);
            kernels[6].setArg(4, phase_buffers[i]);
            kernels[6].setArg(5, redFactor_buffers[i]);
            kernels[6].setArg(6, lambdaFEFF_buffers[i]);
            kernels[6].setArg(7, real_2phc_buffers[i]);
            kernels[6].setArg(8, real_p_buffers[i]);
            kernels[6].setArg(9, Reff_vecs[i]);
            kernels[6].setArg(10, i);
            command_queue.enqueueNDRangeKernel(kernels[6], NULL, global_item_size_EXAFS, local_item_size, NULL, NULL);
            command_queue.finish();
        }
        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
    
    return 0;
}