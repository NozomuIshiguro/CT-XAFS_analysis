//
//  XANES_fitting.hpp
//  XANES_fitting
//
//  Created by Nozomu Ishiguro on 2015/02/10.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#ifndef CT_XANES_analysis_XANES_fitting_hpp
#define CT_XANES_analysis_XANES_fitting_hpp

#include "OpenCL_analysis.hpp"

#define IMAGE_SIZE_E 1024
#define IMAGE_SIZE_X 2048
#define IMAGE_SIZE_Y 2048
#define IMAGE_SIZE_M 4194304 //2048*2048
#define PARAM_SIZE 8

int OCL_device_list(vector<OCL_platform_device> *plat_dev_list);
int XANES_fit_ocl(fitting_eq fiteq, input_parameter inp,
                  OCL_platform_device plat_dev_list);
int XANES_fit_thread(cl::CommandQueue command_queue, cl::Program program,
                     fitting_eq fiteq,int AngleNo, int thread_id,input_parameter inp,
                     cl::Buffer energy, cl::Buffer C_matrix_buff, cl::Buffer D_vector_buff,
                     cl::Buffer freeFix_buff, cl::Image1DArray refSpectra, cl::Buffer funcMode_buff,
                     vector<float*> mt_vec, int64_t offset, int processImageSizeY);
int readRawFile(string filepath_input,float *binImgf,int imageSizeM);
int outputRawFile_stream(string filename,float*data,size_t pnt_size);
string kernel_preprocessor_nums(fitting_eq fiteq,input_parameter inp);

#endif
