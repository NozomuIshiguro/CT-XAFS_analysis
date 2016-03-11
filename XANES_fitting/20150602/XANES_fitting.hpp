//
//  XANES_fitting.hpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/02/10.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#ifndef CT_XANES_analysis_XANES_fitting_hpp
#define CT_XANES_analysis_XANES_fitting_hpp

#include "OpenCL_analysis.hpp"

#define IMAGE_SIZE_X 2048
#define IMAGE_SIZE_Y 2048
#define IMAGE_SIZE_M 4194304 //2048*2048
#define PARAM_SIZE 8




class fitting_eq{
    float* fitting_para;
    char* free_para;
    float* para_upperlimit;
    float* para_lowerlimit;
    size_t param_size;
    size_t free_param_size;
    string kernel_preprocessor_str;
    vector<string> parameter_name;
    float* freefitting_para;
public:
    fitting_eq(input_parameter inp, string OCL_preprocessor);
    string preprocessor_str();
    float* fit_para();
    char* freefix_para();
    size_t freeParaSize();
    size_t ParaSize();
    string param_name(int i);
    float* freefit_para();
    float* freepara_upperlimit;
    float* freepara_lowerlimit;
};


int OCL_device_list(vector<OCL_platform_device> *plat_dev_list);
int XANES_fit_ocl(fitting_eq fiteq, input_parameter inp,
                  OCL_platform_device plat_dev_list,string fileName_base);
int readRawFile(string filepath_input,float *binImgf);
int outputRawFile_stream(string filename,float*data);
#endif
