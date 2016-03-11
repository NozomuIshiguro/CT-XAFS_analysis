//
//  imageReg.hpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/06/04.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#ifndef CT_XANES_analysis_imageReg_hpp
#define CT_XANES_analysis_imageReg_hpp
#include "OpenCL_analysis.hpp"

class regMode{
    int regModeNo;
    string regModeName;
    string oss_target;
    int transpara_num;
    int reductpara_num;

public:
    regMode(int regmodeNumber);
    int get_regModeNo();
    string ofs_transpara();
    string get_regModeName();
    string get_oss_target();
    string oss_sample(float *transpara,float *transpara_error,
                      int *error_precision);
    void changeRegMode(int regmodeNumber);
    int get_transpara_num();
    int get_reductpara_num();
    void reset_transpara(float *transpara,int num_parallel);
    cl::Program buildImageRegProgram(cl::Context context,int definite);
};

class mask{
public:
    mask(input_parameter inp);
    int refMask_shape;
    int refMask_x;
    int refMask_y;
    int refMask_width;
    int refMask_height;
    int sampleMask_shape;
    int sampleMask_x;
    int sampleMask_y;
    int sampleMask_width;
    int sampleMask_height;
};


#endif
