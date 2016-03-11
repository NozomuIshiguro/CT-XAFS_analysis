//
//  imageReg.hpp
//  Image registration share
//
//  Created by Nozomu Ishiguro on 2015/06/04.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#ifndef CT_XANES_analysis_imageReg_hpp
#define CT_XANES_analysis_imageReg_hpp
#include "OpenCL_analysis.hpp"

class regMode{
    int regModeNo;
    int cntModeNo;
    string regModeName;
    string oss_target;
    int p_num;
    int cp_num;
    
public:
    regMode(int regmodeNumber,int cntmodeNumber);
    int get_regModeNo();
    int get_contrastModeNo();
    string ofs_transpara();
    string get_regModeName();
    string get_oss_target();
    string oss_sample(float *transpara,float *transpara_error,
                      int *p_precision,int *p_err_precision);
    int get_p_num();
    int get_cp_num();
    cl::Program buildImageRegProgram(cl::Context context);
    
    float *p_ini;
};

class mask{
public:
    mask(input_parameter inp);
    int refMask_shape;
    int refMask_x;
    int refMask_y;
    int refMask_width;
    int refMask_height;
    float refMask_angle;
    int sampleMask_shape;
    int sampleMask_x;
    int sampleMask_y;
    int sampleMask_width;
    int sampleMask_height;
    float sampleMask_angle;
};


#endif
