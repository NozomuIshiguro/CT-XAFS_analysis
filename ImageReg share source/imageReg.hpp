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
    string regModeName;
    string oss_target;
    int p_num;
    
public:
    regMode(int regmodeNumber);
    int get_regModeNo();
    int get_contrastModeNo();
    string ofs_transpara();
    string get_regModeName();
    string get_oss_target();
    string get_oss_target(float *p_vec);
    
    string oss_sample(float *transpara,float *transpara_error,
                      int *p_precision,int *p_err_precision);
    int get_p_num();

	void set_pfix(input_parameter inp);
    cl::Program buildImageRegProgram(cl::Context context, int imageSizeX, int imageSizeY);
    
    float *p_ini;
    float *p_fix;
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
