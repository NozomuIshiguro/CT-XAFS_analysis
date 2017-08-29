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
    int diffstep;
    bool displayCnt;
    
public:
    regMode(int regmodeNumber,int cntFitmode);
    int get_regModeNo();
    string ofs_transpara();
    string get_regModeName();
    string get_oss_target();
    string get_oss_target(float *p_vec);
    
    string oss_sample(float *transpara,float *transpara_error,
                      int *p_precision,int *p_err_precision);
    int get_p_num();
    bool getCntBool();
	void set_pfix(input_parameter inp);
    cl::Program buildImageRegProgram(cl::Context context, int imageSizeX, int imageSizeY);
    
    float *p_ini;
    char *p_fix;
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
                      int num_trial, float lambda);

#endif
