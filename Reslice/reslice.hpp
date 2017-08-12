//
//  reslice.hpp
//  Reslice
//
//  Created by Nozomu Ishiguro on 2015/06/22.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#ifndef __CT_XANES_analysis__reslice_ocl__
#define __CT_XANES_analysis__reslice_ocl__

#include "OpenCL_analysis.hpp"

int reslice_ocl(input_parameter inp,OCL_platform_device plat_dev_list,string fileName_base);


int readRawFile(string filepath_input,float *binImgf,int imageSizeM);
int outputRawFile_stream(string filename,float *data,size_t pnt_size);
int reslice_programBuild(cl::Context context,vector<cl::Kernel> *kernels,
                         int startAngleNo, int endAngleNo, input_parameter inp);
int reslice(cl::CommandQueue command_queue, cl::Kernel kernel,
            int num_angle,int iter,float baseup,
            vector<float*> mt_vec, vector<float*> prj_vec,
            cl::Buffer xshift_buff,cl::Buffer zshift_buff,
            int imageSizeX, int imageSizeY);
int reslice_mtImg(OCL_platform_device plat_dev_list,cl::Kernel kernel,
                  vector<float*> mt_img_vec, vector<float*> prj_img_vec,
                  input_parameter inp);


#endif /* defined(__CT_XANES_analysis__reslice_ocl__) */
