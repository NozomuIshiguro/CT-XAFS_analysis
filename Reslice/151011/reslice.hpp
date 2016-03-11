//
//  reslice.hpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/06/22.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#ifndef __CT_XANES_analysis__reslice_ocl__
#define __CT_XANES_analysis__reslice_ocl__

#include "OpenCL_analysis.hpp"

int reslice_ocl(input_parameter inp,OCL_platform_device plat_dev_list,string fileName_base);
int reslice_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                   int startAngleNo, int EndAngleNo,int EnergyNo,float baseup,
                   string input_dir, string output_dir,
                   string fileName_base_i,string fileName_base_o,
                   int startX, int endX, int startZ, int endZ,
                   bool Zcorr, bool Xcorr, bool last);

int readRawFile(string filepath_input,float *binImgf);
int outputRawFile_stream(string filename,float *data,size_t pnt_size);
int xyshift_output_thread(int num_angle,string output_dir,float *xshift, float *zshift);
int prj_output_thread(int startZ,int EndZ,int EnergyNo,int num_angle,
                          string output_dir,string fileName_base,
                          vector<float*> prj_vec);

#endif /* defined(__CT_XANES_analysis__reslice_ocl__) */
