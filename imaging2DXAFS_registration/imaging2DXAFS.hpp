//
//  imaging2DXAFS.hpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/05/28.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#ifndef CT_XANES_analysis_imaging2DXAFS_hpp
#define CT_XANES_analysis_imaging2DXAFS_hpp

#include "OpenCL_analysis.hpp"
#include "imageReg.hpp"

int OCL_device_list(vector<OCL_platform_device> *plat_dev_list);
int readRawFile(string filepath_input,float *binImgf, int startnum, int endnum);
int outputRawFile_stream(string filename,float*data,size_t pnt_size);

int imageRegistlation_2D_ocl(input_parameter inp,
                             OCL_platform_device plat_dev_list, regMode regmode);


#endif
