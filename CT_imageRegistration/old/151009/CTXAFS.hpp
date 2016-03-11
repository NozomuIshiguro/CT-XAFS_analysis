//
//  CTXAFS.hpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/02/06.
//  Copyright (c) 2015 Nozomu Ishiguro. All rights reserved.
//

#ifndef __CT_XANES_analysis__CTXAFS_cpp_h
#define __CT_XANES_analysis__CTXAFS_cpp_h

#include "OpenCL_analysis.hpp"
#include "imageReg.hpp"

int OCL_device_list(vector<OCL_platform_device> *plat_dev_list);
int readHisFile_stream(string filename, int startnum, int endnum, float *binImgf,size_t shift);
int outputRawFile_stream(string filename,float*data,size_t pnt_size);

int imageRegistlation_ocl(string fileName_base, input_parameter inp,
                          OCL_platform_device plat_dev_list, regMode regmode);

#endif /* defined(__CT_XANES_analysis__CT_imageRegistration__) */
