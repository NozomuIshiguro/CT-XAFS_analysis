//
//  imageBatchBinning.hpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2017/07/18.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#ifndef imageBatchBinning_hpp
#define imageBatchBinning_hpp

#include "OpenCL_analysis.hpp"

int OCL_device_list(vector<OCL_platform_device> *plat_dev_list);
int readRawFile(string filepath_input,float *binImgf,int imageSizeM);
int outputRawFile_stream(string filename,float*data,size_t pnt_size);
int imageBatchBinning_OCL(input_parameter inp,OCL_platform_device plat_dev_list);

extern vector<thread> binning_th;

#endif /* imageBatchBinning_hpp */
