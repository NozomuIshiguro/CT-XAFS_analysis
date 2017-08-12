//
//  ImagingXAFS_RegAndBkgRemoval.hpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2017/05/19.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#ifndef ImagingXAFS_RegAndBkgRemoval_h
#define ImagingXAFS_RegAndBkgRemoval_h

#include "OpenCL_analysis.hpp"
#include "imageReg.hpp"

int OCL_device_list(vector<OCL_platform_device> *plat_dev_list);
int readRawFile(string filepath_input,float *binImgf, int startnum, int endnum, int imgSize);
int outputRawFile_stream(string filename,float*data,size_t pnt_size);

int imXAFS_RegAndBkgRemoval_ocl(string fileName_base, input_parameter inp,
                                OCL_platform_device plat_dev_list,regMode regmode);

int mt_output_thread(int startAngleNo, int EndAngleNo,
                     input_parameter inp,
                     vector<float*> mt_outputs,float* p_pointer,float* p_err_pointer,
                     regMode regmode,int thread_id);

int imXAFSCT_imageReg_thread(cl::CommandQueue command_queue, CL_objects CLO,
                             vector<unsigned short*> It_img_target,vector<unsigned short*> It_img_sample,
                             int startAngleNo,int EndAngleNo,
                             input_parameter inp, regMode regmode, mask msk, int thread_id);


int imXAFSRegBkg_data_input_thread(int AngleN, int thread_id,
                                   cl::CommandQueue command_queue,CL_objects CLO,
                                   string fileName_base,input_parameter inp,
                                   regMode regmode, mask msk);

#endif /* ImagingXAFS_RegAndBkgRemoval_h */
