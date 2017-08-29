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
int readHisFile_stream(string filename, int startnum, int endnum, unsigned short *binImgf,int imageSizeM);
int outputRawFile_stream(string filename,float*data,size_t pnt_size);

int imageRegistlation_ocl(string fileName_base, input_parameter inp,
                          OCL_platform_device plat_dev_list, regMode regmode);


int data_input_thread(int thread_id, cl::CommandQueue command_queue, CL_objects CLO,
                      int startAngleNo,int EndAngleNo,
                      string fileName_base, input_parameter inp,
                      regMode regmode, mask msk);

int mergeRawhisBuffers(cl::CommandQueue queue, cl::Kernel kernel,
                       const cl::NDRange global_item_size,const cl::NDRange local_item_size,
                       unsigned short *img, cl::Buffer img_buffer,int mergeN,int imageSizeM);

int mt_conversion(cl::CommandQueue queue,cl::Kernel kernel,
                  cl::Buffer dark_buffer,cl::Buffer I0_buffer,
                  cl::Buffer mt_buffer,cl::Image2DArray mt_image,cl::Image2DArray mt_outputImg,
                  const cl::NDRange global_item_size,const cl::NDRange local_item_size,const
                  cl::NDRange global_item_offset, unsigned short *It_pointer,
                  int dA, mask msk, bool refBool, int imageSizeM);

#endif
