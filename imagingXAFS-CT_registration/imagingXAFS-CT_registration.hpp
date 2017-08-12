//
//  imagingXAFSXAFS-CT_registration.hpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2016/10/10.
//  Copyright © 2016年 Nozomu Ishiguro. All rights reserved.
//

#ifndef imagingXAFSXAFS_CT_registration_h
#define imagingXAFSXAFS_CT_registration_h


#include "OpenCL_analysis.hpp"
#include "imageReg.hpp"

int OCL_device_list(vector<OCL_platform_device> *plat_dev_list);
int readHisFile_stream(string filename, int startnum, int endnum, unsigned short *binImgf);
int outputRawFile_stream(string filename,float*data,size_t pnt_size);

int imXAFSCT_Registration_ocl(string fileName_base, input_parameter inp,
                          OCL_platform_device plat_dev_list, regMode regmode);

int mt_output_thread(int startAngleNo, int EndAngleNo,
                     input_parameter inp,
                     vector<float*> mt_outputs,float* p_pointer,float* p_err_pointer,
                     regMode regmode,int thread_id);

int imXAFSCT_imageReg_thread(cl::CommandQueue command_queue, CL_objects CLO,
                    vector<unsigned short*> It_img_target,vector<unsigned short*> It_img_sample,
                    int startAngleNo,int EndAngleNo,
                    input_parameter inp, regMode regmode, mask msk, int thread_id);


int imXAFSCT_data_input_thread(int thread_id, cl::CommandQueue command_queue,
                               CL_objects CLO,int startAngleNo,int EndAngleNo,
                               string fileName_base, input_parameter inp,
                               regMode regmode, mask msk);

int mergeRawhisBuffers(cl::CommandQueue queue, cl::Kernel kernel,
                       const cl::NDRange global_item_size,const cl::NDRange local_item_size,
                       unsigned short *img, cl::Buffer img_buffer,int mergeN);

int imXAFSCT_mt_conversion(cl::CommandQueue queue,cl::Kernel kernel,
                  cl::Buffer dark_buffer, cl::Buffer I0_buffer,
                  cl::Buffer mt_buffer,cl::Image2DArray mt_image,cl::Image2DArray mt_outputImg,
                  const cl::NDRange global_item_size,const cl::NDRange local_item_size,
                  vector<unsigned short*> It_pointer,int E_num,int dA, mask msk, bool refBool);

#endif

