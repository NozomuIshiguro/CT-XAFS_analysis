//
//  opencl_deveice_info.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/03/06.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"

int Deviceinfo(OCL_platform_device plat_dev_list,bool supportImg)
{
    cl_int ret;
    
    int t=0;
    for (int i=0; i<plat_dev_list.platsize(); i++) {
        for (int j=0; j<plat_dev_list.devsize(i); j++) {
	        try{
				cout << "Device No. " << t + 1 << endl;
				string platform_param;
				plat_dev_list.plat(i).getInfo(CL_PLATFORM_NAME, &platform_param);
				cout << "CL PLATFORM NAME: " << platform_param << endl;
				plat_dev_list.plat(i).getInfo(CL_PLATFORM_VERSION, &platform_param);
				cout << "   " << platform_param << endl;
				string device_pram;
				size_t device_pram_size[3] = { 0,0,0 };
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_NAME, &device_pram);
				cout << "CL DEVICE NAME: " << device_pram << endl;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &device_pram_size);
				cout << "CL DEVICE MAX COMPUTE UNITS: " << device_pram_size[0] << endl;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &device_pram_size);
				cout << "CL DEVICE MAX WORK GROUP SIZE: " << device_pram_size[0] << endl;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &device_pram_size);
				cout << "CL DEVICE MAX WORK ITEM SIZE: " << device_pram_size[0] << ", " << device_pram_size[1] << ", " << device_pram_size[2] << endl;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_MAX_PARAMETER_SIZE, &device_pram_size);
				cout << "CL DEVICE MAX CLOCK FREQUENCY: " << device_pram_size[0] << " MHz" << endl;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &device_pram_size);
				cout << "CL DEVICE MAX PARAMETER SIZE: " << device_pram_size[0] << endl;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &device_pram_size);
				cout << "CL DEVICE CL DEVICE_GLOBAL MEM SIZE: " << device_pram_size[0] << " bytes" << endl;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &device_pram_size);
				cout << "CL DEVICE CL DEVICE LOCAL MEM SIZE: " << device_pram_size[0] << " bytes" << endl;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &device_pram_size);
				cout << "CL DEVICE CL DEVICE MAX MEM ALLOC SIZE: " << device_pram_size[0] << " bytes" << endl;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, &device_pram_size);
				cout << "CL DEVICE CONSTANT BUFFER SIZE: " << device_pram_size[0] << " bytes" << endl;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, &device_pram_size);
				cout << "CL DEVICE IMAGE MAX BUFFER SIZE: " << device_pram_size[0] << endl;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_IMAGE2D_MAX_HEIGHT, &device_pram_size);
				cout << "CL DEVICE MAX 2D IMAGE HEIGHT: " << device_pram_size[0] << endl;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_IMAGE2D_MAX_WIDTH, &device_pram_size);
				cout << "CL DEVICE MAX 2D IMAGE WIDTH: " << device_pram_size[0] << endl;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_IMAGE3D_MAX_HEIGHT, &device_pram_size);
				cout << "CL DEVICE MAX 3D IMAGE HEIGHT: " << device_pram_size[0] << endl;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_IMAGE3D_MAX_WIDTH, &device_pram_size);
				cout << "CL DEVICE MAX 3D IMAGE WIDTH: " << device_pram_size[0] << endl;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_IMAGE3D_MAX_DEPTH, &device_pram_size);
				cout << "CL DEVICE MAX 3D IMAGE DEPTH: " << device_pram_size[0] << endl;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, &device_pram_size);
				cout << "CL DEVICE MAX IMAGE ARRAY SIZE: " << device_pram_size[0] << endl;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_MAX_READ_IMAGE_ARGS, &device_pram_size);
				cout << "CL DEVICE MAX READ IMAGE ARGS: " << device_pram_size[0] << endl;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_MAX_WRITE_IMAGE_ARGS, &device_pram_size);
				cout << "CL DEVICE MAX WRITE IMAGE ARGS: " << device_pram_size[0] << endl;


				/*Open CL command que create*/
				cl_command_queue_properties device_queue_info;
				ret = plat_dev_list.dev(i, j).getInfo(CL_DEVICE_QUEUE_PROPERTIES, &device_queue_info);
				if ((device_queue_info & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0x0) {
					cout << "OUT OF ORDER EXEC MODE ENABLE" << endl << endl;
				}
				else {
					cout << "OUT OF ORDER EXEC MODE NOT ENABLE" << endl << endl;
				}

				//support image
				if (supportImg) {
					vector<cl::ImageFormat> suppport_fmt;
					plat_dev_list.queue(i, j).getInfo<CL_QUEUE_CONTEXT>().getSupportedImageFormats(CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, &suppport_fmt);
					cout << "SUPPORTED CL_MEM_OBJECT_IMAGE2D FORMAT" << endl;
					cout << "   ";
					for (int k = 0; k<suppport_fmt.size(); k++) {
						switch (suppport_fmt[k].image_channel_order) {
						case CL_R:
							cout << " order:CL_R";
							break;
						case CL_A:
							cout << " order:CL_A";
							break;
						case CL_RG:
							cout << " order:CL_RG";
							break;
						case CL_RA:
							cout << " order:CL_RA";
							break;
						case CL_RGB:
							cout << " order:CL_RGB";
							break;
						case CL_RGBA:
							cout << " order:CL_RGBA";
							break;
						case CL_BGRA:
							cout << " order:CL_BGRA";
							break;
						case CL_ARGB:
							cout << " order:CL_ARGB";
							break;
						case CL_INTENSITY:
							cout << " order:CL_INTENSITY";
							break;
						case CL_LUMINANCE:
							cout << " order:CL_LUMINANCE";
							break;
						case CL_Rx:
							cout << " order:CL_Rx";
							break;
						case CL_RGx:
							cout << " order:CL_RGx";
							break;
						case CL_RGBx:
							cout << " order:CL_RGBx";
							break;
						case CL_DEPTH:
							cout << " order:CL_DEPTH";
							break;
						case CL_DEPTH_STENCIL:
							cout << " order:CL_DEPTH_STENCIL";
							break;

						default:
							cout << " order:UNKOWN(" << suppport_fmt[k].image_channel_order << ")";
							break;
						}
						cout << ", ";
						switch (suppport_fmt[k].image_channel_data_type) {
						case CL_SNORM_INT8:
							cout << "type:CL_SNORM_INT8";
							break;
						case CL_SNORM_INT16:
							cout << "type:CL_SNORM_INT16";
							break;
						case CL_UNORM_INT8:
							cout << "type:CL_UNORM_INT8";
							break;
						case CL_UNORM_INT16:
							cout << "type:CL_SNORM_INT8";
							break;
						case CL_UNORM_SHORT_565:
							cout << "type:CL_UNORM_SHORT_565";
							break;
						case CL_UNORM_SHORT_555:
							cout << "type:CL_UNORM_SHORT_555";
							break;
						case CL_UNORM_INT_101010:
							cout << "type:CL_UNORM_INT_101010 ";
							break;
						case CL_SIGNED_INT8:
							cout << "type:CL_SIGNED_INT8";
							break;
						case CL_SIGNED_INT16:
							cout << "type:CL_SIGNED_INT16";
							break;
						case CL_SIGNED_INT32:
							cout << "type:CL_SIGNED_INT32";
							break;
						case CL_UNSIGNED_INT8:
							cout << "type:CL_UNSIGNED_INT8";
							break;
						case CL_UNSIGNED_INT16:
							cout << "type:CL_UNSIGNED_INT16";
							break;
						case CL_UNSIGNED_INT32:
							cout << "type:CL_UNSIGNED_INT32";
							break;
						case CL_HALF_FLOAT:
							cout << "type:CL_HALF_FLOAT";
							break;
						case CL_FLOAT:
							cout << "type:CL_FLOAT";
							break;
						case CL_UNORM_INT24:
							cout << "type:CL_UNORM_INT24";
							break;

						default:
							break;
						}
						cout << endl;
					}
					cout << endl;
				}
			}catch (cl::Error ret) {
				cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
			}
            t++;
        }
    }
    return 0;
}