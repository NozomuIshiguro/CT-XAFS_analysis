/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^estimateBkgImg_kernel)(const cl_ndrange *ndrange, cl_image bkg_img, cl_image mt_data_img, cl_image grid_img, cl_image sample_img, cl_float* para, cl_float* contrast, cl_float* weight);
extern void (^estimateResidueImg_kernel)(const cl_ndrange *ndrange, cl_image residue_img, cl_image mt_data_img, cl_image bkg_img);
extern void (^estimateGridImg_kernel)(const cl_ndrange *ndrange, cl_image grid_img, cl_image residue_reg_img, cl_image sample_img, cl_float* contrast, cl_float* weight);
extern void (^estimateSampleImg_kernel)(const cl_ndrange *ndrange, cl_image sample_img, cl_image residue_reg_img, cl_image grid_img);
extern void (^estimateSampleContrast1_kernel)(const cl_ndrange *ndrange, cl_image residue_reg_img, cl_image grid_img, cl_image sample_img, cl_float* sum, size_t loc_mem);
extern void (^estimateSampleContrast2_kernel)(const cl_ndrange *ndrange, cl_float* sum, cl_float* contrast, size_t loc_mem);
