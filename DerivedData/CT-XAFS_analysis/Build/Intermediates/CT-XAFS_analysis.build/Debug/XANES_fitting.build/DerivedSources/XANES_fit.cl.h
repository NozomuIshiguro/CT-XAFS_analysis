/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^chi2Stack_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* fp_img, cl_image refSpectra, cl_float* chi2_img, cl_float* energy, cl_int* funcModeList, cl_int startEnum, cl_int endEnum, cl_float* weight_img, cl_float* weight_thd_img, cl_float CI);
extern void (^chi2_tJdF_tJJ_Stack_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* fp_img, cl_image refSpectra, cl_float* chi2_img, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* energy, cl_int* funcModeList, cl_char* p_fix, cl_int startEnum, cl_int endEnum, cl_float* weight_img, cl_float* weight_thd_img);
extern void (^setMask_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_char* mask_img);
extern void (^applyMask_kernel)(const cl_ndrange *ndrange, cl_float* fp_img, cl_char* mask);
extern void (^redimension_refSpecta_kernel)(const cl_ndrange *ndrange, cl_image refSpectra, cl_image refSpectrum_raw, cl_float* energy, cl_int numE, cl_int offset);
