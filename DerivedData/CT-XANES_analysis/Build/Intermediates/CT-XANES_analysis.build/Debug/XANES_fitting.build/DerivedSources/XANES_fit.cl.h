/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^chi2Stack_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* fp_img, cl_image refSpectra, cl_float* chi2_img, cl_float* energy, cl_int* funcModeList, cl_int startEnum, cl_int endEnum);
extern void (^chi2_tJdF_tJJ_Stack_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* fp_img, cl_image refSpectra, cl_float* chi2_img, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* energy, cl_int* funcModeList, cl_char* p_fix, cl_int startEnum, cl_int endEnum);
extern void (^constrain_kernel)(const cl_ndrange *ndrange, cl_float* fp_cnd_img, cl_float* C_mat, cl_float* D_vec);
extern void (^setMask_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_char* mask_img);
extern void (^applyMask_kernel)(const cl_ndrange *ndrange, cl_float* fp_img, cl_char* mask);
extern void (^partialDerivativeOfGradiant_kernel)(const cl_ndrange *ndrange, cl_float* original_img_src, cl_float* img_src, cl_float* img_dest, cl_float* epsilon_g, cl_float* alpha_g);
extern void (^redimension_refSpecta_kernel)(const cl_ndrange *ndrange, cl_image refSpectra, cl_image refSpectrum_raw, cl_float* energy, cl_int numE, cl_int offset);
extern void (^SoftThresholdingFunc_kernel)(const cl_ndrange *ndrange, cl_float* fp_img, cl_float* dp_img, cl_float* dp_cnd_img, cl_float* inv_tJJ_img, cl_char* p_fix, cl_float* lambda_fista);
extern void (^FISTAupdate_kernel)(const cl_ndrange *ndrange, cl_float* fp_new_img, cl_float* fp_old_img, cl_float* beta_img, cl_float* w_img);
extern void (^powerIteration_kernel)(const cl_ndrange *ndrange, cl_float* A_img, cl_float* maxEval_img, cl_int iter);
