/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^mtFit_kernel)(const cl_ndrange *ndrange, cl_float* mt_fit_img, cl_float* fp_img, cl_float energy, cl_int fpOffset, cl_int funcMode, cl_image refSpectrum);
extern void (^Jacobian_kernel)(const cl_ndrange *ndrange, cl_float* J_img, cl_float* fp_img, cl_float energy, cl_int fpOffset, cl_int funcMode, cl_image refSpectrum);
extern void (^chi2_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* mt_fit_img, cl_float* chi2_img, cl_int en);
extern void (^tJdF_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* mt_fit_img, cl_float* J_img, cl_float* tJdF_img, cl_int en);
extern void (^tJJ_kernel)(const cl_ndrange *ndrange, cl_float* J_img, cl_float* tJJ_img);
extern void (^LevenbergMarquardt_kernel)(const cl_ndrange *ndrange, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* fp_img, cl_float* fp_cnd_img, cl_float* lambda_img, cl_char* p_fix);
extern void (^constrain_kernel)(const cl_ndrange *ndrange, cl_float* fp_cnd_img, cl_float* C_mat, cl_float* D_vec);
extern void (^UpdateFp_kernel)(const cl_ndrange *ndrange, cl_float* fp_img, cl_float* fp_cnd_img, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* lambda_img, cl_float* nyu_img, cl_float* chi2_old_img, cl_float* chi2_new_img);
extern void (^setMask_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_char* mask_img);
extern void (^applyThreshold_kernel)(const cl_ndrange *ndrange, cl_float* fit_results_img, cl_char* mask);
extern void (^partialDerivativeOfGradiant_kernel)(const cl_ndrange *ndrange, cl_float* original_img_src, cl_float* img_src, cl_float* img_dest, cl_float* epsilon_g, cl_float* alpha_g);
extern void (^redimension_refSpecta_kernel)(const cl_ndrange *ndrange, cl_image refSpectra, cl_image refSpectra_raw, cl_float* energy, cl_int numE);
