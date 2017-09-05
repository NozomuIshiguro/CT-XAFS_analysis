/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^LevenbergMarquardt_kernel)(const cl_ndrange *ndrange, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* dp_img, cl_float* lambda_img, cl_char* p_fix, cl_float* inv_tJJ_img);
extern void (^estimate_dL_kernel)(const cl_ndrange *ndrange, cl_float* dp_img, cl_float* tJJ_img, cl_float* tJdF_img, cl_float* lambda_img, cl_float* dL_img);
extern void (^updatePara_kernel)(const cl_ndrange *ndrange, cl_float* dp, cl_float* fp, cl_int z_id1, cl_int z_id2);
extern void (^evaluateUpdateCandidate_kernel)(const cl_ndrange *ndrange, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* lambda_img, cl_float* nyu_img, cl_float* dF2_old_img, cl_float* dF2_new_img, cl_float* dL_img, cl_float* rho_img);
extern void (^updateOrRestore_kernel)(const cl_ndrange *ndrange, cl_float* para_img, cl_float* para_img_backup, cl_float* rho_img);
extern void (^updateOrHold_kernel)(const cl_ndrange *ndrange, cl_float* para_img, cl_float* para_img_cnd, cl_float* rho_img);
