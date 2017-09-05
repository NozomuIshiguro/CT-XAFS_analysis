/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^LevenbergMarquardt_kernel)(const cl_ndrange *ndrange, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* dp_img, cl_float* lambda_img, cl_char* p_fix);
extern void (^estimate_dL_kernel)(const cl_ndrange *ndrange, cl_float* dp_img, cl_float* tJJ_img, cl_float* tJdF_img, cl_float* lambda_img, cl_float* dL_img);
extern void (^updatePara_kernel)(const cl_ndrange *ndrange, cl_float* dp, cl_float* fp, cl_int z_id1, cl_int z_id2);
extern void (^evaluateUpdateCandidate_kernel)(const cl_ndrange *ndrange, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* lambda_img, cl_float* nyu_img, cl_float* dF2_old_img, cl_float* dF2_new_img, cl_float* dL_img, cl_float* rho_img);
extern void (^updateOrRestore_kernel)(const cl_ndrange *ndrange, cl_float* para_img, cl_float* para_img_backup, cl_float* rho_img, cl_int z_id1, cl_int z_id2);
extern void (^updateOrHold_kernel)(const cl_ndrange *ndrange, cl_float* para_img, cl_float* para_img_cnd, cl_float* rho_img);
extern void (^FISTA_kernel)(const cl_ndrange *ndrange, cl_float* fp_img, cl_float* dp_img, cl_float* tJJ_img, cl_float* tJdF_img, cl_char* p_fix, cl_float* lambda_LM, cl_float* lambda_fista);
extern void (^contrain_0_kernel)(const cl_ndrange *ndrange, cl_float* C_mat, cl_float* C2_vec);
extern void (^contrain_1_kernel)(const cl_ndrange *ndrange, cl_float* fp_cnd_img, cl_float* eval_img, cl_float* C_mat, cl_int c_num, cl_int p_num);
extern void (^contrain_2_kernel)(const cl_ndrange *ndrange, cl_float* fp_cnd_img, cl_float* weight_img, cl_float* eval_img, cl_float* C_mat, cl_float* D_vec, cl_float* C2_vec, cl_int c_num, cl_int p_num, cl_char weight_b);
