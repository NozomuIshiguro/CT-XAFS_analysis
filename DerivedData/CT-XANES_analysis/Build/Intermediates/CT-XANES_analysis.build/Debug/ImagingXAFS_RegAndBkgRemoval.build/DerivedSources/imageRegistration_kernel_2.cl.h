/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^mt_conversion_kernel)(const cl_ndrange *ndrange, cl_float* dark, cl_float* I0, cl_ushort* It_buffer, cl_float* mt_buffer, cl_image mt_img1, cl_image mt_img2, cl_int shapeNo, cl_int startpntX, cl_int startpntY, cl_uint width, cl_uint height, cl_float angle, cl_int evaluatemode);
extern void (^mt_transfer_kernel)(const cl_ndrange *ndrange, cl_image mt_img_src, cl_image mt_img_dest, cl_int layerN, cl_int shapeNo, cl_int startpntX, cl_int startpntY, cl_uint width, cl_uint height, cl_float angle);
extern void (^imageReg_Jacobian_kernel)(const cl_ndrange *ndrange, cl_image mt_s_img, cl_image mt_t_img, cl_float* Jacobian, cl_float* p, cl_float* p_target, cl_int mergeN);
extern void (^imageReg_dF_kernel)(const cl_ndrange *ndrange, cl_image mt_t_img, cl_image mt_s_img, cl_float* chi2, cl_float* p, cl_float* p_target, cl_int mergeN);
extern void (^imageReg_dF_EnergyStack_kernel)(const cl_ndrange *ndrange, cl_image mt_t_img, cl_image mt_s_img, cl_float* chi2, cl_float* p, cl_float* p_target, cl_int mergeN);
extern void (^imageReg_tJJ_reduction_kernel)(const cl_ndrange *ndrange, cl_float* Jacobian, cl_float* tJJ, size_t loc_mem, cl_int paraN1, cl_int paraN2, cl_int mergeN);
extern void (^imageReg_tJdF_reduction_kernel)(const cl_ndrange *ndrange, cl_float* Jacobian, cl_float* dF, cl_float* tJdF, size_t loc_mem, cl_int mergeN);
extern void (^imageReg_chi2_reduction_kernel)(const cl_ndrange *ndrange, cl_float* dF, cl_float* chi2, size_t loc_mem, cl_int mergeN);
extern void (^imageReg_LM_kernel)(const cl_ndrange *ndrange, cl_float* tJJ_g, cl_float* tJdF_g, cl_float* p_g, cl_float* p_cnd, cl_float* p_err, cl_float* p_fix, cl_float* lambda_g);
extern void (^imageReg_Update_kernel)(const cl_ndrange *ndrange, cl_float* tJJ, cl_float* tJdF, cl_float* p_g, cl_float* p_cnd_g, cl_float* p_fix, cl_float* chi2_old, cl_float* chi2_new, cl_float* lambda_g, cl_float* nyu_g);
extern void (^merge_mt_kernel)(const cl_ndrange *ndrange, cl_image mt_sample, cl_float* mt_output);
extern void (^merge_rawhisdata_kernel)(const cl_ndrange *ndrange, cl_ushort* rawhisdata, cl_float* outputdata, cl_int mergeN);
extern void (^imQXAFS_smoothing_kernel)(const cl_ndrange *ndrange, cl_float* rawmtdata, cl_float* outputdata, cl_int mergeN);
extern void (^inverse_mt_transfer_kernel)(const cl_ndrange *ndrange, cl_image mt_img_dest, cl_image mt_img_src, cl_int layerN);
