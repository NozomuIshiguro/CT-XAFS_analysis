/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^mt_conversion_kernel)(const cl_ndrange *ndrange, cl_float* dark, cl_float* I0, cl_ushort* It_buffer, cl_float* mt_buffer, cl_image mt_img1, cl_image mt_img2, cl_int shapeNo, cl_int startpntX, cl_int startpntY, cl_uint width, cl_uint height, cl_float angle, cl_int evaluatemode);
extern void (^mt_transfer_kernel)(const cl_ndrange *ndrange, cl_float* mt_buffer, cl_image mt_img1, cl_image mt_img2, cl_int shapeNo, cl_int startpntX, cl_int startpntY, cl_uint width, cl_uint height, cl_float angle);
extern void (^merge_kernel)(const cl_ndrange *ndrange, cl_image input_img, cl_image output_img, cl_uint mergeN);
extern void (^imageReg1X_kernel)(const cl_ndrange *ndrange, cl_image mt_t_img, cl_image mt_s_img, cl_image weight_img, cl_float* dF2, cl_float* dF, cl_float CI, cl_float* p, cl_float* p_target, cl_char* p_fix, cl_float* dF2X, cl_float* tJdFX, cl_float* tJJX, cl_float* devX, cl_int mergeN, size_t loc_mem, cl_int difstep);
extern void (^imageReg1Y_kernel)(const cl_ndrange *ndrange, cl_float* dF2X, cl_float* tJdFX, cl_float* tJJX, cl_float* devX, cl_float* dF2, cl_float* tJdF, cl_float* tJJ, cl_float* dev, cl_int mergeN, size_t loc_mem);
extern void (^imageReg2X_kernel)(const cl_ndrange *ndrange, cl_image mt_t_img, cl_image mt_s_img, cl_image weight_img, cl_float* p, cl_float* p_target, cl_float* dF2X, cl_float* dFX, cl_float* devX, cl_int mergeN, size_t loc_mem);
extern void (^imageReg2Y_kernel)(const cl_ndrange *ndrange, cl_float* dF2X, cl_float* dFX, cl_float* devX, cl_float* dF2, cl_float* dF, cl_int mergeN, size_t loc_mem);
extern void (^estimateParaError_kernel)(const cl_ndrange *ndrange, cl_float* p_error, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* dev_img);
extern void (^output_imgReg_result_kernel)(const cl_ndrange *ndrange, cl_image mt_img, cl_float* mt_buf, cl_float* p);
extern void (^merge_mt_kernel)(const cl_ndrange *ndrange, cl_image mt_sample, cl_float* mt_output);
extern void (^merge_rawhisdata_kernel)(const cl_ndrange *ndrange, cl_ushort* rawhisdata, cl_float* outputdata, cl_int mergeN);
extern void (^imQXAFS_smoothing_kernel)(const cl_ndrange *ndrange, cl_float* rawmtdata, cl_float* outputdata, cl_int mergeN);
extern void (^updateWeight_kernel)(const cl_ndrange *ndrange, cl_image mt_t_img, cl_image mt_s_img, cl_image weight_img, cl_float* dF2, cl_float* dF, cl_float CI, cl_float* p, cl_float* p_target, cl_int mergeN);
