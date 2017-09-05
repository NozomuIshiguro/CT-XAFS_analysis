/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^spinFactor_kernel)(const cl_ndrange *ndrange, cl_float2* W);
extern void (^zeroPadding_kernel)(const cl_ndrange *ndrange, cl_image prj_img, cl_float2* xc);
extern void (^bitReverse_kernel)(const cl_ndrange *ndrange, cl_float2* xc_src, cl_float2* xc_dest, cl_uint iter);
extern void (^butterfly_kernel)(const cl_ndrange *ndrange, cl_float2* xc, cl_float2* W, cl_uint flag, cl_int iter);
extern void (^filtering_kernel)(const cl_ndrange *ndrange, cl_float2* xc);
extern void (^normalization_kernel)(const cl_ndrange *ndrange, cl_float2* xc);
extern void (^outputImage_kernel)(const cl_ndrange *ndrange, cl_float2* xc, cl_image fprj_img);
extern void (^backProjectionFBP_kernel)(const cl_ndrange *ndrange, cl_image fprj_img, cl_image reconst_img, cl_float* angle);
