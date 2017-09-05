/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^XY_transpose_kernel)(const cl_ndrange *ndrange, cl_float2* src, cl_float2* dest);
extern void (^XZ_transpose_kernel)(const cl_ndrange *ndrange, cl_float2* src, cl_float2* dest, cl_int offsetX_src, cl_int x_size_src, cl_int offsetX_dst, cl_int x_size_dst, cl_int offsetY_src, cl_int y_size_src, cl_int offsetY_dst, cl_int y_size_dst);
extern void (^spinFact_kernel)(const cl_ndrange *ndrange, cl_float2* w);
extern void (^bitReverse_kernel)(const cl_ndrange *ndrange, cl_float2* src, cl_float2* dest, cl_int M);
extern void (^bitReverseAndXZ_transpose_kernel)(const cl_ndrange *ndrange, cl_float2* src, cl_float2* dest, cl_int M, cl_int offsetX_src, cl_int x_size_src, cl_int offsetX_dst, cl_int x_size_dst, cl_int offsetY_src, cl_int y_size_src, cl_int offsetY_dst, cl_int y_size_dst);
extern void (^butterfly_kernel)(const cl_ndrange *ndrange, cl_float2* x_fft, cl_float2* w, cl_uint iter, cl_uint flag);
extern void (^FFTnorm_kernel)(const cl_ndrange *ndrange, cl_float2* x_fft, cl_float xgrid);
extern void (^IFFTnorm_kernel)(const cl_ndrange *ndrange, cl_float2* x_ifft, cl_float xgrid);
