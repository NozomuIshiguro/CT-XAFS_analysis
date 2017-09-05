/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^reslice_kernel)(const cl_ndrange *ndrange, cl_image mt_img, cl_image prj_img, cl_float* Xshift, cl_float* Yshift, cl_float baseup, cl_int th, cl_int th_offset, cl_char correction);
extern void (^xprojection_kernel)(const cl_ndrange *ndrange, cl_image mt_img, cl_float* xproj, cl_int startX, cl_int endX, cl_int th);
extern void (^zcorrection_kernel)(const cl_ndrange *ndrange, cl_image xprj_img, cl_float* yshift, size_t target_xprj, size_t loc_mem, cl_int startY, cl_int endY);
