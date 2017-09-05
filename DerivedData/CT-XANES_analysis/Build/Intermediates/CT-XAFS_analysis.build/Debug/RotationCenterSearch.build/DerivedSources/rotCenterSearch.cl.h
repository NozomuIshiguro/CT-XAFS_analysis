/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^rotCenterShift_kernel)(const cl_ndrange *ndrange, cl_image prj_input_img, cl_image prj_output_img, cl_float rotCenterShift);
extern void (^setMask_kernel)(const cl_ndrange *ndrange, cl_image mask_img, cl_int offsetN, cl_float startShift, cl_float shiftStep, cl_float min_ang, cl_float max_ang);
extern void (^imgAVG_kernel)(const cl_ndrange *ndrange, cl_image img, cl_image mask_img, size_t loc_mem, cl_float* avg);
extern void (^imgSTDEV_kernel)(const cl_ndrange *ndrange, cl_image img, cl_image mask_img, size_t loc_mem, cl_float* avg, cl_float* stedev);
extern void (^imgFocusIndex_kernel)(const cl_ndrange *ndrange, cl_image img, cl_image mask_img, size_t loc_mem, cl_float* Findex);
