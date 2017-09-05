/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^sinogramCorrection_kernel)(const cl_ndrange *ndrange, cl_image prj_img_src, cl_image prj_img_dst, cl_float* angle, cl_int mode, cl_float a, cl_float b);
extern void (^setThreshold_kernel)(const cl_ndrange *ndrange, cl_image img_src, cl_image img_dest, cl_float threshold);
extern void (^baseUp_kernel)(const cl_ndrange *ndrange, cl_image img_src, cl_image img_dest, cl_float* baseup, cl_int order);
extern void (^findMinimumX_kernel)(const cl_ndrange *ndrange, cl_image img_src, size_t loc_mem, cl_float* minimumY);
extern void (^findMinimumY_kernel)(const cl_ndrange *ndrange, cl_float* minimumY, size_t loc_mem, cl_float* minimum);
extern void (^partialDerivativeOfGradiant_kernel)(const cl_ndrange *ndrange, cl_image original_img_src, cl_image img_src, cl_image img_dest, cl_float epsilon, cl_float alpha);
extern void (^Profection_kernel)(const cl_ndrange *ndrange, cl_image reconst_img, cl_image prj_img, cl_float* angle, cl_int sub);
extern void (^backProjection_kernel)(const cl_ndrange *ndrange, cl_image reconst_dest_img, cl_image prj_img, cl_float* angle, cl_int sub);
