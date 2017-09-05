/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^powerIter1_kernel)(const cl_ndrange *ndrange, cl_image reconst_img, cl_image prj_img, cl_float* angle);
extern void (^powerIter2_kernel)(const cl_ndrange *ndrange, cl_image reconst_cnd_img, cl_image prj_img, cl_float* angle);
extern void (^powerIter3_kernel)(const cl_ndrange *ndrange, cl_image reconst_cnd_img, cl_image reconst_new_img, cl_float* L2abs);
extern void (^imageL2AbsX_kernel)(const cl_ndrange *ndrange, cl_image img_src, size_t loc_mem, cl_float* L2absY);
extern void (^imageL2AbsY_kernel)(const cl_ndrange *ndrange, cl_float* L2absY, size_t loc_mem, cl_float* L2abs);
extern void (^FISTAbackProjection_kernel)(const cl_ndrange *ndrange, cl_image reconst_img, cl_image reconst_dest_img, cl_image dprj_img, cl_float* angle, cl_float* L, cl_int sub, cl_float alpha);
extern void (^ISTA_kernel)(const cl_ndrange *ndrange, cl_image reconst_v_img, cl_image reconst_x_new_img, cl_int sub, cl_float* L);
extern void (^FISTA_kernel)(const cl_ndrange *ndrange, cl_image reconst_x_img, cl_image reconst_v_img, cl_image reconst_b_img, cl_image reconst_w_new_img, cl_image reconst_x_new_img, cl_image reconst_b_new_img, cl_int sub, cl_float* L);
