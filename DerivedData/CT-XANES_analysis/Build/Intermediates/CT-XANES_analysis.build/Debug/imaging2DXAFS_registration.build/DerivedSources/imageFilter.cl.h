/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^meanImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_uint radius);
extern void (^addImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float val);
extern void (^subtractImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float val);
extern void (^MultiplyImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float val);
extern void (^DivideImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float val);
extern void (^RemoveNANImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float val);
extern void (^MinImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float minval);
extern void (^MaxImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float maxval);
extern void (^expImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2);
extern void (^lnImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2);
extern void (^rapizoralImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2);
