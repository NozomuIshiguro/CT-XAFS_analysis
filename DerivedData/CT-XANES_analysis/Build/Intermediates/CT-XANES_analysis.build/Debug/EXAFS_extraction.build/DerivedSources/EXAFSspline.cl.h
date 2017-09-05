/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^Bspline_basis_zero_kernel)(const cl_ndrange *ndrange, cl_float* basis, cl_float h);
extern void (^Bspline_orderUpdatingMatrix_kernel)(const cl_ndrange *ndrange, cl_float* OUM, cl_float h, cl_int order);
extern void (^Bspline_basis_updateOrder_kernel)(const cl_ndrange *ndrange, cl_float* basis_src, cl_float* basis_dest, cl_float* OUM);
extern void (^Bspline_kernel)(const cl_ndrange *ndrange, cl_float* spline, cl_float* basis);
