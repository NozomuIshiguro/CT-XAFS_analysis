/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^estimateBkg_kernel)(const cl_ndrange *ndrange, cl_float* bkg_img, cl_float* fp_img, cl_int funcmode, cl_float E0);
extern void (^estimateEJ_kernel)(const cl_ndrange *ndrange, cl_float* ej_img, cl_float* bkg_img, cl_float* fp_img);
extern void (^redimension_mt2chi_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* chi_img, cl_float* energy, cl_int numPnts, cl_int koffset, cl_int ksize);
extern void (^convert2realChi_kernel)(const cl_ndrange *ndrange, cl_float2* chi_cmplx_img, cl_float* chi_img);
extern void (^Bspline_basis_zero_kernel)(const cl_ndrange *ndrange, cl_float* basis, cl_int N_ctrlP, cl_float zoneStart, cl_float zonePitch, cl_int basisOffset);
extern void (^Bspline_basis_updateOrder_kernel)(const cl_ndrange *ndrange, cl_float* basis_src, cl_float* basis_dest, cl_int N_ctrlP, cl_float zoneStart, cl_float zonePitch, cl_int basisOffset, cl_int order);
extern void (^Bspline_kernel)(const cl_ndrange *ndrange, cl_float* spline, cl_float* basis, cl_float* ctrlP, cl_int N_ctrlP, cl_int basisOffset, cl_int order);
extern void (^BsplineRemoval_kernel)(const cl_ndrange *ndrange, cl_float2* chiData_img, cl_float2* chiFit_img, cl_float* basis, cl_float* ctrlP, cl_int N_ctrlP, cl_int basisOffset, cl_int order);
extern void (^Jacobian_BsplineRemoval_kernel)(const cl_ndrange *ndrange, cl_float2* J, cl_int paraID, cl_float* basis, cl_int N_ctrlP, cl_int basisOffset, cl_int order);
extern void (^kweight_kernel)(const cl_ndrange *ndrange, cl_float2* chi_img, cl_int kw);
extern void (^initialCtrlP_kernel)(const cl_ndrange *ndrange, cl_float2* chiData_img, cl_float* ctrlP_img, cl_float knotPitch, cl_int ksize);
