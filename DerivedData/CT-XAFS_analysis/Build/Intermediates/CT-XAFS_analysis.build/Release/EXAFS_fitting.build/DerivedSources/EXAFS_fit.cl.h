/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^hanningWindowFuncIMGarray_kernel)(const cl_ndrange *ndrange, cl_float2* wave, cl_float zmin, cl_float zmax, cl_float windz, cl_float zgrid);
extern void (^redimension_feffShellPara_kernel)(const cl_ndrange *ndrange, cl_float* paraW, cl_image paraW_raw, cl_float* kw, cl_int numPnts);
extern void (^outputchi_kernel)(const cl_ndrange *ndrange, cl_float2* chi, cl_float Reff, cl_float* S02, cl_float* CN, cl_float* dR, cl_float* dE0, cl_float* ss, cl_float* E0imag, cl_float* C3, cl_float* C4, cl_float* real2phcW, cl_float* magW, cl_float* phaseW, cl_float* redFactorW, cl_float* lambdaW, cl_float* real_pW, cl_int kw);
extern void (^outputchi_r_kernel)(const cl_ndrange *ndrange, cl_float* chi, cl_float Reff, cl_float* S02, cl_float* CN, cl_float* dR, cl_float* dE0, cl_float* ss, cl_float* E0imag, cl_float* C3, cl_float* C4, cl_float* real2phcW, cl_float* magW, cl_float* phaseW, cl_float* redFactorW, cl_float* lambdaW, cl_float* real_pW, cl_int kw);
extern void (^Jacobian_k_new_kernel)(const cl_ndrange *ndrange, cl_float2* J, cl_float Reff, cl_float* S02, cl_float* CN, cl_float* dR, cl_float* dE0, cl_float* ss, cl_float* E0imag, cl_float* C3, cl_float* C4, cl_float* real2phcW, cl_float* magW, cl_float* phaseW, cl_float* redFactorW, cl_float* lambdaW, cl_float* real_pW, cl_int kw, cl_int paramode, cl_int realpart);
extern void (^Jacobian_k_kernel)(const cl_ndrange *ndrange, cl_float2* J, cl_float Reff, cl_float* S02, cl_float* CN, cl_float* dR, cl_float* dE0, cl_float* ss, cl_float* E0imag, cl_float* C3, cl_float* C4, cl_float* real2phcW, cl_float* magW, cl_float* phaseW, cl_float* redFactorW, cl_float* lambdaW, cl_float* real_pW, cl_int kw, cl_int paramode, cl_int realpart);
extern void (^estimate_tJJ_kernel)(const cl_ndrange *ndrange, cl_float* tJJ, cl_float2* J1, cl_float2* J2, cl_int z_id, cl_int kRqsize);
extern void (^estimate_tJdF_kernel)(const cl_ndrange *ndrange, cl_float* tJdF, cl_float2* J, cl_float2* chi_data, cl_float2* chi_fit, cl_int z_id, cl_int kRqsize);
extern void (^estimate_dF2_kernel)(const cl_ndrange *ndrange, cl_float* dF2, cl_float2* chi_data, cl_float2* chi_fit, cl_int kRqsize);
extern void (^estimate_Rfactor_kernel)(const cl_ndrange *ndrange, cl_float* Rfactor, cl_float2* chi_data, cl_float2* chi_fit, cl_int kRqsize);
extern void (^estimate_error_kernel)(const cl_ndrange *ndrange, cl_float* tJdF, cl_float* p_error);
extern void (^chi2cmplxChi_imgStck_kernel)(const cl_ndrange *ndrange, cl_float* chi, cl_float2* chi_c, cl_int kn, cl_int koffset, cl_int kw);
extern void (^chi2cmplxChi_chiStck_kernel)(const cl_ndrange *ndrange, cl_float* chi, cl_float2* chi_c, cl_int XY, cl_int kw, cl_int XY_size);
extern void (^CNweighten_kernel)(const cl_ndrange *ndrange, cl_float* CN, cl_float* edgeJ, cl_float iniCN);
extern void (^outputBondDistance_kernel)(const cl_ndrange *ndrange, cl_float* dR, cl_float* R, cl_float Reff);
