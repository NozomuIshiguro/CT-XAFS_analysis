/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <OpenCL/opencl.h>
extern void (^assign2FittingEq_kernel)(const cl_ndrange *ndrange, cl_float* p, cl_image mt_fit_img, cl_float* energyList, cl_int Enum);
extern void (^assign2Jacobian_kernel)(const cl_ndrange *ndrange, cl_float* p, cl_image jacob_img, cl_float* energyList, cl_int Enum);
extern void (^projectionToDeltaMt_kernel)(const cl_ndrange *ndrange, cl_image mt_fit_img, cl_image prj_mt_img, cl_image prj_delta_mt_img, cl_float* anglelist, cl_int sub, cl_int Enum);
extern void (^projectionArray_kernel)(const cl_ndrange *ndrange, cl_image src_img, cl_image prj_img, cl_float* anglelist, cl_int sub);
extern void (^backProjectionSingle_kernel)(const cl_ndrange *ndrange, cl_float* bprj_img, cl_image prj_img, cl_float* anglelist, cl_int sub);
extern void (^backProjectionArray_kernel)(const cl_ndrange *ndrange, cl_float* bprj_img, cl_image prj_img, cl_float* anglelist, cl_int sub);
extern void (^backProjectionArrayFull_kernel)(const cl_ndrange *ndrange, cl_float* bprj_img, cl_image prj_img, cl_float* anglelist);
extern void (^calcChi2_1_kernel)(const cl_ndrange *ndrange, cl_float* chi2, cl_image prj_delta_mt_img);
extern void (^calcChi2_2_kernel)(const cl_ndrange *ndrange, cl_float* chi2, size_t loc_mem);
extern void (^calc_tJJ_tJdF_kernel)(const cl_ndrange *ndrange, cl_float* bprj_jacob, cl_float* bprj_delta_mt, cl_float* tJJ, cl_float* tJdF);
extern void (^calc_pCandidate_kernel)(const cl_ndrange *ndrange, cl_float* p_cnd, cl_float* tJJ, cl_float* tJdF, cl_float lambda, cl_float* dL, cl_char* p_fix);
extern void (^calc_dL_kernel)(const cl_ndrange *ndrange, cl_float* dL, size_t loc_mem);
extern void (^setConstrain_kernel)(const cl_ndrange *ndrange, cl_float* p_cnd, cl_float* C_mat, cl_float* D_vec);
extern void (^sinogramCorrection_kernel)(const cl_ndrange *ndrange, cl_image prj_img_src, cl_image prj_img_dst, cl_float* angle, cl_int mode, cl_int Enum);
extern void (^circleAttenuator_kernel)(const cl_ndrange *ndrange, cl_float* p, cl_float* attenuator);
extern void (^parameterMask_kernel)(const cl_ndrange *ndrange, cl_float* p, cl_float* mask_img, cl_float* attenuator);
extern void (^assign2FittingEq_EArray_kernel)(const cl_ndrange *ndrange, cl_float* p, cl_image mt_fit_img, cl_float* energyList);
extern void (^assign2Jacobian_EArray_kernel)(const cl_ndrange *ndrange, cl_float* p, cl_image jacob_img, cl_float* energyList);
extern void (^projectionToDeltaMt_Earray_kernel)(const cl_ndrange *ndrange, cl_image mt_fit_img, cl_image prj_mt_img, cl_image prj_delta_mt_img, cl_float* anglelist, cl_int sub);
extern void (^calcChi2_1_EArray_kernel)(const cl_ndrange *ndrange, cl_float* chi2, cl_image prj_delta_mt_img);
extern void (^calc_tJJ_tJdF_EArray_kernel)(const cl_ndrange *ndrange, cl_float* bprj_jacob, cl_float* bprj_delta_mt, cl_float* tJJ, cl_float* tJdF, cl_int Enum);
extern void (^assign2Jacobian_2_kernel)(const cl_ndrange *ndrange, cl_float* p, cl_float* jacob_buff, cl_float* energyList, cl_int Enum);
