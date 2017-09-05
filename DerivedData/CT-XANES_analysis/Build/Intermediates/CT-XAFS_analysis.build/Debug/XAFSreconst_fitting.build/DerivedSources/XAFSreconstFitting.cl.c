/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "XAFSreconstFitting.cl.h"

static void initBlocks(void);

// Initialize static data structures
static block_kernel_pair pair_map[22] = {
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL }
};

static block_kernel_map bmap = { 0, 22, initBlocks, pair_map };

// Block function
void (^assign2FittingEq_kernel)(const cl_ndrange *ndrange, cl_float* p, cl_image mt_fit_img, cl_float* energyList, cl_int Enum) =
^(const cl_ndrange *ndrange, cl_float* p, cl_image mt_fit_img, cl_float* energyList, cl_int Enum) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel assign2FittingEq does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, p, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_fit_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, energyList, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(Enum), &Enum, &kargs);
  gcl_log_cl_fatal(err, "setting argument for assign2FittingEq failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing assign2FittingEq failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^assign2Jacobian_kernel)(const cl_ndrange *ndrange, cl_float* p, cl_image jacob_img, cl_float* energyList, cl_int Enum) =
^(const cl_ndrange *ndrange, cl_float* p, cl_image jacob_img, cl_float* energyList, cl_int Enum) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel assign2Jacobian does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, p, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, jacob_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, energyList, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(Enum), &Enum, &kargs);
  gcl_log_cl_fatal(err, "setting argument for assign2Jacobian failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing assign2Jacobian failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^projectionToDeltaMt_kernel)(const cl_ndrange *ndrange, cl_image mt_fit_img, cl_image prj_mt_img, cl_image prj_delta_mt_img, cl_float* anglelist, cl_int sub, cl_int Enum) =
^(const cl_ndrange *ndrange, cl_image mt_fit_img, cl_image prj_mt_img, cl_image prj_delta_mt_img, cl_float* anglelist, cl_int sub, cl_int Enum) {
  int err = 0;
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel projectionToDeltaMt does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_fit_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_mt_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, prj_delta_mt_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, anglelist, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(sub), &sub, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(Enum), &Enum, &kargs);
  gcl_log_cl_fatal(err, "setting argument for projectionToDeltaMt failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing projectionToDeltaMt failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^projectionArray_kernel)(const cl_ndrange *ndrange, cl_image src_img, cl_image prj_img, cl_float* anglelist, cl_int sub) =
^(const cl_ndrange *ndrange, cl_image src_img, cl_image prj_img, cl_float* anglelist, cl_int sub) {
  int err = 0;
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel projectionArray does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, src_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, anglelist, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(sub), &sub, &kargs);
  gcl_log_cl_fatal(err, "setting argument for projectionArray failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing projectionArray failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^backProjectionSingle_kernel)(const cl_ndrange *ndrange, cl_float* bprj_img, cl_image prj_img, cl_float* anglelist, cl_int sub) =
^(const cl_ndrange *ndrange, cl_float* bprj_img, cl_image prj_img, cl_float* anglelist, cl_int sub) {
  int err = 0;
  cl_kernel k = bmap.map[4].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[4].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel backProjectionSingle does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, bprj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, anglelist, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(sub), &sub, &kargs);
  gcl_log_cl_fatal(err, "setting argument for backProjectionSingle failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing backProjectionSingle failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^backProjectionArray_kernel)(const cl_ndrange *ndrange, cl_float* bprj_img, cl_image prj_img, cl_float* anglelist, cl_int sub) =
^(const cl_ndrange *ndrange, cl_float* bprj_img, cl_image prj_img, cl_float* anglelist, cl_int sub) {
  int err = 0;
  cl_kernel k = bmap.map[5].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[5].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel backProjectionArray does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, bprj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, anglelist, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(sub), &sub, &kargs);
  gcl_log_cl_fatal(err, "setting argument for backProjectionArray failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing backProjectionArray failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^backProjectionArrayFull_kernel)(const cl_ndrange *ndrange, cl_float* bprj_img, cl_image prj_img, cl_float* anglelist) =
^(const cl_ndrange *ndrange, cl_float* bprj_img, cl_image prj_img, cl_float* anglelist) {
  int err = 0;
  cl_kernel k = bmap.map[6].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[6].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel backProjectionArrayFull does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, bprj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, anglelist, &kargs);
  gcl_log_cl_fatal(err, "setting argument for backProjectionArrayFull failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing backProjectionArrayFull failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^calcChi2_1_kernel)(const cl_ndrange *ndrange, cl_float* chi2, cl_image prj_delta_mt_img) =
^(const cl_ndrange *ndrange, cl_float* chi2, cl_image prj_delta_mt_img) {
  int err = 0;
  cl_kernel k = bmap.map[7].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[7].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel calcChi2_1 does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, chi2, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_delta_mt_img, &kargs);
  gcl_log_cl_fatal(err, "setting argument for calcChi2_1 failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing calcChi2_1 failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^calcChi2_2_kernel)(const cl_ndrange *ndrange, cl_float* chi2, size_t loc_mem) =
^(const cl_ndrange *ndrange, cl_float* chi2, size_t loc_mem) {
  int err = 0;
  cl_kernel k = bmap.map[8].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[8].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel calcChi2_2 does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, chi2, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, loc_mem, NULL, &kargs);
  gcl_log_cl_fatal(err, "setting argument for calcChi2_2 failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing calcChi2_2 failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^calc_tJJ_tJdF_kernel)(const cl_ndrange *ndrange, cl_float* bprj_jacob, cl_float* bprj_delta_mt, cl_float* tJJ, cl_float* tJdF) =
^(const cl_ndrange *ndrange, cl_float* bprj_jacob, cl_float* bprj_delta_mt, cl_float* tJJ, cl_float* tJdF) {
  int err = 0;
  cl_kernel k = bmap.map[9].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[9].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel calc_tJJ_tJdF does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, bprj_jacob, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, bprj_delta_mt, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, tJJ, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, tJdF, &kargs);
  gcl_log_cl_fatal(err, "setting argument for calc_tJJ_tJdF failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing calc_tJJ_tJdF failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^calc_pCandidate_kernel)(const cl_ndrange *ndrange, cl_float* p_cnd, cl_float* tJJ, cl_float* tJdF, cl_float lambda, cl_float* dL, cl_char* p_fix) =
^(const cl_ndrange *ndrange, cl_float* p_cnd, cl_float* tJJ, cl_float* tJdF, cl_float lambda, cl_float* dL, cl_char* p_fix) {
  int err = 0;
  cl_kernel k = bmap.map[10].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[10].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel calc_pCandidate does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, p_cnd, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, tJJ, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, tJdF, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(lambda), &lambda, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, dL, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, p_fix, &kargs);
  gcl_log_cl_fatal(err, "setting argument for calc_pCandidate failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing calc_pCandidate failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^calc_dL_kernel)(const cl_ndrange *ndrange, cl_float* dL, size_t loc_mem) =
^(const cl_ndrange *ndrange, cl_float* dL, size_t loc_mem) {
  int err = 0;
  cl_kernel k = bmap.map[11].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[11].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel calc_dL does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, dL, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, loc_mem, NULL, &kargs);
  gcl_log_cl_fatal(err, "setting argument for calc_dL failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing calc_dL failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^setConstrain_kernel)(const cl_ndrange *ndrange, cl_float* p_cnd, cl_float* C_mat, cl_float* D_vec) =
^(const cl_ndrange *ndrange, cl_float* p_cnd, cl_float* C_mat, cl_float* D_vec) {
  int err = 0;
  cl_kernel k = bmap.map[12].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[12].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel setConstrain does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, p_cnd, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, C_mat, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, D_vec, &kargs);
  gcl_log_cl_fatal(err, "setting argument for setConstrain failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing setConstrain failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^sinogramCorrection_kernel)(const cl_ndrange *ndrange, cl_image prj_img_src, cl_image prj_img_dst, cl_float* angle, cl_int mode, cl_int Enum) =
^(const cl_ndrange *ndrange, cl_image prj_img_src, cl_image prj_img_dst, cl_float* angle, cl_int mode, cl_int Enum) {
  int err = 0;
  cl_kernel k = bmap.map[13].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[13].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel sinogramCorrection does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, prj_img_src, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_img_dst, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, angle, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(mode), &mode, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(Enum), &Enum, &kargs);
  gcl_log_cl_fatal(err, "setting argument for sinogramCorrection failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing sinogramCorrection failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^circleAttenuator_kernel)(const cl_ndrange *ndrange, cl_float* p, cl_float* attenuator) =
^(const cl_ndrange *ndrange, cl_float* p, cl_float* attenuator) {
  int err = 0;
  cl_kernel k = bmap.map[14].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[14].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel circleAttenuator does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, p, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, attenuator, &kargs);
  gcl_log_cl_fatal(err, "setting argument for circleAttenuator failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing circleAttenuator failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^parameterMask_kernel)(const cl_ndrange *ndrange, cl_float* p, cl_float* mask_img, cl_float* attenuator) =
^(const cl_ndrange *ndrange, cl_float* p, cl_float* mask_img, cl_float* attenuator) {
  int err = 0;
  cl_kernel k = bmap.map[15].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[15].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel parameterMask does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, p, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mask_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, attenuator, &kargs);
  gcl_log_cl_fatal(err, "setting argument for parameterMask failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing parameterMask failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^assign2FittingEq_EArray_kernel)(const cl_ndrange *ndrange, cl_float* p, cl_image mt_fit_img, cl_float* energyList) =
^(const cl_ndrange *ndrange, cl_float* p, cl_image mt_fit_img, cl_float* energyList) {
  int err = 0;
  cl_kernel k = bmap.map[16].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[16].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel assign2FittingEq_EArray does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, p, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_fit_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, energyList, &kargs);
  gcl_log_cl_fatal(err, "setting argument for assign2FittingEq_EArray failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing assign2FittingEq_EArray failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^assign2Jacobian_EArray_kernel)(const cl_ndrange *ndrange, cl_float* p, cl_image jacob_img, cl_float* energyList) =
^(const cl_ndrange *ndrange, cl_float* p, cl_image jacob_img, cl_float* energyList) {
  int err = 0;
  cl_kernel k = bmap.map[17].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[17].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel assign2Jacobian_EArray does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, p, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, jacob_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, energyList, &kargs);
  gcl_log_cl_fatal(err, "setting argument for assign2Jacobian_EArray failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing assign2Jacobian_EArray failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^projectionToDeltaMt_Earray_kernel)(const cl_ndrange *ndrange, cl_image mt_fit_img, cl_image prj_mt_img, cl_image prj_delta_mt_img, cl_float* anglelist, cl_int sub) =
^(const cl_ndrange *ndrange, cl_image mt_fit_img, cl_image prj_mt_img, cl_image prj_delta_mt_img, cl_float* anglelist, cl_int sub) {
  int err = 0;
  cl_kernel k = bmap.map[18].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[18].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel projectionToDeltaMt_Earray does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_fit_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_mt_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, prj_delta_mt_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, anglelist, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(sub), &sub, &kargs);
  gcl_log_cl_fatal(err, "setting argument for projectionToDeltaMt_Earray failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing projectionToDeltaMt_Earray failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^calcChi2_1_EArray_kernel)(const cl_ndrange *ndrange, cl_float* chi2, cl_image prj_delta_mt_img) =
^(const cl_ndrange *ndrange, cl_float* chi2, cl_image prj_delta_mt_img) {
  int err = 0;
  cl_kernel k = bmap.map[19].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[19].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel calcChi2_1_EArray does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, chi2, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_delta_mt_img, &kargs);
  gcl_log_cl_fatal(err, "setting argument for calcChi2_1_EArray failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing calcChi2_1_EArray failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^calc_tJJ_tJdF_EArray_kernel)(const cl_ndrange *ndrange, cl_float* bprj_jacob, cl_float* bprj_delta_mt, cl_float* tJJ, cl_float* tJdF, cl_int Enum) =
^(const cl_ndrange *ndrange, cl_float* bprj_jacob, cl_float* bprj_delta_mt, cl_float* tJJ, cl_float* tJdF, cl_int Enum) {
  int err = 0;
  cl_kernel k = bmap.map[20].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[20].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel calc_tJJ_tJdF_EArray does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, bprj_jacob, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, bprj_delta_mt, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, tJJ, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, tJdF, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(Enum), &Enum, &kargs);
  gcl_log_cl_fatal(err, "setting argument for calc_tJJ_tJdF_EArray failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing calc_tJJ_tJdF_EArray failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^assign2Jacobian_2_kernel)(const cl_ndrange *ndrange, cl_float* p, cl_float* jacob_buff, cl_float* energyList, cl_int Enum) =
^(const cl_ndrange *ndrange, cl_float* p, cl_float* jacob_buff, cl_float* energyList, cl_int Enum) {
  int err = 0;
  cl_kernel k = bmap.map[21].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[21].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel assign2Jacobian_2 does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, p, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, jacob_buff, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, energyList, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(Enum), &Enum, &kargs);
  gcl_log_cl_fatal(err, "setting argument for assign2Jacobian_2 failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing assign2Jacobian_2 failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/XAFSreconstFitting.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == assign2FittingEq_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "assign2FittingEq", &err);
          assert(bmap.map[1].block_ptr == assign2Jacobian_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "assign2Jacobian", &err);
          assert(bmap.map[2].block_ptr == projectionToDeltaMt_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "projectionToDeltaMt", &err);
          assert(bmap.map[3].block_ptr == projectionArray_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "projectionArray", &err);
          assert(bmap.map[4].block_ptr == backProjectionSingle_kernel && "mismatch block");
          bmap.map[4].kernel = clCreateKernel(bmap.program, "backProjectionSingle", &err);
          assert(bmap.map[5].block_ptr == backProjectionArray_kernel && "mismatch block");
          bmap.map[5].kernel = clCreateKernel(bmap.program, "backProjectionArray", &err);
          assert(bmap.map[6].block_ptr == backProjectionArrayFull_kernel && "mismatch block");
          bmap.map[6].kernel = clCreateKernel(bmap.program, "backProjectionArrayFull", &err);
          assert(bmap.map[7].block_ptr == calcChi2_1_kernel && "mismatch block");
          bmap.map[7].kernel = clCreateKernel(bmap.program, "calcChi2_1", &err);
          assert(bmap.map[8].block_ptr == calcChi2_2_kernel && "mismatch block");
          bmap.map[8].kernel = clCreateKernel(bmap.program, "calcChi2_2", &err);
          assert(bmap.map[9].block_ptr == calc_tJJ_tJdF_kernel && "mismatch block");
          bmap.map[9].kernel = clCreateKernel(bmap.program, "calc_tJJ_tJdF", &err);
          assert(bmap.map[10].block_ptr == calc_pCandidate_kernel && "mismatch block");
          bmap.map[10].kernel = clCreateKernel(bmap.program, "calc_pCandidate", &err);
          assert(bmap.map[11].block_ptr == calc_dL_kernel && "mismatch block");
          bmap.map[11].kernel = clCreateKernel(bmap.program, "calc_dL", &err);
          assert(bmap.map[12].block_ptr == setConstrain_kernel && "mismatch block");
          bmap.map[12].kernel = clCreateKernel(bmap.program, "setConstrain", &err);
          assert(bmap.map[13].block_ptr == sinogramCorrection_kernel && "mismatch block");
          bmap.map[13].kernel = clCreateKernel(bmap.program, "sinogramCorrection", &err);
          assert(bmap.map[14].block_ptr == circleAttenuator_kernel && "mismatch block");
          bmap.map[14].kernel = clCreateKernel(bmap.program, "circleAttenuator", &err);
          assert(bmap.map[15].block_ptr == parameterMask_kernel && "mismatch block");
          bmap.map[15].kernel = clCreateKernel(bmap.program, "parameterMask", &err);
          assert(bmap.map[16].block_ptr == assign2FittingEq_EArray_kernel && "mismatch block");
          bmap.map[16].kernel = clCreateKernel(bmap.program, "assign2FittingEq_EArray", &err);
          assert(bmap.map[17].block_ptr == assign2Jacobian_EArray_kernel && "mismatch block");
          bmap.map[17].kernel = clCreateKernel(bmap.program, "assign2Jacobian_EArray", &err);
          assert(bmap.map[18].block_ptr == projectionToDeltaMt_Earray_kernel && "mismatch block");
          bmap.map[18].kernel = clCreateKernel(bmap.program, "projectionToDeltaMt_Earray", &err);
          assert(bmap.map[19].block_ptr == calcChi2_1_EArray_kernel && "mismatch block");
          bmap.map[19].kernel = clCreateKernel(bmap.program, "calcChi2_1_EArray", &err);
          assert(bmap.map[20].block_ptr == calc_tJJ_tJdF_EArray_kernel && "mismatch block");
          bmap.map[20].kernel = clCreateKernel(bmap.program, "calc_tJJ_tJdF_EArray", &err);
          assert(bmap.map[21].block_ptr == assign2Jacobian_2_kernel && "mismatch block");
          bmap.map[21].kernel = clCreateKernel(bmap.program, "assign2Jacobian_2", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = assign2FittingEq_kernel;
  bmap.map[1].block_ptr = assign2Jacobian_kernel;
  bmap.map[2].block_ptr = projectionToDeltaMt_kernel;
  bmap.map[3].block_ptr = projectionArray_kernel;
  bmap.map[4].block_ptr = backProjectionSingle_kernel;
  bmap.map[5].block_ptr = backProjectionArray_kernel;
  bmap.map[6].block_ptr = backProjectionArrayFull_kernel;
  bmap.map[7].block_ptr = calcChi2_1_kernel;
  bmap.map[8].block_ptr = calcChi2_2_kernel;
  bmap.map[9].block_ptr = calc_tJJ_tJdF_kernel;
  bmap.map[10].block_ptr = calc_pCandidate_kernel;
  bmap.map[11].block_ptr = calc_dL_kernel;
  bmap.map[12].block_ptr = setConstrain_kernel;
  bmap.map[13].block_ptr = sinogramCorrection_kernel;
  bmap.map[14].block_ptr = circleAttenuator_kernel;
  bmap.map[15].block_ptr = parameterMask_kernel;
  bmap.map[16].block_ptr = assign2FittingEq_EArray_kernel;
  bmap.map[17].block_ptr = assign2Jacobian_EArray_kernel;
  bmap.map[18].block_ptr = projectionToDeltaMt_Earray_kernel;
  bmap.map[19].block_ptr = calcChi2_1_EArray_kernel;
  bmap.map[20].block_ptr = calc_tJJ_tJdF_EArray_kernel;
  bmap.map[21].block_ptr = assign2Jacobian_2_kernel;
}

