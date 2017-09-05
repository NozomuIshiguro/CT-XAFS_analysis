/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "EXAFS_fit.cl.h"

static void initBlocks(void);

// Initialize static data structures
static block_kernel_pair pair_map[19] = {
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

static block_kernel_map bmap = { 0, 19, initBlocks, pair_map };

// Block function
void (^hanningWindowFuncIMGarray_kernel)(const cl_ndrange *ndrange, cl_float2* wave, cl_float zmin, cl_float zmax, cl_float windz, cl_float zgrid) =
^(const cl_ndrange *ndrange, cl_float2* wave, cl_float zmin, cl_float zmax, cl_float windz, cl_float zgrid) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel hanningWindowFuncIMGarray does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, wave, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, sizeof(zmin), &zmin, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(zmax), &zmax, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(windz), &windz, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(zgrid), &zgrid, &kargs);
  gcl_log_cl_fatal(err, "setting argument for hanningWindowFuncIMGarray failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing hanningWindowFuncIMGarray failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^redimension_feffShellPara_kernel)(const cl_ndrange *ndrange, cl_float* paraW, cl_image paraW_raw, cl_float* kw, cl_int numPnts) =
^(const cl_ndrange *ndrange, cl_float* paraW, cl_image paraW_raw, cl_float* kw, cl_int numPnts) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel redimension_feffShellPara does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, paraW, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, paraW_raw, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, kw, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(numPnts), &numPnts, &kargs);
  gcl_log_cl_fatal(err, "setting argument for redimension_feffShellPara failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing redimension_feffShellPara failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^outputchi_kernel)(const cl_ndrange *ndrange, cl_float2* chi, cl_float Reff, cl_float* S02, cl_float* CN, cl_float* dR, cl_float* dE0, cl_float* ss, cl_float* E0imag, cl_float* C3, cl_float* C4, cl_float* real2phcW, cl_float* magW, cl_float* phaseW, cl_float* redFactorW, cl_float* lambdaW, cl_float* real_pW, cl_int kw) =
^(const cl_ndrange *ndrange, cl_float2* chi, cl_float Reff, cl_float* S02, cl_float* CN, cl_float* dR, cl_float* dE0, cl_float* ss, cl_float* E0imag, cl_float* C3, cl_float* C4, cl_float* real2phcW, cl_float* magW, cl_float* phaseW, cl_float* redFactorW, cl_float* lambdaW, cl_float* real_pW, cl_int kw) {
  int err = 0;
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel outputchi does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, chi, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, sizeof(Reff), &Reff, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, S02, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, CN, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, dR, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, dE0, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 6, ss, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 7, E0imag, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 8, C3, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 9, C4, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 10, real2phcW, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 11, magW, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 12, phaseW, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 13, redFactorW, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 14, lambdaW, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 15, real_pW, &kargs);
  err |= gclSetKernelArgAPPLE(k, 16, sizeof(kw), &kw, &kargs);
  gcl_log_cl_fatal(err, "setting argument for outputchi failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing outputchi failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^Jacobian_k_new_kernel)(const cl_ndrange *ndrange, cl_float2* J, cl_float Reff, cl_float* S02, cl_float* CN, cl_float* dR, cl_float* dE0, cl_float* ss, cl_float* E0imag, cl_float* C3, cl_float* C4, cl_float* real2phcW, cl_float* magW, cl_float* phaseW, cl_float* redFactorW, cl_float* lambdaW, cl_float* real_pW, cl_int kw, cl_int paramode) =
^(const cl_ndrange *ndrange, cl_float2* J, cl_float Reff, cl_float* S02, cl_float* CN, cl_float* dR, cl_float* dE0, cl_float* ss, cl_float* E0imag, cl_float* C3, cl_float* C4, cl_float* real2phcW, cl_float* magW, cl_float* phaseW, cl_float* redFactorW, cl_float* lambdaW, cl_float* real_pW, cl_int kw, cl_int paramode) {
  int err = 0;
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel Jacobian_k_new does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, J, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, sizeof(Reff), &Reff, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, S02, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, CN, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, dR, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, dE0, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 6, ss, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 7, E0imag, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 8, C3, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 9, C4, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 10, real2phcW, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 11, magW, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 12, phaseW, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 13, redFactorW, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 14, lambdaW, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 15, real_pW, &kargs);
  err |= gclSetKernelArgAPPLE(k, 16, sizeof(kw), &kw, &kargs);
  err |= gclSetKernelArgAPPLE(k, 17, sizeof(paramode), &paramode, &kargs);
  gcl_log_cl_fatal(err, "setting argument for Jacobian_k_new failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing Jacobian_k_new failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^Jacobian_k_kernel)(const cl_ndrange *ndrange, cl_float2* J, cl_float Reff, cl_float* S02, cl_float* CN, cl_float* dR, cl_float* dE0, cl_float* ss, cl_float* E0imag, cl_float* C3, cl_float* C4, cl_float* real2phcW, cl_float* magW, cl_float* phaseW, cl_float* redFactorW, cl_float* lambdaW, cl_float* real_pW, cl_int kw, cl_int paramode, cl_int realpart) =
^(const cl_ndrange *ndrange, cl_float2* J, cl_float Reff, cl_float* S02, cl_float* CN, cl_float* dR, cl_float* dE0, cl_float* ss, cl_float* E0imag, cl_float* C3, cl_float* C4, cl_float* real2phcW, cl_float* magW, cl_float* phaseW, cl_float* redFactorW, cl_float* lambdaW, cl_float* real_pW, cl_int kw, cl_int paramode, cl_int realpart) {
  int err = 0;
  cl_kernel k = bmap.map[4].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[4].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel Jacobian_k does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, J, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, sizeof(Reff), &Reff, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, S02, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, CN, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, dR, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, dE0, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 6, ss, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 7, E0imag, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 8, C3, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 9, C4, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 10, real2phcW, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 11, magW, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 12, phaseW, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 13, redFactorW, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 14, lambdaW, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 15, real_pW, &kargs);
  err |= gclSetKernelArgAPPLE(k, 16, sizeof(kw), &kw, &kargs);
  err |= gclSetKernelArgAPPLE(k, 17, sizeof(paramode), &paramode, &kargs);
  err |= gclSetKernelArgAPPLE(k, 18, sizeof(realpart), &realpart, &kargs);
  gcl_log_cl_fatal(err, "setting argument for Jacobian_k failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing Jacobian_k failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^estimate_tJJ_kernel)(const cl_ndrange *ndrange, cl_float* tJJ, cl_float2* J1, cl_float2* J2, cl_int z_id, cl_int kRqsize) =
^(const cl_ndrange *ndrange, cl_float* tJJ, cl_float2* J1, cl_float2* J2, cl_int z_id, cl_int kRqsize) {
  int err = 0;
  cl_kernel k = bmap.map[5].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[5].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel estimate_tJJ does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, tJJ, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, J1, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, J2, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(z_id), &z_id, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(kRqsize), &kRqsize, &kargs);
  gcl_log_cl_fatal(err, "setting argument for estimate_tJJ failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing estimate_tJJ failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^estimate_tJdF_kernel)(const cl_ndrange *ndrange, cl_float* tJdF, cl_float2* J, cl_float2* chi_data, cl_float2* chi_fit, cl_int z_id, cl_int kRqsize) =
^(const cl_ndrange *ndrange, cl_float* tJdF, cl_float2* J, cl_float2* chi_data, cl_float2* chi_fit, cl_int z_id, cl_int kRqsize) {
  int err = 0;
  cl_kernel k = bmap.map[6].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[6].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel estimate_tJdF does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, tJdF, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, J, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, chi_data, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, chi_fit, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(z_id), &z_id, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(kRqsize), &kRqsize, &kargs);
  gcl_log_cl_fatal(err, "setting argument for estimate_tJdF failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing estimate_tJdF failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^estimate_dF2_kernel)(const cl_ndrange *ndrange, cl_float* dF2, cl_float2* chi_data, cl_float2* chi_fit, cl_int kRqsize) =
^(const cl_ndrange *ndrange, cl_float* dF2, cl_float2* chi_data, cl_float2* chi_fit, cl_int kRqsize) {
  int err = 0;
  cl_kernel k = bmap.map[7].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[7].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel estimate_dF2 does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, dF2, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, chi_data, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, chi_fit, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(kRqsize), &kRqsize, &kargs);
  gcl_log_cl_fatal(err, "setting argument for estimate_dF2 failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing estimate_dF2 failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^estimate_Rfactor_kernel)(const cl_ndrange *ndrange, cl_float* Rfactor, cl_float2* chi_data, cl_float2* chi_fit, cl_int kRqsize) =
^(const cl_ndrange *ndrange, cl_float* Rfactor, cl_float2* chi_data, cl_float2* chi_fit, cl_int kRqsize) {
  int err = 0;
  cl_kernel k = bmap.map[8].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[8].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel estimate_Rfactor does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, Rfactor, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, chi_data, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, chi_fit, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(kRqsize), &kRqsize, &kargs);
  gcl_log_cl_fatal(err, "setting argument for estimate_Rfactor failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing estimate_Rfactor failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^estimate_error_kernel)(const cl_ndrange *ndrange, cl_float* tJdF, cl_float* p_error) =
^(const cl_ndrange *ndrange, cl_float* tJdF, cl_float* p_error) {
  int err = 0;
  cl_kernel k = bmap.map[9].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[9].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel estimate_error does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, tJdF, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, p_error, &kargs);
  gcl_log_cl_fatal(err, "setting argument for estimate_error failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing estimate_error failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^chi2cmplxChi_imgStck_kernel)(const cl_ndrange *ndrange, cl_float* chi, cl_float2* chi_c, cl_int kn, cl_int kw) =
^(const cl_ndrange *ndrange, cl_float* chi, cl_float2* chi_c, cl_int kn, cl_int kw) {
  int err = 0;
  cl_kernel k = bmap.map[10].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[10].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel chi2cmplxChi_imgStck does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, chi, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, chi_c, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(kn), &kn, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(kw), &kw, &kargs);
  gcl_log_cl_fatal(err, "setting argument for chi2cmplxChi_imgStck failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing chi2cmplxChi_imgStck failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^chi2cmplxChi_chiStck_kernel)(const cl_ndrange *ndrange, cl_float* chi, cl_float2* chi_c, cl_int XY, cl_int kw, cl_int XY_size) =
^(const cl_ndrange *ndrange, cl_float* chi, cl_float2* chi_c, cl_int XY, cl_int kw, cl_int XY_size) {
  int err = 0;
  cl_kernel k = bmap.map[11].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[11].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel chi2cmplxChi_chiStck does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, chi, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, chi_c, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(XY), &XY, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(kw), &kw, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(XY_size), &XY_size, &kargs);
  gcl_log_cl_fatal(err, "setting argument for chi2cmplxChi_chiStck failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing chi2cmplxChi_chiStck failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^CNweighten_kernel)(const cl_ndrange *ndrange, cl_float* CN, cl_float* edgeJ, cl_float iniCN) =
^(const cl_ndrange *ndrange, cl_float* CN, cl_float* edgeJ, cl_float iniCN) {
  int err = 0;
  cl_kernel k = bmap.map[12].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[12].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel CNweighten does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, CN, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, edgeJ, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(iniCN), &iniCN, &kargs);
  gcl_log_cl_fatal(err, "setting argument for CNweighten failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing CNweighten failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^LevenbergMarquardt_kernel)(const cl_ndrange *ndrange, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* dp_img, cl_float* lambda_img, cl_float* inv_tJJ_img) =
^(const cl_ndrange *ndrange, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* dp_img, cl_float* lambda_img, cl_float* inv_tJJ_img) {
  int err = 0;
  cl_kernel k = bmap.map[13].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[13].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel LevenbergMarquardt does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, tJdF_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, tJJ_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, dp_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, lambda_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, inv_tJJ_img, &kargs);
  gcl_log_cl_fatal(err, "setting argument for LevenbergMarquardt failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing LevenbergMarquardt failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^estimate_dL_kernel)(const cl_ndrange *ndrange, cl_float* dp_img, cl_float* tJJ_img, cl_float* tJdF_img, cl_float* lambda_img, cl_float* dL_img) =
^(const cl_ndrange *ndrange, cl_float* dp_img, cl_float* tJJ_img, cl_float* tJdF_img, cl_float* lambda_img, cl_float* dL_img) {
  int err = 0;
  cl_kernel k = bmap.map[14].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[14].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel estimate_dL does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, dp_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, tJJ_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, tJdF_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, lambda_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, dL_img, &kargs);
  gcl_log_cl_fatal(err, "setting argument for estimate_dL failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing estimate_dL failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^updatePara_kernel)(const cl_ndrange *ndrange, cl_float* dp, cl_float* fp, cl_int z_id) =
^(const cl_ndrange *ndrange, cl_float* dp, cl_float* fp, cl_int z_id) {
  int err = 0;
  cl_kernel k = bmap.map[15].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[15].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel updatePara does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, dp, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, fp, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(z_id), &z_id, &kargs);
  gcl_log_cl_fatal(err, "setting argument for updatePara failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing updatePara failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^evaluateUpdateCandidate_kernel)(const cl_ndrange *ndrange, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* lambda_img, cl_float* nyu_img, cl_float* dF2_old_img, cl_float* dF2_new_img, cl_float* dL_img, cl_float* rho_img) =
^(const cl_ndrange *ndrange, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* lambda_img, cl_float* nyu_img, cl_float* dF2_old_img, cl_float* dF2_new_img, cl_float* dL_img, cl_float* rho_img) {
  int err = 0;
  cl_kernel k = bmap.map[16].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[16].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel evaluateUpdateCandidate does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, tJdF_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, tJJ_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, lambda_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, nyu_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, dF2_old_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, dF2_new_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 6, dL_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 7, rho_img, &kargs);
  gcl_log_cl_fatal(err, "setting argument for evaluateUpdateCandidate failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing evaluateUpdateCandidate failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^updateOrRestore_kernel)(const cl_ndrange *ndrange, cl_float* para_img, cl_float* para_img_backup, cl_float* rho_img) =
^(const cl_ndrange *ndrange, cl_float* para_img, cl_float* para_img_backup, cl_float* rho_img) {
  int err = 0;
  cl_kernel k = bmap.map[17].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[17].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel updateOrRestore does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, para_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, para_img_backup, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, rho_img, &kargs);
  gcl_log_cl_fatal(err, "setting argument for updateOrRestore failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing updateOrRestore failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^outputBondDistance_kernel)(const cl_ndrange *ndrange, cl_float* dR, cl_float* R, cl_float Reff) =
^(const cl_ndrange *ndrange, cl_float* dR, cl_float* R, cl_float Reff) {
  int err = 0;
  cl_kernel k = bmap.map[18].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[18].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel outputBondDistance does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, dR, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, R, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(Reff), &Reff, &kargs);
  gcl_log_cl_fatal(err, "setting argument for outputBondDistance failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing outputBondDistance failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/EXAFS_fit.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == hanningWindowFuncIMGarray_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "hanningWindowFuncIMGarray", &err);
          assert(bmap.map[1].block_ptr == redimension_feffShellPara_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "redimension_feffShellPara", &err);
          assert(bmap.map[2].block_ptr == outputchi_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "outputchi", &err);
          assert(bmap.map[3].block_ptr == Jacobian_k_new_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "Jacobian_k_new", &err);
          assert(bmap.map[4].block_ptr == Jacobian_k_kernel && "mismatch block");
          bmap.map[4].kernel = clCreateKernel(bmap.program, "Jacobian_k", &err);
          assert(bmap.map[5].block_ptr == estimate_tJJ_kernel && "mismatch block");
          bmap.map[5].kernel = clCreateKernel(bmap.program, "estimate_tJJ", &err);
          assert(bmap.map[6].block_ptr == estimate_tJdF_kernel && "mismatch block");
          bmap.map[6].kernel = clCreateKernel(bmap.program, "estimate_tJdF", &err);
          assert(bmap.map[7].block_ptr == estimate_dF2_kernel && "mismatch block");
          bmap.map[7].kernel = clCreateKernel(bmap.program, "estimate_dF2", &err);
          assert(bmap.map[8].block_ptr == estimate_Rfactor_kernel && "mismatch block");
          bmap.map[8].kernel = clCreateKernel(bmap.program, "estimate_Rfactor", &err);
          assert(bmap.map[9].block_ptr == estimate_error_kernel && "mismatch block");
          bmap.map[9].kernel = clCreateKernel(bmap.program, "estimate_error", &err);
          assert(bmap.map[10].block_ptr == chi2cmplxChi_imgStck_kernel && "mismatch block");
          bmap.map[10].kernel = clCreateKernel(bmap.program, "chi2cmplxChi_imgStck", &err);
          assert(bmap.map[11].block_ptr == chi2cmplxChi_chiStck_kernel && "mismatch block");
          bmap.map[11].kernel = clCreateKernel(bmap.program, "chi2cmplxChi_chiStck", &err);
          assert(bmap.map[12].block_ptr == CNweighten_kernel && "mismatch block");
          bmap.map[12].kernel = clCreateKernel(bmap.program, "CNweighten", &err);
          assert(bmap.map[13].block_ptr == LevenbergMarquardt_kernel && "mismatch block");
          bmap.map[13].kernel = clCreateKernel(bmap.program, "LevenbergMarquardt", &err);
          assert(bmap.map[14].block_ptr == estimate_dL_kernel && "mismatch block");
          bmap.map[14].kernel = clCreateKernel(bmap.program, "estimate_dL", &err);
          assert(bmap.map[15].block_ptr == updatePara_kernel && "mismatch block");
          bmap.map[15].kernel = clCreateKernel(bmap.program, "updatePara", &err);
          assert(bmap.map[16].block_ptr == evaluateUpdateCandidate_kernel && "mismatch block");
          bmap.map[16].kernel = clCreateKernel(bmap.program, "evaluateUpdateCandidate", &err);
          assert(bmap.map[17].block_ptr == updateOrRestore_kernel && "mismatch block");
          bmap.map[17].kernel = clCreateKernel(bmap.program, "updateOrRestore", &err);
          assert(bmap.map[18].block_ptr == outputBondDistance_kernel && "mismatch block");
          bmap.map[18].kernel = clCreateKernel(bmap.program, "outputBondDistance", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = hanningWindowFuncIMGarray_kernel;
  bmap.map[1].block_ptr = redimension_feffShellPara_kernel;
  bmap.map[2].block_ptr = outputchi_kernel;
  bmap.map[3].block_ptr = Jacobian_k_new_kernel;
  bmap.map[4].block_ptr = Jacobian_k_kernel;
  bmap.map[5].block_ptr = estimate_tJJ_kernel;
  bmap.map[6].block_ptr = estimate_tJdF_kernel;
  bmap.map[7].block_ptr = estimate_dF2_kernel;
  bmap.map[8].block_ptr = estimate_Rfactor_kernel;
  bmap.map[9].block_ptr = estimate_error_kernel;
  bmap.map[10].block_ptr = chi2cmplxChi_imgStck_kernel;
  bmap.map[11].block_ptr = chi2cmplxChi_chiStck_kernel;
  bmap.map[12].block_ptr = CNweighten_kernel;
  bmap.map[13].block_ptr = LevenbergMarquardt_kernel;
  bmap.map[14].block_ptr = estimate_dL_kernel;
  bmap.map[15].block_ptr = updatePara_kernel;
  bmap.map[16].block_ptr = evaluateUpdateCandidate_kernel;
  bmap.map[17].block_ptr = updateOrRestore_kernel;
  bmap.map[18].block_ptr = outputBondDistance_kernel;
}

