/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "XANES_fitting2.cl.h"

static void initBlocks(void);

// Initialize static data structures
static block_kernel_pair pair_map[12] = {
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

static block_kernel_map bmap = { 0, 12, initBlocks, pair_map };

// Block function
void (^mtFit_kernel)(const cl_ndrange *ndrange, cl_float* mt_fit_img, cl_float* fp_img, cl_float energy, cl_int fpOffset, cl_int funcMode, cl_image refSpectrum) =
^(const cl_ndrange *ndrange, cl_float* mt_fit_img, cl_float* fp_img, cl_float energy, cl_int fpOffset, cl_int funcMode, cl_image refSpectrum) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel mtFit does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_fit_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, fp_img, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(energy), &energy, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(fpOffset), &fpOffset, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(funcMode), &funcMode, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, refSpectrum, &kargs);
  gcl_log_cl_fatal(err, "setting argument for mtFit failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing mtFit failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^Jacobian_kernel)(const cl_ndrange *ndrange, cl_float* J_img, cl_float* fp_img, cl_float energy, cl_int fpOffset, cl_int funcMode, cl_image refSpectrum) =
^(const cl_ndrange *ndrange, cl_float* J_img, cl_float* fp_img, cl_float energy, cl_int fpOffset, cl_int funcMode, cl_image refSpectrum) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel Jacobian does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, J_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, fp_img, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(energy), &energy, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(fpOffset), &fpOffset, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(funcMode), &funcMode, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, refSpectrum, &kargs);
  gcl_log_cl_fatal(err, "setting argument for Jacobian failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing Jacobian failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^chi2_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* mt_fit_img, cl_float* chi2_img, cl_int en) =
^(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* mt_fit_img, cl_float* chi2_img, cl_int en) {
  int err = 0;
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel chi2 does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_fit_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, chi2_img, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(en), &en, &kargs);
  gcl_log_cl_fatal(err, "setting argument for chi2 failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing chi2 failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^tJdF_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* mt_fit_img, cl_float* J_img, cl_float* tJdF_img, cl_int en) =
^(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* mt_fit_img, cl_float* J_img, cl_float* tJdF_img, cl_int en) {
  int err = 0;
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel tJdF does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_fit_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, J_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, tJdF_img, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(en), &en, &kargs);
  gcl_log_cl_fatal(err, "setting argument for tJdF failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing tJdF failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^tJJ_kernel)(const cl_ndrange *ndrange, cl_float* J_img, cl_float* tJJ_img) =
^(const cl_ndrange *ndrange, cl_float* J_img, cl_float* tJJ_img) {
  int err = 0;
  cl_kernel k = bmap.map[4].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[4].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel tJJ does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, J_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, tJJ_img, &kargs);
  gcl_log_cl_fatal(err, "setting argument for tJJ failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing tJJ failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^LevenbergMarquardt_kernel)(const cl_ndrange *ndrange, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* fp_img, cl_float* fp_cnd_img, cl_float* lambda_img, cl_char* p_fix) =
^(const cl_ndrange *ndrange, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* fp_img, cl_float* fp_cnd_img, cl_float* lambda_img, cl_char* p_fix) {
  int err = 0;
  cl_kernel k = bmap.map[5].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[5].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel LevenbergMarquardt does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, tJdF_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, tJJ_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, fp_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, fp_cnd_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, lambda_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, p_fix, &kargs);
  gcl_log_cl_fatal(err, "setting argument for LevenbergMarquardt failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing LevenbergMarquardt failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^constrain_kernel)(const cl_ndrange *ndrange, cl_float* fp_cnd_img, cl_float* C_mat, cl_float* D_vec) =
^(const cl_ndrange *ndrange, cl_float* fp_cnd_img, cl_float* C_mat, cl_float* D_vec) {
  int err = 0;
  cl_kernel k = bmap.map[6].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[6].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel constrain does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, fp_cnd_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, C_mat, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, D_vec, &kargs);
  gcl_log_cl_fatal(err, "setting argument for constrain failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing constrain failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^UpdateFp_kernel)(const cl_ndrange *ndrange, cl_float* fp_img, cl_float* fp_cnd_img, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* lambda_img, cl_float* nyu_img, cl_float* chi2_old_img, cl_float* chi2_new_img) =
^(const cl_ndrange *ndrange, cl_float* fp_img, cl_float* fp_cnd_img, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* lambda_img, cl_float* nyu_img, cl_float* chi2_old_img, cl_float* chi2_new_img) {
  int err = 0;
  cl_kernel k = bmap.map[7].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[7].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel UpdateFp does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, fp_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, fp_cnd_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, tJdF_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, tJJ_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, lambda_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, nyu_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 6, chi2_old_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 7, chi2_new_img, &kargs);
  gcl_log_cl_fatal(err, "setting argument for UpdateFp failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing UpdateFp failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^setMask_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_char* mask_img) =
^(const cl_ndrange *ndrange, cl_float* mt_img, cl_char* mask_img) {
  int err = 0;
  cl_kernel k = bmap.map[8].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[8].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel setMask does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mask_img, &kargs);
  gcl_log_cl_fatal(err, "setting argument for setMask failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing setMask failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^applyThreshold_kernel)(const cl_ndrange *ndrange, cl_float* fit_results_img, cl_char* mask) =
^(const cl_ndrange *ndrange, cl_float* fit_results_img, cl_char* mask) {
  int err = 0;
  cl_kernel k = bmap.map[9].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[9].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel applyThreshold does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, fit_results_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mask, &kargs);
  gcl_log_cl_fatal(err, "setting argument for applyThreshold failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing applyThreshold failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^partialDerivativeOfGradiant_kernel)(const cl_ndrange *ndrange, cl_float* original_img_src, cl_float* img_src, cl_float* img_dest, cl_float* epsilon_g, cl_float* alpha_g) =
^(const cl_ndrange *ndrange, cl_float* original_img_src, cl_float* img_src, cl_float* img_dest, cl_float* epsilon_g, cl_float* alpha_g) {
  int err = 0;
  cl_kernel k = bmap.map[10].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[10].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel partialDerivativeOfGradiant does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, original_img_src, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, img_src, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, img_dest, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, epsilon_g, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, alpha_g, &kargs);
  gcl_log_cl_fatal(err, "setting argument for partialDerivativeOfGradiant failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing partialDerivativeOfGradiant failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^redimension_refSpecta_kernel)(const cl_ndrange *ndrange, cl_image refSpectra, cl_image refSpectra_raw, cl_float* energy, cl_int numE) =
^(const cl_ndrange *ndrange, cl_image refSpectra, cl_image refSpectra_raw, cl_float* energy, cl_int numE) {
  int err = 0;
  cl_kernel k = bmap.map[11].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[11].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel redimension_refSpecta does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, refSpectra, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, refSpectra_raw, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, energy, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(numE), &numE, &kargs);
  gcl_log_cl_fatal(err, "setting argument for redimension_refSpecta failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing redimension_refSpecta failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/XANES_fitting2.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == mtFit_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "mtFit", &err);
          assert(bmap.map[1].block_ptr == Jacobian_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "Jacobian", &err);
          assert(bmap.map[2].block_ptr == chi2_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "chi2", &err);
          assert(bmap.map[3].block_ptr == tJdF_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "tJdF", &err);
          assert(bmap.map[4].block_ptr == tJJ_kernel && "mismatch block");
          bmap.map[4].kernel = clCreateKernel(bmap.program, "tJJ", &err);
          assert(bmap.map[5].block_ptr == LevenbergMarquardt_kernel && "mismatch block");
          bmap.map[5].kernel = clCreateKernel(bmap.program, "LevenbergMarquardt", &err);
          assert(bmap.map[6].block_ptr == constrain_kernel && "mismatch block");
          bmap.map[6].kernel = clCreateKernel(bmap.program, "constrain", &err);
          assert(bmap.map[7].block_ptr == UpdateFp_kernel && "mismatch block");
          bmap.map[7].kernel = clCreateKernel(bmap.program, "UpdateFp", &err);
          assert(bmap.map[8].block_ptr == setMask_kernel && "mismatch block");
          bmap.map[8].kernel = clCreateKernel(bmap.program, "setMask", &err);
          assert(bmap.map[9].block_ptr == applyThreshold_kernel && "mismatch block");
          bmap.map[9].kernel = clCreateKernel(bmap.program, "applyThreshold", &err);
          assert(bmap.map[10].block_ptr == partialDerivativeOfGradiant_kernel && "mismatch block");
          bmap.map[10].kernel = clCreateKernel(bmap.program, "partialDerivativeOfGradiant", &err);
          assert(bmap.map[11].block_ptr == redimension_refSpecta_kernel && "mismatch block");
          bmap.map[11].kernel = clCreateKernel(bmap.program, "redimension_refSpecta", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = mtFit_kernel;
  bmap.map[1].block_ptr = Jacobian_kernel;
  bmap.map[2].block_ptr = chi2_kernel;
  bmap.map[3].block_ptr = tJdF_kernel;
  bmap.map[4].block_ptr = tJJ_kernel;
  bmap.map[5].block_ptr = LevenbergMarquardt_kernel;
  bmap.map[6].block_ptr = constrain_kernel;
  bmap.map[7].block_ptr = UpdateFp_kernel;
  bmap.map[8].block_ptr = setMask_kernel;
  bmap.map[9].block_ptr = applyThreshold_kernel;
  bmap.map[10].block_ptr = partialDerivativeOfGradiant_kernel;
  bmap.map[11].block_ptr = redimension_refSpecta_kernel;
}

