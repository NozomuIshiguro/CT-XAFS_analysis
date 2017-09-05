/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "XANES_fit.cl.h"

static void initBlocks(void);

// Initialize static data structures
static block_kernel_pair pair_map[10] = {
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

static block_kernel_map bmap = { 0, 10, initBlocks, pair_map };

// Block function
void (^chi2Stack_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* fp_img, cl_image refSpectra, cl_float* chi2_img, cl_float* energy, cl_int* funcModeList, cl_int startEnum, cl_int endEnum) =
^(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* fp_img, cl_image refSpectra, cl_float* chi2_img, cl_float* energy, cl_int* funcModeList, cl_int startEnum, cl_int endEnum) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel chi2Stack does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, fp_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, refSpectra, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, chi2_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, energy, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, funcModeList, &kargs);
  err |= gclSetKernelArgAPPLE(k, 6, sizeof(startEnum), &startEnum, &kargs);
  err |= gclSetKernelArgAPPLE(k, 7, sizeof(endEnum), &endEnum, &kargs);
  gcl_log_cl_fatal(err, "setting argument for chi2Stack failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing chi2Stack failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^chi2_tJdF_tJJ_Stack_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* fp_img, cl_image refSpectra, cl_float* chi2_img, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* energy, cl_int* funcModeList, cl_char* p_fix, cl_int startEnum, cl_int endEnum) =
^(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* fp_img, cl_image refSpectra, cl_float* chi2_img, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* energy, cl_int* funcModeList, cl_char* p_fix, cl_int startEnum, cl_int endEnum) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel chi2_tJdF_tJJ_Stack does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, fp_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, refSpectra, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, chi2_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, tJdF_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, tJJ_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 6, energy, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 7, funcModeList, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 8, p_fix, &kargs);
  err |= gclSetKernelArgAPPLE(k, 9, sizeof(startEnum), &startEnum, &kargs);
  err |= gclSetKernelArgAPPLE(k, 10, sizeof(endEnum), &endEnum, &kargs);
  gcl_log_cl_fatal(err, "setting argument for chi2_tJdF_tJJ_Stack failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing chi2_tJdF_tJJ_Stack failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^constrain_kernel)(const cl_ndrange *ndrange, cl_float* fp_cnd_img, cl_float* C_mat, cl_float* D_vec) =
^(const cl_ndrange *ndrange, cl_float* fp_cnd_img, cl_float* C_mat, cl_float* D_vec) {
  int err = 0;
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
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

void (^setMask_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_char* mask_img) =
^(const cl_ndrange *ndrange, cl_float* mt_img, cl_char* mask_img) {
  int err = 0;
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
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

void (^applyMask_kernel)(const cl_ndrange *ndrange, cl_float* fp_img, cl_char* mask) =
^(const cl_ndrange *ndrange, cl_float* fp_img, cl_char* mask) {
  int err = 0;
  cl_kernel k = bmap.map[4].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[4].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel applyMask does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, fp_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mask, &kargs);
  gcl_log_cl_fatal(err, "setting argument for applyMask failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing applyMask failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^partialDerivativeOfGradiant_kernel)(const cl_ndrange *ndrange, cl_float* original_img_src, cl_float* img_src, cl_float* img_dest, cl_float* epsilon_g, cl_float* alpha_g) =
^(const cl_ndrange *ndrange, cl_float* original_img_src, cl_float* img_src, cl_float* img_dest, cl_float* epsilon_g, cl_float* alpha_g) {
  int err = 0;
  cl_kernel k = bmap.map[5].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[5].kernel;
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

void (^redimension_refSpecta_kernel)(const cl_ndrange *ndrange, cl_image refSpectra, cl_image refSpectrum_raw, cl_float* energy, cl_int numE, cl_int offset) =
^(const cl_ndrange *ndrange, cl_image refSpectra, cl_image refSpectrum_raw, cl_float* energy, cl_int numE, cl_int offset) {
  int err = 0;
  cl_kernel k = bmap.map[6].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[6].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel redimension_refSpecta does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, refSpectra, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, refSpectrum_raw, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, energy, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(numE), &numE, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(offset), &offset, &kargs);
  gcl_log_cl_fatal(err, "setting argument for redimension_refSpecta failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing redimension_refSpecta failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^SoftThresholdingFunc_kernel)(const cl_ndrange *ndrange, cl_float* fp_img, cl_float* dp_img, cl_float* dp_cnd_img, cl_float* inv_tJJ_img, cl_char* p_fix, cl_float* lambda_fista) =
^(const cl_ndrange *ndrange, cl_float* fp_img, cl_float* dp_img, cl_float* dp_cnd_img, cl_float* inv_tJJ_img, cl_char* p_fix, cl_float* lambda_fista) {
  int err = 0;
  cl_kernel k = bmap.map[7].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[7].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel SoftThresholdingFunc does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, fp_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, dp_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, dp_cnd_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, inv_tJJ_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, p_fix, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, lambda_fista, &kargs);
  gcl_log_cl_fatal(err, "setting argument for SoftThresholdingFunc failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing SoftThresholdingFunc failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^FISTAupdate_kernel)(const cl_ndrange *ndrange, cl_float* fp_new_img, cl_float* fp_old_img, cl_float* beta_img, cl_float* w_img) =
^(const cl_ndrange *ndrange, cl_float* fp_new_img, cl_float* fp_old_img, cl_float* beta_img, cl_float* w_img) {
  int err = 0;
  cl_kernel k = bmap.map[8].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[8].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel FISTAupdate does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, fp_new_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, fp_old_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, beta_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, w_img, &kargs);
  gcl_log_cl_fatal(err, "setting argument for FISTAupdate failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing FISTAupdate failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^powerIteration_kernel)(const cl_ndrange *ndrange, cl_float* A_img, cl_float* maxEval_img, cl_int iter) =
^(const cl_ndrange *ndrange, cl_float* A_img, cl_float* maxEval_img, cl_int iter) {
  int err = 0;
  cl_kernel k = bmap.map[9].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[9].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel powerIteration does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, A_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, maxEval_img, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(iter), &iter, &kargs);
  gcl_log_cl_fatal(err, "setting argument for powerIteration failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing powerIteration failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/XANES_fit.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == chi2Stack_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "chi2Stack", &err);
          assert(bmap.map[1].block_ptr == chi2_tJdF_tJJ_Stack_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "chi2_tJdF_tJJ_Stack", &err);
          assert(bmap.map[2].block_ptr == constrain_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "constrain", &err);
          assert(bmap.map[3].block_ptr == setMask_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "setMask", &err);
          assert(bmap.map[4].block_ptr == applyMask_kernel && "mismatch block");
          bmap.map[4].kernel = clCreateKernel(bmap.program, "applyMask", &err);
          assert(bmap.map[5].block_ptr == partialDerivativeOfGradiant_kernel && "mismatch block");
          bmap.map[5].kernel = clCreateKernel(bmap.program, "partialDerivativeOfGradiant", &err);
          assert(bmap.map[6].block_ptr == redimension_refSpecta_kernel && "mismatch block");
          bmap.map[6].kernel = clCreateKernel(bmap.program, "redimension_refSpecta", &err);
          assert(bmap.map[7].block_ptr == SoftThresholdingFunc_kernel && "mismatch block");
          bmap.map[7].kernel = clCreateKernel(bmap.program, "SoftThresholdingFunc", &err);
          assert(bmap.map[8].block_ptr == FISTAupdate_kernel && "mismatch block");
          bmap.map[8].kernel = clCreateKernel(bmap.program, "FISTAupdate", &err);
          assert(bmap.map[9].block_ptr == powerIteration_kernel && "mismatch block");
          bmap.map[9].kernel = clCreateKernel(bmap.program, "powerIteration", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = chi2Stack_kernel;
  bmap.map[1].block_ptr = chi2_tJdF_tJJ_Stack_kernel;
  bmap.map[2].block_ptr = constrain_kernel;
  bmap.map[3].block_ptr = setMask_kernel;
  bmap.map[4].block_ptr = applyMask_kernel;
  bmap.map[5].block_ptr = partialDerivativeOfGradiant_kernel;
  bmap.map[6].block_ptr = redimension_refSpecta_kernel;
  bmap.map[7].block_ptr = SoftThresholdingFunc_kernel;
  bmap.map[8].block_ptr = FISTAupdate_kernel;
  bmap.map[9].block_ptr = powerIteration_kernel;
}

