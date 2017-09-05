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
static block_kernel_pair pair_map[5] = {
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL }
};

static block_kernel_map bmap = { 0, 5, initBlocks, pair_map };

// Block function
void (^chi2Stack_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* fp_img, cl_image refSpectra, cl_float* chi2_img, cl_float* energy, cl_int* funcModeList, cl_int startEnum, cl_int endEnum, cl_float* weight_img, cl_float* weight_thd_img, cl_float CI) =
^(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* fp_img, cl_image refSpectra, cl_float* chi2_img, cl_float* energy, cl_int* funcModeList, cl_int startEnum, cl_int endEnum, cl_float* weight_img, cl_float* weight_thd_img, cl_float CI) {
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
  err |= gclSetKernelArgMemAPPLE(k, 8, weight_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 9, weight_thd_img, &kargs);
  err |= gclSetKernelArgAPPLE(k, 10, sizeof(CI), &CI, &kargs);
  gcl_log_cl_fatal(err, "setting argument for chi2Stack failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing chi2Stack failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^chi2_tJdF_tJJ_Stack_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* fp_img, cl_image refSpectra, cl_float* chi2_img, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* energy, cl_int* funcModeList, cl_char* p_fix, cl_int startEnum, cl_int endEnum, cl_float* weight_img, cl_float* weight_thd_img) =
^(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* fp_img, cl_image refSpectra, cl_float* chi2_img, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* energy, cl_int* funcModeList, cl_char* p_fix, cl_int startEnum, cl_int endEnum, cl_float* weight_img, cl_float* weight_thd_img) {
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
  err |= gclSetKernelArgMemAPPLE(k, 11, weight_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 12, weight_thd_img, &kargs);
  gcl_log_cl_fatal(err, "setting argument for chi2_tJdF_tJJ_Stack failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing chi2_tJdF_tJJ_Stack failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^setMask_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_char* mask_img) =
^(const cl_ndrange *ndrange, cl_float* mt_img, cl_char* mask_img) {
  int err = 0;
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
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
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
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

void (^redimension_refSpecta_kernel)(const cl_ndrange *ndrange, cl_image refSpectra, cl_image refSpectrum_raw, cl_float* energy, cl_int numE, cl_int offset) =
^(const cl_ndrange *ndrange, cl_image refSpectra, cl_image refSpectrum_raw, cl_float* energy, cl_int numE, cl_int offset) {
  int err = 0;
  cl_kernel k = bmap.map[4].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[4].kernel;
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
          assert(bmap.map[2].block_ptr == setMask_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "setMask", &err);
          assert(bmap.map[3].block_ptr == applyMask_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "applyMask", &err);
          assert(bmap.map[4].block_ptr == redimension_refSpecta_kernel && "mismatch block");
          bmap.map[4].kernel = clCreateKernel(bmap.program, "redimension_refSpecta", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = chi2Stack_kernel;
  bmap.map[1].block_ptr = chi2_tJdF_tJJ_Stack_kernel;
  bmap.map[2].block_ptr = setMask_kernel;
  bmap.map[3].block_ptr = applyMask_kernel;
  bmap.map[4].block_ptr = redimension_refSpecta_kernel;
}

