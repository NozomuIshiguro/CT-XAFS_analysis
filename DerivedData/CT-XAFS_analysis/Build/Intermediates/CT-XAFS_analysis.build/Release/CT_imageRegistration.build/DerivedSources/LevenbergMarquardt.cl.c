/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "LevenbergMarquardt.cl.h"

static void initBlocks(void);

// Initialize static data structures
static block_kernel_pair pair_map[6] = {
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL }
};

static block_kernel_map bmap = { 0, 6, initBlocks, pair_map };

// Block function
void (^LevenbergMarquardt_kernel)(const cl_ndrange *ndrange, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* dp_img, cl_float* lambda_img, cl_char* p_fix) =
^(const cl_ndrange *ndrange, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* dp_img, cl_float* lambda_img, cl_char* p_fix) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel LevenbergMarquardt does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, tJdF_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, tJJ_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, dp_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, lambda_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, p_fix, &kargs);
  gcl_log_cl_fatal(err, "setting argument for LevenbergMarquardt failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing LevenbergMarquardt failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^estimate_dL_kernel)(const cl_ndrange *ndrange, cl_float* dp_img, cl_float* tJJ_img, cl_float* tJdF_img, cl_float* lambda_img, cl_float* dL_img) =
^(const cl_ndrange *ndrange, cl_float* dp_img, cl_float* tJJ_img, cl_float* tJdF_img, cl_float* lambda_img, cl_float* dL_img) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
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

void (^updatePara_kernel)(const cl_ndrange *ndrange, cl_float* dp, cl_float* fp, cl_int z_id1, cl_int z_id2) =
^(const cl_ndrange *ndrange, cl_float* dp, cl_float* fp, cl_int z_id1, cl_int z_id2) {
  int err = 0;
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel updatePara does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, dp, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, fp, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(z_id1), &z_id1, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(z_id2), &z_id2, &kargs);
  gcl_log_cl_fatal(err, "setting argument for updatePara failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing updatePara failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^evaluateUpdateCandidate_kernel)(const cl_ndrange *ndrange, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* lambda_img, cl_float* nyu_img, cl_float* dF2_old_img, cl_float* dF2_new_img, cl_float* dL_img, cl_float* rho_img) =
^(const cl_ndrange *ndrange, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* lambda_img, cl_float* nyu_img, cl_float* dF2_old_img, cl_float* dF2_new_img, cl_float* dL_img, cl_float* rho_img) {
  int err = 0;
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
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
  cl_kernel k = bmap.map[4].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[4].kernel;
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

void (^updateOrHold_kernel)(const cl_ndrange *ndrange, cl_float* para_img, cl_float* para_img_cnd, cl_float* rho_img) =
^(const cl_ndrange *ndrange, cl_float* para_img, cl_float* para_img_cnd, cl_float* rho_img) {
  int err = 0;
  cl_kernel k = bmap.map[5].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[5].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel updateOrHold does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, para_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, para_img_cnd, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, rho_img, &kargs);
  gcl_log_cl_fatal(err, "setting argument for updateOrHold failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing updateOrHold failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/LevenbergMarquardt.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == LevenbergMarquardt_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "LevenbergMarquardt", &err);
          assert(bmap.map[1].block_ptr == estimate_dL_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "estimate_dL", &err);
          assert(bmap.map[2].block_ptr == updatePara_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "updatePara", &err);
          assert(bmap.map[3].block_ptr == evaluateUpdateCandidate_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "evaluateUpdateCandidate", &err);
          assert(bmap.map[4].block_ptr == updateOrRestore_kernel && "mismatch block");
          bmap.map[4].kernel = clCreateKernel(bmap.program, "updateOrRestore", &err);
          assert(bmap.map[5].block_ptr == updateOrHold_kernel && "mismatch block");
          bmap.map[5].kernel = clCreateKernel(bmap.program, "updateOrHold", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = LevenbergMarquardt_kernel;
  bmap.map[1].block_ptr = estimate_dL_kernel;
  bmap.map[2].block_ptr = updatePara_kernel;
  bmap.map[3].block_ptr = evaluateUpdateCandidate_kernel;
  bmap.map[4].block_ptr = updateOrRestore_kernel;
  bmap.map[5].block_ptr = updateOrHold_kernel;
}

