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
static block_kernel_pair pair_map[11] = {
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

static block_kernel_map bmap = { 0, 11, initBlocks, pair_map };

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

void (^updateOrRestore_kernel)(const cl_ndrange *ndrange, cl_float* para_img, cl_float* para_img_backup, cl_float* rho_img, cl_int z_id1, cl_int z_id2) =
^(const cl_ndrange *ndrange, cl_float* para_img, cl_float* para_img_backup, cl_float* rho_img, cl_int z_id1, cl_int z_id2) {
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
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(z_id1), &z_id1, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(z_id2), &z_id2, &kargs);
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

void (^ISTA_kernel)(const cl_ndrange *ndrange, cl_float* fp_img, cl_float* tJJ_img, cl_char* p_fix, cl_float* lambda_LM, cl_float* lambda_fista) =
^(const cl_ndrange *ndrange, cl_float* fp_img, cl_float* tJJ_img, cl_char* p_fix, cl_float* lambda_LM, cl_float* lambda_fista) {
  int err = 0;
  cl_kernel k = bmap.map[6].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[6].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel ISTA does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, fp_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, tJJ_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, p_fix, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, lambda_LM, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, lambda_fista, &kargs);
  gcl_log_cl_fatal(err, "setting argument for ISTA failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing ISTA failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^FISTA_kernel)(const cl_ndrange *ndrange, cl_float* fp_x_img, cl_float* fp_w_img, cl_float* beta_img, cl_float* tJJ_img, cl_char* p_fix, cl_float* lambda_LM, cl_float* lambda_fista) =
^(const cl_ndrange *ndrange, cl_float* fp_x_img, cl_float* fp_w_img, cl_float* beta_img, cl_float* tJJ_img, cl_char* p_fix, cl_float* lambda_LM, cl_float* lambda_fista) {
  int err = 0;
  cl_kernel k = bmap.map[7].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[7].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel FISTA does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, fp_x_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, fp_w_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, beta_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, tJJ_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, p_fix, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, lambda_LM, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 6, lambda_fista, &kargs);
  gcl_log_cl_fatal(err, "setting argument for FISTA failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing FISTA failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^contrain_0_kernel)(const cl_ndrange *ndrange, cl_float* C_mat, cl_float* C2_vec) =
^(const cl_ndrange *ndrange, cl_float* C_mat, cl_float* C2_vec) {
  int err = 0;
  cl_kernel k = bmap.map[8].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[8].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel contrain_0 does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, C_mat, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, C2_vec, &kargs);
  gcl_log_cl_fatal(err, "setting argument for contrain_0 failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing contrain_0 failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^contrain_1_kernel)(const cl_ndrange *ndrange, cl_float* fp_cnd_img, cl_float* eval_img, cl_float* C_mat, cl_int c_num, cl_int p_num) =
^(const cl_ndrange *ndrange, cl_float* fp_cnd_img, cl_float* eval_img, cl_float* C_mat, cl_int c_num, cl_int p_num) {
  int err = 0;
  cl_kernel k = bmap.map[9].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[9].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel contrain_1 does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, fp_cnd_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, eval_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, C_mat, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(c_num), &c_num, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(p_num), &p_num, &kargs);
  gcl_log_cl_fatal(err, "setting argument for contrain_1 failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing contrain_1 failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^contrain_2_kernel)(const cl_ndrange *ndrange, cl_float* fp_cnd_img, cl_float* weight_img, cl_float* eval_img, cl_float* C_mat, cl_float* D_vec, cl_float* C2_vec, cl_int c_num, cl_int p_num, cl_char weight_b) =
^(const cl_ndrange *ndrange, cl_float* fp_cnd_img, cl_float* weight_img, cl_float* eval_img, cl_float* C_mat, cl_float* D_vec, cl_float* C2_vec, cl_int c_num, cl_int p_num, cl_char weight_b) {
  int err = 0;
  cl_kernel k = bmap.map[10].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[10].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel contrain_2 does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, fp_cnd_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, weight_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, eval_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, C_mat, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, D_vec, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, C2_vec, &kargs);
  err |= gclSetKernelArgAPPLE(k, 6, sizeof(c_num), &c_num, &kargs);
  err |= gclSetKernelArgAPPLE(k, 7, sizeof(p_num), &p_num, &kargs);
  err |= gclSetKernelArgAPPLE(k, 8, sizeof(weight_b), &weight_b, &kargs);
  gcl_log_cl_fatal(err, "setting argument for contrain_2 failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing contrain_2 failed");
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
          assert(bmap.map[6].block_ptr == ISTA_kernel && "mismatch block");
          bmap.map[6].kernel = clCreateKernel(bmap.program, "ISTA", &err);
          assert(bmap.map[7].block_ptr == FISTA_kernel && "mismatch block");
          bmap.map[7].kernel = clCreateKernel(bmap.program, "FISTA", &err);
          assert(bmap.map[8].block_ptr == contrain_0_kernel && "mismatch block");
          bmap.map[8].kernel = clCreateKernel(bmap.program, "contrain_0", &err);
          assert(bmap.map[9].block_ptr == contrain_1_kernel && "mismatch block");
          bmap.map[9].kernel = clCreateKernel(bmap.program, "contrain_1", &err);
          assert(bmap.map[10].block_ptr == contrain_2_kernel && "mismatch block");
          bmap.map[10].kernel = clCreateKernel(bmap.program, "contrain_2", &err);
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
  bmap.map[6].block_ptr = ISTA_kernel;
  bmap.map[7].block_ptr = FISTA_kernel;
  bmap.map[8].block_ptr = contrain_0_kernel;
  bmap.map[9].block_ptr = contrain_1_kernel;
  bmap.map[10].block_ptr = contrain_2_kernel;
}

