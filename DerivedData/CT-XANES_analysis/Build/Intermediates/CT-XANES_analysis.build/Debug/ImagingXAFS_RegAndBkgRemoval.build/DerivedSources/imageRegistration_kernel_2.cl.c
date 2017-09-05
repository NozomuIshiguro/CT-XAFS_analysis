/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "imageRegistration_kernel_2.cl.h"

static void initBlocks(void);

// Initialize static data structures
static block_kernel_pair pair_map[14] = {
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

static block_kernel_map bmap = { 0, 14, initBlocks, pair_map };

// Block function
void (^mt_conversion_kernel)(const cl_ndrange *ndrange, cl_float* dark, cl_float* I0, cl_ushort* It_buffer, cl_float* mt_buffer, cl_image mt_img1, cl_image mt_img2, cl_int shapeNo, cl_int startpntX, cl_int startpntY, cl_uint width, cl_uint height, cl_float angle, cl_int evaluatemode) =
^(const cl_ndrange *ndrange, cl_float* dark, cl_float* I0, cl_ushort* It_buffer, cl_float* mt_buffer, cl_image mt_img1, cl_image mt_img2, cl_int shapeNo, cl_int startpntX, cl_int startpntY, cl_uint width, cl_uint height, cl_float angle, cl_int evaluatemode) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel mt_conversion does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, dark, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, I0, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, It_buffer, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, mt_buffer, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, mt_img1, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, mt_img2, &kargs);
  err |= gclSetKernelArgAPPLE(k, 6, sizeof(shapeNo), &shapeNo, &kargs);
  err |= gclSetKernelArgAPPLE(k, 7, sizeof(startpntX), &startpntX, &kargs);
  err |= gclSetKernelArgAPPLE(k, 8, sizeof(startpntY), &startpntY, &kargs);
  err |= gclSetKernelArgAPPLE(k, 9, sizeof(width), &width, &kargs);
  err |= gclSetKernelArgAPPLE(k, 10, sizeof(height), &height, &kargs);
  err |= gclSetKernelArgAPPLE(k, 11, sizeof(angle), &angle, &kargs);
  err |= gclSetKernelArgAPPLE(k, 12, sizeof(evaluatemode), &evaluatemode, &kargs);
  gcl_log_cl_fatal(err, "setting argument for mt_conversion failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing mt_conversion failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^mt_transfer_kernel)(const cl_ndrange *ndrange, cl_image mt_img_src, cl_image mt_img_dest, cl_int layerN, cl_int shapeNo, cl_int startpntX, cl_int startpntY, cl_uint width, cl_uint height, cl_float angle) =
^(const cl_ndrange *ndrange, cl_image mt_img_src, cl_image mt_img_dest, cl_int layerN, cl_int shapeNo, cl_int startpntX, cl_int startpntY, cl_uint width, cl_uint height, cl_float angle) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel mt_transfer does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img_src, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_img_dest, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(layerN), &layerN, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(shapeNo), &shapeNo, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(startpntX), &startpntX, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(startpntY), &startpntY, &kargs);
  err |= gclSetKernelArgAPPLE(k, 6, sizeof(width), &width, &kargs);
  err |= gclSetKernelArgAPPLE(k, 7, sizeof(height), &height, &kargs);
  err |= gclSetKernelArgAPPLE(k, 8, sizeof(angle), &angle, &kargs);
  gcl_log_cl_fatal(err, "setting argument for mt_transfer failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing mt_transfer failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^imageReg_Jacobian_kernel)(const cl_ndrange *ndrange, cl_image mt_s_img, cl_image mt_t_img, cl_float* Jacobian, cl_float* p, cl_float* p_target, cl_int mergeN) =
^(const cl_ndrange *ndrange, cl_image mt_s_img, cl_image mt_t_img, cl_float* Jacobian, cl_float* p, cl_float* p_target, cl_int mergeN) {
  int err = 0;
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imageReg_Jacobian does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_s_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_t_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, Jacobian, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, p, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, p_target, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(mergeN), &mergeN, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imageReg_Jacobian failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imageReg_Jacobian failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^imageReg_dF_kernel)(const cl_ndrange *ndrange, cl_image mt_t_img, cl_image mt_s_img, cl_float* chi2, cl_float* p, cl_float* p_target, cl_int mergeN) =
^(const cl_ndrange *ndrange, cl_image mt_t_img, cl_image mt_s_img, cl_float* chi2, cl_float* p, cl_float* p_target, cl_int mergeN) {
  int err = 0;
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imageReg_dF does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_t_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_s_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, chi2, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, p, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, p_target, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(mergeN), &mergeN, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imageReg_dF failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imageReg_dF failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^imageReg_dF_EnergyStack_kernel)(const cl_ndrange *ndrange, cl_image mt_t_img, cl_image mt_s_img, cl_float* chi2, cl_float* p, cl_float* p_target, cl_int mergeN) =
^(const cl_ndrange *ndrange, cl_image mt_t_img, cl_image mt_s_img, cl_float* chi2, cl_float* p, cl_float* p_target, cl_int mergeN) {
  int err = 0;
  cl_kernel k = bmap.map[4].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[4].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imageReg_dF_EnergyStack does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_t_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_s_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, chi2, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, p, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, p_target, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(mergeN), &mergeN, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imageReg_dF_EnergyStack failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imageReg_dF_EnergyStack failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^imageReg_tJJ_reduction_kernel)(const cl_ndrange *ndrange, cl_float* Jacobian, cl_float* tJJ, size_t loc_mem, cl_int paraN1, cl_int paraN2, cl_int mergeN) =
^(const cl_ndrange *ndrange, cl_float* Jacobian, cl_float* tJJ, size_t loc_mem, cl_int paraN1, cl_int paraN2, cl_int mergeN) {
  int err = 0;
  cl_kernel k = bmap.map[5].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[5].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imageReg_tJJ_reduction does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, Jacobian, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, tJJ, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, loc_mem, NULL, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(paraN1), &paraN1, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(paraN2), &paraN2, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(mergeN), &mergeN, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imageReg_tJJ_reduction failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imageReg_tJJ_reduction failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^imageReg_tJdF_reduction_kernel)(const cl_ndrange *ndrange, cl_float* Jacobian, cl_float* dF, cl_float* tJdF, size_t loc_mem, cl_int mergeN) =
^(const cl_ndrange *ndrange, cl_float* Jacobian, cl_float* dF, cl_float* tJdF, size_t loc_mem, cl_int mergeN) {
  int err = 0;
  cl_kernel k = bmap.map[6].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[6].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imageReg_tJdF_reduction does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, Jacobian, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, dF, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, tJdF, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, loc_mem, NULL, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(mergeN), &mergeN, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imageReg_tJdF_reduction failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imageReg_tJdF_reduction failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^imageReg_chi2_reduction_kernel)(const cl_ndrange *ndrange, cl_float* dF, cl_float* chi2, size_t loc_mem, cl_int mergeN) =
^(const cl_ndrange *ndrange, cl_float* dF, cl_float* chi2, size_t loc_mem, cl_int mergeN) {
  int err = 0;
  cl_kernel k = bmap.map[7].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[7].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imageReg_chi2_reduction does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, dF, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, chi2, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, loc_mem, NULL, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(mergeN), &mergeN, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imageReg_chi2_reduction failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imageReg_chi2_reduction failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^imageReg_LM_kernel)(const cl_ndrange *ndrange, cl_float* tJJ_g, cl_float* tJdF_g, cl_float* p_g, cl_float* p_cnd, cl_float* p_err, cl_float* p_fix, cl_float* lambda_g) =
^(const cl_ndrange *ndrange, cl_float* tJJ_g, cl_float* tJdF_g, cl_float* p_g, cl_float* p_cnd, cl_float* p_err, cl_float* p_fix, cl_float* lambda_g) {
  int err = 0;
  cl_kernel k = bmap.map[8].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[8].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imageReg_LM does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, tJJ_g, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, tJdF_g, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, p_g, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, p_cnd, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, p_err, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, p_fix, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 6, lambda_g, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imageReg_LM failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imageReg_LM failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^imageReg_Update_kernel)(const cl_ndrange *ndrange, cl_float* tJJ, cl_float* tJdF, cl_float* p_g, cl_float* p_cnd_g, cl_float* p_fix, cl_float* chi2_old, cl_float* chi2_new, cl_float* lambda_g, cl_float* nyu_g) =
^(const cl_ndrange *ndrange, cl_float* tJJ, cl_float* tJdF, cl_float* p_g, cl_float* p_cnd_g, cl_float* p_fix, cl_float* chi2_old, cl_float* chi2_new, cl_float* lambda_g, cl_float* nyu_g) {
  int err = 0;
  cl_kernel k = bmap.map[9].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[9].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imageReg_Update does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, tJJ, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, tJdF, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, p_g, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, p_cnd_g, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, p_fix, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, chi2_old, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 6, chi2_new, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 7, lambda_g, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 8, nyu_g, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imageReg_Update failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imageReg_Update failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^merge_mt_kernel)(const cl_ndrange *ndrange, cl_image mt_sample, cl_float* mt_output) =
^(const cl_ndrange *ndrange, cl_image mt_sample, cl_float* mt_output) {
  int err = 0;
  cl_kernel k = bmap.map[10].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[10].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel merge_mt does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_sample, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_output, &kargs);
  gcl_log_cl_fatal(err, "setting argument for merge_mt failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing merge_mt failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^merge_rawhisdata_kernel)(const cl_ndrange *ndrange, cl_ushort* rawhisdata, cl_float* outputdata, cl_int mergeN) =
^(const cl_ndrange *ndrange, cl_ushort* rawhisdata, cl_float* outputdata, cl_int mergeN) {
  int err = 0;
  cl_kernel k = bmap.map[11].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[11].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel merge_rawhisdata does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, rawhisdata, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, outputdata, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(mergeN), &mergeN, &kargs);
  gcl_log_cl_fatal(err, "setting argument for merge_rawhisdata failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing merge_rawhisdata failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^imQXAFS_smoothing_kernel)(const cl_ndrange *ndrange, cl_float* rawmtdata, cl_float* outputdata, cl_int mergeN) =
^(const cl_ndrange *ndrange, cl_float* rawmtdata, cl_float* outputdata, cl_int mergeN) {
  int err = 0;
  cl_kernel k = bmap.map[12].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[12].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imQXAFS_smoothing does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, rawmtdata, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, outputdata, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(mergeN), &mergeN, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imQXAFS_smoothing failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imQXAFS_smoothing failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^inverse_mt_transfer_kernel)(const cl_ndrange *ndrange, cl_image mt_img_dest, cl_image mt_img_src, cl_int layerN) =
^(const cl_ndrange *ndrange, cl_image mt_img_dest, cl_image mt_img_src, cl_int layerN) {
  int err = 0;
  cl_kernel k = bmap.map[13].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[13].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel inverse_mt_transfer does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img_dest, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_img_src, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(layerN), &layerN, &kargs);
  gcl_log_cl_fatal(err, "setting argument for inverse_mt_transfer failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing inverse_mt_transfer failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/imageRegistration_kernel_2.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == mt_conversion_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "mt_conversion", &err);
          assert(bmap.map[1].block_ptr == mt_transfer_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "mt_transfer", &err);
          assert(bmap.map[2].block_ptr == imageReg_Jacobian_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "imageReg_Jacobian", &err);
          assert(bmap.map[3].block_ptr == imageReg_dF_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "imageReg_dF", &err);
          assert(bmap.map[4].block_ptr == imageReg_dF_EnergyStack_kernel && "mismatch block");
          bmap.map[4].kernel = clCreateKernel(bmap.program, "imageReg_dF_EnergyStack", &err);
          assert(bmap.map[5].block_ptr == imageReg_tJJ_reduction_kernel && "mismatch block");
          bmap.map[5].kernel = clCreateKernel(bmap.program, "imageReg_tJJ_reduction", &err);
          assert(bmap.map[6].block_ptr == imageReg_tJdF_reduction_kernel && "mismatch block");
          bmap.map[6].kernel = clCreateKernel(bmap.program, "imageReg_tJdF_reduction", &err);
          assert(bmap.map[7].block_ptr == imageReg_chi2_reduction_kernel && "mismatch block");
          bmap.map[7].kernel = clCreateKernel(bmap.program, "imageReg_chi2_reduction", &err);
          assert(bmap.map[8].block_ptr == imageReg_LM_kernel && "mismatch block");
          bmap.map[8].kernel = clCreateKernel(bmap.program, "imageReg_LM", &err);
          assert(bmap.map[9].block_ptr == imageReg_Update_kernel && "mismatch block");
          bmap.map[9].kernel = clCreateKernel(bmap.program, "imageReg_Update", &err);
          assert(bmap.map[10].block_ptr == merge_mt_kernel && "mismatch block");
          bmap.map[10].kernel = clCreateKernel(bmap.program, "merge_mt", &err);
          assert(bmap.map[11].block_ptr == merge_rawhisdata_kernel && "mismatch block");
          bmap.map[11].kernel = clCreateKernel(bmap.program, "merge_rawhisdata", &err);
          assert(bmap.map[12].block_ptr == imQXAFS_smoothing_kernel && "mismatch block");
          bmap.map[12].kernel = clCreateKernel(bmap.program, "imQXAFS_smoothing", &err);
          assert(bmap.map[13].block_ptr == inverse_mt_transfer_kernel && "mismatch block");
          bmap.map[13].kernel = clCreateKernel(bmap.program, "inverse_mt_transfer", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = mt_conversion_kernel;
  bmap.map[1].block_ptr = mt_transfer_kernel;
  bmap.map[2].block_ptr = imageReg_Jacobian_kernel;
  bmap.map[3].block_ptr = imageReg_dF_kernel;
  bmap.map[4].block_ptr = imageReg_dF_EnergyStack_kernel;
  bmap.map[5].block_ptr = imageReg_tJJ_reduction_kernel;
  bmap.map[6].block_ptr = imageReg_tJdF_reduction_kernel;
  bmap.map[7].block_ptr = imageReg_chi2_reduction_kernel;
  bmap.map[8].block_ptr = imageReg_LM_kernel;
  bmap.map[9].block_ptr = imageReg_Update_kernel;
  bmap.map[10].block_ptr = merge_mt_kernel;
  bmap.map[11].block_ptr = merge_rawhisdata_kernel;
  bmap.map[12].block_ptr = imQXAFS_smoothing_kernel;
  bmap.map[13].block_ptr = inverse_mt_transfer_kernel;
}

