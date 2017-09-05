/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "imageregistration_kernel_src.cl.h"

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
void (^mt_conversion_kernel)(const cl_ndrange *ndrange, cl_float* dark, cl_float* I0, cl_ushort* It_buffer, cl_float* mt_buffer, cl_image mt_img1, cl_image mt_img2, cl_int shapeNo, cl_int startpntX, cl_int startpntY, cl_uint width, cl_uint height, cl_float angle) =
^(const cl_ndrange *ndrange, cl_float* dark, cl_float* I0, cl_ushort* It_buffer, cl_float* mt_buffer, cl_image mt_img1, cl_image mt_img2, cl_int shapeNo, cl_int startpntX, cl_int startpntY, cl_uint width, cl_uint height, cl_float angle) {
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
  gcl_log_cl_fatal(err, "setting argument for mt_conversion failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing mt_conversion failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^mt_conversion_binning_kernel)(const cl_ndrange *ndrange, cl_float* dark_buffer, cl_float* I0_buffer, cl_ushort* It_buffer, cl_float* mt_buffer, cl_image mt_img1, cl_image mt_img2, cl_int shapeNo, cl_int startpntX, cl_int startpntY, cl_uint width, cl_uint height, cl_float angle, cl_int binningN) =
^(const cl_ndrange *ndrange, cl_float* dark_buffer, cl_float* I0_buffer, cl_ushort* It_buffer, cl_float* mt_buffer, cl_image mt_img1, cl_image mt_img2, cl_int shapeNo, cl_int startpntX, cl_int startpntY, cl_uint width, cl_uint height, cl_float angle, cl_int binningN) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel mt_conversion_binning does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, dark_buffer, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, I0_buffer, &kargs);
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
  err |= gclSetKernelArgAPPLE(k, 12, sizeof(binningN), &binningN, &kargs);
  gcl_log_cl_fatal(err, "setting argument for mt_conversion_binning failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing mt_conversion_binning failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^mt_transfer_kernel)(const cl_ndrange *ndrange, cl_float* mt_buffer, cl_image mt_img1, cl_image mt_img2, cl_int shapeNo, cl_int startpntX, cl_int startpntY, cl_uint width, cl_uint height, cl_float angle) =
^(const cl_ndrange *ndrange, cl_float* mt_buffer, cl_image mt_img1, cl_image mt_img2, cl_int shapeNo, cl_int startpntX, cl_int startpntY, cl_uint width, cl_uint height, cl_float angle) {
  int err = 0;
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel mt_transfer does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_buffer, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_img1, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, mt_img2, &kargs);
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

void (^merge_kernel)(const cl_ndrange *ndrange, cl_image input_img, cl_image output_img, cl_uint mergeN) =
^(const cl_ndrange *ndrange, cl_image input_img, cl_image output_img, cl_uint mergeN) {
  int err = 0;
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel merge does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, input_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, output_img, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(mergeN), &mergeN, &kargs);
  gcl_log_cl_fatal(err, "setting argument for merge failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing merge failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^imageReg1X_kernel)(const cl_ndrange *ndrange, cl_image mt_t_img, cl_image mt_s_img, cl_image weight_img, cl_float* dF2, cl_float* dF, cl_float CI, cl_float* p, cl_float* p_target, cl_char* p_fix, cl_float* dF2X, cl_float* tJdFX, cl_float* tJJX, cl_float* devX, cl_int mergeN, size_t loc_mem, cl_int difstep) =
^(const cl_ndrange *ndrange, cl_image mt_t_img, cl_image mt_s_img, cl_image weight_img, cl_float* dF2, cl_float* dF, cl_float CI, cl_float* p, cl_float* p_target, cl_char* p_fix, cl_float* dF2X, cl_float* tJdFX, cl_float* tJJX, cl_float* devX, cl_int mergeN, size_t loc_mem, cl_int difstep) {
  int err = 0;
  cl_kernel k = bmap.map[4].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[4].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imageReg1X does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_t_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_s_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, weight_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, dF2, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, dF, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(CI), &CI, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 6, p, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 7, p_target, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 8, p_fix, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 9, dF2X, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 10, tJdFX, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 11, tJJX, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 12, devX, &kargs);
  err |= gclSetKernelArgAPPLE(k, 13, sizeof(mergeN), &mergeN, &kargs);
  err |= gclSetKernelArgAPPLE(k, 14, loc_mem, NULL, &kargs);
  err |= gclSetKernelArgAPPLE(k, 15, sizeof(difstep), &difstep, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imageReg1X failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imageReg1X failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^imageReg1Y_kernel)(const cl_ndrange *ndrange, cl_float* dF2X, cl_float* tJdFX, cl_float* tJJX, cl_float* devX, cl_float* dF2, cl_float* tJdF, cl_float* tJJ, cl_float* dev, cl_int mergeN, size_t loc_mem) =
^(const cl_ndrange *ndrange, cl_float* dF2X, cl_float* tJdFX, cl_float* tJJX, cl_float* devX, cl_float* dF2, cl_float* tJdF, cl_float* tJJ, cl_float* dev, cl_int mergeN, size_t loc_mem) {
  int err = 0;
  cl_kernel k = bmap.map[5].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[5].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imageReg1Y does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, dF2X, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, tJdFX, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, tJJX, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, devX, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, dF2, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, tJdF, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 6, tJJ, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 7, dev, &kargs);
  err |= gclSetKernelArgAPPLE(k, 8, sizeof(mergeN), &mergeN, &kargs);
  err |= gclSetKernelArgAPPLE(k, 9, loc_mem, NULL, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imageReg1Y failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imageReg1Y failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^imageReg2X_kernel)(const cl_ndrange *ndrange, cl_image mt_t_img, cl_image mt_s_img, cl_image weight_img, cl_float* p, cl_float* p_target, cl_float* dF2X, cl_float* dFX, cl_float* devX, cl_int mergeN, size_t loc_mem) =
^(const cl_ndrange *ndrange, cl_image mt_t_img, cl_image mt_s_img, cl_image weight_img, cl_float* p, cl_float* p_target, cl_float* dF2X, cl_float* dFX, cl_float* devX, cl_int mergeN, size_t loc_mem) {
  int err = 0;
  cl_kernel k = bmap.map[6].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[6].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imageReg2X does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_t_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_s_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, weight_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, p, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, p_target, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, dF2X, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 6, dFX, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 7, devX, &kargs);
  err |= gclSetKernelArgAPPLE(k, 8, sizeof(mergeN), &mergeN, &kargs);
  err |= gclSetKernelArgAPPLE(k, 9, loc_mem, NULL, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imageReg2X failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imageReg2X failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^imageReg2Y_kernel)(const cl_ndrange *ndrange, cl_float* dF2X, cl_float* dFX, cl_float* devX, cl_float* dF2, cl_float* dF, cl_int mergeN, size_t loc_mem) =
^(const cl_ndrange *ndrange, cl_float* dF2X, cl_float* dFX, cl_float* devX, cl_float* dF2, cl_float* dF, cl_int mergeN, size_t loc_mem) {
  int err = 0;
  cl_kernel k = bmap.map[7].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[7].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imageReg2Y does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, dF2X, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, dFX, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, devX, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, dF2, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, dF, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(mergeN), &mergeN, &kargs);
  err |= gclSetKernelArgAPPLE(k, 6, loc_mem, NULL, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imageReg2Y failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imageReg2Y failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^estimateParaError_kernel)(const cl_ndrange *ndrange, cl_float* p_error, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* dev_img) =
^(const cl_ndrange *ndrange, cl_float* p_error, cl_float* tJdF_img, cl_float* tJJ_img, cl_float* dev_img) {
  int err = 0;
  cl_kernel k = bmap.map[8].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[8].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel estimateParaError does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, p_error, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, tJdF_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, tJJ_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, dev_img, &kargs);
  gcl_log_cl_fatal(err, "setting argument for estimateParaError failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing estimateParaError failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^output_imgReg_result_kernel)(const cl_ndrange *ndrange, cl_image mt_img, cl_float* mt_buf, cl_float* p) =
^(const cl_ndrange *ndrange, cl_image mt_img, cl_float* mt_buf, cl_float* p) {
  int err = 0;
  cl_kernel k = bmap.map[9].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[9].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel output_imgReg_result does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_buf, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, p, &kargs);
  gcl_log_cl_fatal(err, "setting argument for output_imgReg_result failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing output_imgReg_result failed");
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

void (^updateWeight_kernel)(const cl_ndrange *ndrange, cl_image mt_t_img, cl_image mt_s_img, cl_image weight_img, cl_float* dF2, cl_float* dF, cl_float CI, cl_float* p, cl_float* p_target, cl_int mergeN) =
^(const cl_ndrange *ndrange, cl_image mt_t_img, cl_image mt_s_img, cl_image weight_img, cl_float* dF2, cl_float* dF, cl_float CI, cl_float* p, cl_float* p_target, cl_int mergeN) {
  int err = 0;
  cl_kernel k = bmap.map[13].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[13].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel updateWeight does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_t_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_s_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, weight_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, dF2, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, dF, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(CI), &CI, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 6, p, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 7, p_target, &kargs);
  err |= gclSetKernelArgAPPLE(k, 8, sizeof(mergeN), &mergeN, &kargs);
  gcl_log_cl_fatal(err, "setting argument for updateWeight failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing updateWeight failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/imageregistration_kernel_src.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == mt_conversion_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "mt_conversion", &err);
          assert(bmap.map[1].block_ptr == mt_conversion_binning_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "mt_conversion_binning", &err);
          assert(bmap.map[2].block_ptr == mt_transfer_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "mt_transfer", &err);
          assert(bmap.map[3].block_ptr == merge_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "merge", &err);
          assert(bmap.map[4].block_ptr == imageReg1X_kernel && "mismatch block");
          bmap.map[4].kernel = clCreateKernel(bmap.program, "imageReg1X", &err);
          assert(bmap.map[5].block_ptr == imageReg1Y_kernel && "mismatch block");
          bmap.map[5].kernel = clCreateKernel(bmap.program, "imageReg1Y", &err);
          assert(bmap.map[6].block_ptr == imageReg2X_kernel && "mismatch block");
          bmap.map[6].kernel = clCreateKernel(bmap.program, "imageReg2X", &err);
          assert(bmap.map[7].block_ptr == imageReg2Y_kernel && "mismatch block");
          bmap.map[7].kernel = clCreateKernel(bmap.program, "imageReg2Y", &err);
          assert(bmap.map[8].block_ptr == estimateParaError_kernel && "mismatch block");
          bmap.map[8].kernel = clCreateKernel(bmap.program, "estimateParaError", &err);
          assert(bmap.map[9].block_ptr == output_imgReg_result_kernel && "mismatch block");
          bmap.map[9].kernel = clCreateKernel(bmap.program, "output_imgReg_result", &err);
          assert(bmap.map[10].block_ptr == merge_mt_kernel && "mismatch block");
          bmap.map[10].kernel = clCreateKernel(bmap.program, "merge_mt", &err);
          assert(bmap.map[11].block_ptr == merge_rawhisdata_kernel && "mismatch block");
          bmap.map[11].kernel = clCreateKernel(bmap.program, "merge_rawhisdata", &err);
          assert(bmap.map[12].block_ptr == imQXAFS_smoothing_kernel && "mismatch block");
          bmap.map[12].kernel = clCreateKernel(bmap.program, "imQXAFS_smoothing", &err);
          assert(bmap.map[13].block_ptr == updateWeight_kernel && "mismatch block");
          bmap.map[13].kernel = clCreateKernel(bmap.program, "updateWeight", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = mt_conversion_kernel;
  bmap.map[1].block_ptr = mt_conversion_binning_kernel;
  bmap.map[2].block_ptr = mt_transfer_kernel;
  bmap.map[3].block_ptr = merge_kernel;
  bmap.map[4].block_ptr = imageReg1X_kernel;
  bmap.map[5].block_ptr = imageReg1Y_kernel;
  bmap.map[6].block_ptr = imageReg2X_kernel;
  bmap.map[7].block_ptr = imageReg2Y_kernel;
  bmap.map[8].block_ptr = estimateParaError_kernel;
  bmap.map[9].block_ptr = output_imgReg_result_kernel;
  bmap.map[10].block_ptr = merge_mt_kernel;
  bmap.map[11].block_ptr = merge_rawhisdata_kernel;
  bmap.map[12].block_ptr = imQXAFS_smoothing_kernel;
  bmap.map[13].block_ptr = updateWeight_kernel;
}

