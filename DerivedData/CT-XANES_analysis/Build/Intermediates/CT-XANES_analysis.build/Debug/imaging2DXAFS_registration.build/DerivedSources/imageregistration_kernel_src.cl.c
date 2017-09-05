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
static block_kernel_pair pair_map[7] = {
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL }
};

static block_kernel_map bmap = { 0, 7, initBlocks, pair_map };

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

void (^mt_transfer_kernel)(const cl_ndrange *ndrange, cl_float* mt_buffer, cl_image mt_img1, cl_image mt_img2, cl_int shapeNo, cl_int startpntX, cl_int startpntY, cl_uint width, cl_uint height, cl_float angle) =
^(const cl_ndrange *ndrange, cl_float* mt_buffer, cl_image mt_img1, cl_image mt_img2, cl_int shapeNo, cl_int startpntX, cl_int startpntY, cl_uint width, cl_uint height, cl_float angle) {
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
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
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

void (^imageRegistration_kernel)(const cl_ndrange *ndrange, cl_image mt_t_img, cl_image mt_s_img, cl_float* lambda_buffer, cl_float* p, cl_float* p_err, cl_float* p_target, cl_float* p_fix, size_t loc_mem, cl_uint mergeN, cl_float epsilon) =
^(const cl_ndrange *ndrange, cl_image mt_t_img, cl_image mt_s_img, cl_float* lambda_buffer, cl_float* p, cl_float* p_err, cl_float* p_target, cl_float* p_fix, size_t loc_mem, cl_uint mergeN, cl_float epsilon) {
  int err = 0;
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imageRegistration does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_t_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_s_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, lambda_buffer, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, p, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, p_err, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, p_target, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 6, p_fix, &kargs);
  err |= gclSetKernelArgAPPLE(k, 7, loc_mem, NULL, &kargs);
  err |= gclSetKernelArgAPPLE(k, 8, sizeof(mergeN), &mergeN, &kargs);
  err |= gclSetKernelArgAPPLE(k, 9, sizeof(epsilon), &epsilon, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imageRegistration failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imageRegistration failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^output_imgReg_result_kernel)(const cl_ndrange *ndrange, cl_image mt_img, cl_float* mt_buf, cl_float* p) =
^(const cl_ndrange *ndrange, cl_image mt_img, cl_float* mt_buf, cl_float* p) {
  int err = 0;
  cl_kernel k = bmap.map[4].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[4].kernel;
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
  cl_kernel k = bmap.map[5].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[5].kernel;
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
  cl_kernel k = bmap.map[6].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[6].kernel;
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

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/imageregistration_kernel_src.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == mt_conversion_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "mt_conversion", &err);
          assert(bmap.map[1].block_ptr == mt_transfer_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "mt_transfer", &err);
          assert(bmap.map[2].block_ptr == merge_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "merge", &err);
          assert(bmap.map[3].block_ptr == imageRegistration_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "imageRegistration", &err);
          assert(bmap.map[4].block_ptr == output_imgReg_result_kernel && "mismatch block");
          bmap.map[4].kernel = clCreateKernel(bmap.program, "output_imgReg_result", &err);
          assert(bmap.map[5].block_ptr == merge_mt_kernel && "mismatch block");
          bmap.map[5].kernel = clCreateKernel(bmap.program, "merge_mt", &err);
          assert(bmap.map[6].block_ptr == merge_rawhisdata_kernel && "mismatch block");
          bmap.map[6].kernel = clCreateKernel(bmap.program, "merge_rawhisdata", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = mt_conversion_kernel;
  bmap.map[1].block_ptr = mt_transfer_kernel;
  bmap.map[2].block_ptr = merge_kernel;
  bmap.map[3].block_ptr = imageRegistration_kernel;
  bmap.map[4].block_ptr = output_imgReg_result_kernel;
  bmap.map[5].block_ptr = merge_mt_kernel;
  bmap.map[6].block_ptr = merge_rawhisdata_kernel;
}

