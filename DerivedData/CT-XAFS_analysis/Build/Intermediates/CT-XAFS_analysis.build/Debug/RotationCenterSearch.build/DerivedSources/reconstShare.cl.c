/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "reconstShare.cl.h"

static void initBlocks(void);

// Initialize static data structures
static block_kernel_pair pair_map[8] = {
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL }
};

static block_kernel_map bmap = { 0, 8, initBlocks, pair_map };

// Block function
void (^sinogramCorrection_kernel)(const cl_ndrange *ndrange, cl_image prj_img_src, cl_image prj_img_dst, cl_float* angle, cl_int mode, cl_float a, cl_float b) =
^(const cl_ndrange *ndrange, cl_image prj_img_src, cl_image prj_img_dst, cl_float* angle, cl_int mode, cl_float a, cl_float b) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel sinogramCorrection does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, prj_img_src, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_img_dst, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, angle, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(mode), &mode, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(a), &a, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(b), &b, &kargs);
  gcl_log_cl_fatal(err, "setting argument for sinogramCorrection failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing sinogramCorrection failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^setThreshold_kernel)(const cl_ndrange *ndrange, cl_image img_src, cl_image img_dest, cl_float threshold) =
^(const cl_ndrange *ndrange, cl_image img_src, cl_image img_dest, cl_float threshold) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel setThreshold does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, img_src, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, img_dest, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(threshold), &threshold, &kargs);
  gcl_log_cl_fatal(err, "setting argument for setThreshold failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing setThreshold failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^baseUp_kernel)(const cl_ndrange *ndrange, cl_image img_src, cl_image img_dest, cl_float* baseup, cl_int order) =
^(const cl_ndrange *ndrange, cl_image img_src, cl_image img_dest, cl_float* baseup, cl_int order) {
  int err = 0;
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel baseUp does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, img_src, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, img_dest, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, baseup, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(order), &order, &kargs);
  gcl_log_cl_fatal(err, "setting argument for baseUp failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing baseUp failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^findMinimumX_kernel)(const cl_ndrange *ndrange, cl_image img_src, size_t loc_mem, cl_float* minimumY) =
^(const cl_ndrange *ndrange, cl_image img_src, size_t loc_mem, cl_float* minimumY) {
  int err = 0;
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel findMinimumX does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, img_src, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, loc_mem, NULL, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, minimumY, &kargs);
  gcl_log_cl_fatal(err, "setting argument for findMinimumX failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing findMinimumX failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^findMinimumY_kernel)(const cl_ndrange *ndrange, cl_float* minimumY, size_t loc_mem, cl_float* minimum) =
^(const cl_ndrange *ndrange, cl_float* minimumY, size_t loc_mem, cl_float* minimum) {
  int err = 0;
  cl_kernel k = bmap.map[4].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[4].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel findMinimumY does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, minimumY, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, loc_mem, NULL, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, minimum, &kargs);
  gcl_log_cl_fatal(err, "setting argument for findMinimumY failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing findMinimumY failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^partialDerivativeOfGradiant_kernel)(const cl_ndrange *ndrange, cl_image original_img_src, cl_image img_src, cl_image img_dest, cl_float epsilon, cl_float alpha) =
^(const cl_ndrange *ndrange, cl_image original_img_src, cl_image img_src, cl_image img_dest, cl_float epsilon, cl_float alpha) {
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
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(epsilon), &epsilon, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(alpha), &alpha, &kargs);
  gcl_log_cl_fatal(err, "setting argument for partialDerivativeOfGradiant failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing partialDerivativeOfGradiant failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^projection_kernel)(const cl_ndrange *ndrange, cl_image reconst_img, cl_image prj_img, cl_float* angle, cl_int sub) =
^(const cl_ndrange *ndrange, cl_image reconst_img, cl_image prj_img, cl_float* angle, cl_int sub) {
  int err = 0;
  cl_kernel k = bmap.map[6].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[6].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel projection does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, reconst_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, angle, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(sub), &sub, &kargs);
  gcl_log_cl_fatal(err, "setting argument for projection failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing projection failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^backProjection_kernel)(const cl_ndrange *ndrange, cl_image reconst_dest_img, cl_image prj_img, cl_float* angle, cl_int sub) =
^(const cl_ndrange *ndrange, cl_image reconst_dest_img, cl_image prj_img, cl_float* angle, cl_int sub) {
  int err = 0;
  cl_kernel k = bmap.map[7].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[7].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel backProjection does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, reconst_dest_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, angle, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(sub), &sub, &kargs);
  gcl_log_cl_fatal(err, "setting argument for backProjection failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing backProjection failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/reconstShare.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == sinogramCorrection_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "sinogramCorrection", &err);
          assert(bmap.map[1].block_ptr == setThreshold_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "setThreshold", &err);
          assert(bmap.map[2].block_ptr == baseUp_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "baseUp", &err);
          assert(bmap.map[3].block_ptr == findMinimumX_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "findMinimumX", &err);
          assert(bmap.map[4].block_ptr == findMinimumY_kernel && "mismatch block");
          bmap.map[4].kernel = clCreateKernel(bmap.program, "findMinimumY", &err);
          assert(bmap.map[5].block_ptr == partialDerivativeOfGradiant_kernel && "mismatch block");
          bmap.map[5].kernel = clCreateKernel(bmap.program, "partialDerivativeOfGradiant", &err);
          assert(bmap.map[6].block_ptr == projection_kernel && "mismatch block");
          bmap.map[6].kernel = clCreateKernel(bmap.program, "projection", &err);
          assert(bmap.map[7].block_ptr == backProjection_kernel && "mismatch block");
          bmap.map[7].kernel = clCreateKernel(bmap.program, "backProjection", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = sinogramCorrection_kernel;
  bmap.map[1].block_ptr = setThreshold_kernel;
  bmap.map[2].block_ptr = baseUp_kernel;
  bmap.map[3].block_ptr = findMinimumX_kernel;
  bmap.map[4].block_ptr = findMinimumY_kernel;
  bmap.map[5].block_ptr = partialDerivativeOfGradiant_kernel;
  bmap.map[6].block_ptr = projection_kernel;
  bmap.map[7].block_ptr = backProjection_kernel;
}

