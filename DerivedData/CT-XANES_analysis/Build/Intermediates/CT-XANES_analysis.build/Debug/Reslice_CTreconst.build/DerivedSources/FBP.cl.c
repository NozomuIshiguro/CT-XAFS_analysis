/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "FBP.cl.h"

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
void (^spinFactor_kernel)(const cl_ndrange *ndrange, cl_float2* W) =
^(const cl_ndrange *ndrange, cl_float2* W) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel spinFactor does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, W, &kargs);
  gcl_log_cl_fatal(err, "setting argument for spinFactor failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing spinFactor failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^zeroPadding_kernel)(const cl_ndrange *ndrange, cl_image prj_img, cl_float2* xc) =
^(const cl_ndrange *ndrange, cl_image prj_img, cl_float2* xc) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel zeroPadding does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, prj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, xc, &kargs);
  gcl_log_cl_fatal(err, "setting argument for zeroPadding failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing zeroPadding failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^bitReverse_kernel)(const cl_ndrange *ndrange, cl_float2* xc_src, cl_float2* xc_dest, cl_uint iter) =
^(const cl_ndrange *ndrange, cl_float2* xc_src, cl_float2* xc_dest, cl_uint iter) {
  int err = 0;
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel bitReverse does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, xc_src, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, xc_dest, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(iter), &iter, &kargs);
  gcl_log_cl_fatal(err, "setting argument for bitReverse failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing bitReverse failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^butterfly_kernel)(const cl_ndrange *ndrange, cl_float2* xc, cl_float2* W, cl_uint flag, cl_int iter) =
^(const cl_ndrange *ndrange, cl_float2* xc, cl_float2* W, cl_uint flag, cl_int iter) {
  int err = 0;
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel butterfly does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, xc, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, W, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(flag), &flag, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(iter), &iter, &kargs);
  gcl_log_cl_fatal(err, "setting argument for butterfly failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing butterfly failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^filtering_kernel)(const cl_ndrange *ndrange, cl_float2* xc) =
^(const cl_ndrange *ndrange, cl_float2* xc) {
  int err = 0;
  cl_kernel k = bmap.map[4].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[4].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel filtering does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, xc, &kargs);
  gcl_log_cl_fatal(err, "setting argument for filtering failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing filtering failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^normalization_kernel)(const cl_ndrange *ndrange, cl_float2* xc) =
^(const cl_ndrange *ndrange, cl_float2* xc) {
  int err = 0;
  cl_kernel k = bmap.map[5].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[5].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel normalization does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, xc, &kargs);
  gcl_log_cl_fatal(err, "setting argument for normalization failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing normalization failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^outputImage_kernel)(const cl_ndrange *ndrange, cl_float2* xc, cl_image fprj_img) =
^(const cl_ndrange *ndrange, cl_float2* xc, cl_image fprj_img) {
  int err = 0;
  cl_kernel k = bmap.map[6].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[6].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel outputImage does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, xc, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, fprj_img, &kargs);
  gcl_log_cl_fatal(err, "setting argument for outputImage failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing outputImage failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^backProjectionFBP_kernel)(const cl_ndrange *ndrange, cl_image fprj_img, cl_image reconst_img, cl_float* angle) =
^(const cl_ndrange *ndrange, cl_image fprj_img, cl_image reconst_img, cl_float* angle) {
  int err = 0;
  cl_kernel k = bmap.map[7].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[7].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel backProjectionFBP does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, fprj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, reconst_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, angle, &kargs);
  gcl_log_cl_fatal(err, "setting argument for backProjectionFBP failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing backProjectionFBP failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/FBP.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == spinFactor_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "spinFactor", &err);
          assert(bmap.map[1].block_ptr == zeroPadding_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "zeroPadding", &err);
          assert(bmap.map[2].block_ptr == bitReverse_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "bitReverse", &err);
          assert(bmap.map[3].block_ptr == butterfly_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "butterfly", &err);
          assert(bmap.map[4].block_ptr == filtering_kernel && "mismatch block");
          bmap.map[4].kernel = clCreateKernel(bmap.program, "filtering", &err);
          assert(bmap.map[5].block_ptr == normalization_kernel && "mismatch block");
          bmap.map[5].kernel = clCreateKernel(bmap.program, "normalization", &err);
          assert(bmap.map[6].block_ptr == outputImage_kernel && "mismatch block");
          bmap.map[6].kernel = clCreateKernel(bmap.program, "outputImage", &err);
          assert(bmap.map[7].block_ptr == backProjectionFBP_kernel && "mismatch block");
          bmap.map[7].kernel = clCreateKernel(bmap.program, "backProjectionFBP", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = spinFactor_kernel;
  bmap.map[1].block_ptr = zeroPadding_kernel;
  bmap.map[2].block_ptr = bitReverse_kernel;
  bmap.map[3].block_ptr = butterfly_kernel;
  bmap.map[4].block_ptr = filtering_kernel;
  bmap.map[5].block_ptr = normalization_kernel;
  bmap.map[6].block_ptr = outputImage_kernel;
  bmap.map[7].block_ptr = backProjectionFBP_kernel;
}

