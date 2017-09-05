/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "threeD_FFT.cl.h"

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
void (^XY_transpose_kernel)(const cl_ndrange *ndrange, cl_float2* src, cl_float2* dest) =
^(const cl_ndrange *ndrange, cl_float2* src, cl_float2* dest) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel XY_transpose does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, src, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, dest, &kargs);
  gcl_log_cl_fatal(err, "setting argument for XY_transpose failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing XY_transpose failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^XZ_transpose_kernel)(const cl_ndrange *ndrange, cl_float2* src, cl_float2* dest, cl_int offsetX_src, cl_int x_size_src, cl_int offsetX_dst, cl_int x_size_dst, cl_int offsetY_src, cl_int y_size_src, cl_int offsetY_dst, cl_int y_size_dst) =
^(const cl_ndrange *ndrange, cl_float2* src, cl_float2* dest, cl_int offsetX_src, cl_int x_size_src, cl_int offsetX_dst, cl_int x_size_dst, cl_int offsetY_src, cl_int y_size_src, cl_int offsetY_dst, cl_int y_size_dst) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel XZ_transpose does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, src, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, dest, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(offsetX_src), &offsetX_src, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(x_size_src), &x_size_src, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(offsetX_dst), &offsetX_dst, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(x_size_dst), &x_size_dst, &kargs);
  err |= gclSetKernelArgAPPLE(k, 6, sizeof(offsetY_src), &offsetY_src, &kargs);
  err |= gclSetKernelArgAPPLE(k, 7, sizeof(y_size_src), &y_size_src, &kargs);
  err |= gclSetKernelArgAPPLE(k, 8, sizeof(offsetY_dst), &offsetY_dst, &kargs);
  err |= gclSetKernelArgAPPLE(k, 9, sizeof(y_size_dst), &y_size_dst, &kargs);
  gcl_log_cl_fatal(err, "setting argument for XZ_transpose failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing XZ_transpose failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^spinFact_kernel)(const cl_ndrange *ndrange, cl_float2* w) =
^(const cl_ndrange *ndrange, cl_float2* w) {
  int err = 0;
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel spinFact does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, w, &kargs);
  gcl_log_cl_fatal(err, "setting argument for spinFact failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing spinFact failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^bitReverse_kernel)(const cl_ndrange *ndrange, cl_float2* src, cl_float2* dest, cl_int M) =
^(const cl_ndrange *ndrange, cl_float2* src, cl_float2* dest, cl_int M) {
  int err = 0;
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel bitReverse does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, src, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, dest, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(M), &M, &kargs);
  gcl_log_cl_fatal(err, "setting argument for bitReverse failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing bitReverse failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^bitReverseAndXZ_transpose_kernel)(const cl_ndrange *ndrange, cl_float2* src, cl_float2* dest, cl_int M, cl_int offsetX_src, cl_int x_size_src, cl_int offsetX_dst, cl_int x_size_dst, cl_int offsetY_src, cl_int y_size_src, cl_int offsetY_dst, cl_int y_size_dst) =
^(const cl_ndrange *ndrange, cl_float2* src, cl_float2* dest, cl_int M, cl_int offsetX_src, cl_int x_size_src, cl_int offsetX_dst, cl_int x_size_dst, cl_int offsetY_src, cl_int y_size_src, cl_int offsetY_dst, cl_int y_size_dst) {
  int err = 0;
  cl_kernel k = bmap.map[4].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[4].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel bitReverseAndXZ_transpose does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, src, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, dest, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(M), &M, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(offsetX_src), &offsetX_src, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(x_size_src), &x_size_src, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(offsetX_dst), &offsetX_dst, &kargs);
  err |= gclSetKernelArgAPPLE(k, 6, sizeof(x_size_dst), &x_size_dst, &kargs);
  err |= gclSetKernelArgAPPLE(k, 7, sizeof(offsetY_src), &offsetY_src, &kargs);
  err |= gclSetKernelArgAPPLE(k, 8, sizeof(y_size_src), &y_size_src, &kargs);
  err |= gclSetKernelArgAPPLE(k, 9, sizeof(offsetY_dst), &offsetY_dst, &kargs);
  err |= gclSetKernelArgAPPLE(k, 10, sizeof(y_size_dst), &y_size_dst, &kargs);
  gcl_log_cl_fatal(err, "setting argument for bitReverseAndXZ_transpose failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing bitReverseAndXZ_transpose failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^butterfly_kernel)(const cl_ndrange *ndrange, cl_float2* x_fft, cl_float2* w, cl_uint iter, cl_uint flag) =
^(const cl_ndrange *ndrange, cl_float2* x_fft, cl_float2* w, cl_uint iter, cl_uint flag) {
  int err = 0;
  cl_kernel k = bmap.map[5].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[5].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel butterfly does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, x_fft, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, w, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(iter), &iter, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(flag), &flag, &kargs);
  gcl_log_cl_fatal(err, "setting argument for butterfly failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing butterfly failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^FFTnorm_kernel)(const cl_ndrange *ndrange, cl_float2* x_fft, cl_float xgrid) =
^(const cl_ndrange *ndrange, cl_float2* x_fft, cl_float xgrid) {
  int err = 0;
  cl_kernel k = bmap.map[6].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[6].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel FFTnorm does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, x_fft, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, sizeof(xgrid), &xgrid, &kargs);
  gcl_log_cl_fatal(err, "setting argument for FFTnorm failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing FFTnorm failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^IFFTnorm_kernel)(const cl_ndrange *ndrange, cl_float2* x_ifft, cl_float xgrid) =
^(const cl_ndrange *ndrange, cl_float2* x_ifft, cl_float xgrid) {
  int err = 0;
  cl_kernel k = bmap.map[7].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[7].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel IFFTnorm does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, x_ifft, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, sizeof(xgrid), &xgrid, &kargs);
  gcl_log_cl_fatal(err, "setting argument for IFFTnorm failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing IFFTnorm failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/threeD_FFT.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == XY_transpose_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "XY_transpose", &err);
          assert(bmap.map[1].block_ptr == XZ_transpose_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "XZ_transpose", &err);
          assert(bmap.map[2].block_ptr == spinFact_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "spinFact", &err);
          assert(bmap.map[3].block_ptr == bitReverse_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "bitReverse", &err);
          assert(bmap.map[4].block_ptr == bitReverseAndXZ_transpose_kernel && "mismatch block");
          bmap.map[4].kernel = clCreateKernel(bmap.program, "bitReverseAndXZ_transpose", &err);
          assert(bmap.map[5].block_ptr == butterfly_kernel && "mismatch block");
          bmap.map[5].kernel = clCreateKernel(bmap.program, "butterfly", &err);
          assert(bmap.map[6].block_ptr == FFTnorm_kernel && "mismatch block");
          bmap.map[6].kernel = clCreateKernel(bmap.program, "FFTnorm", &err);
          assert(bmap.map[7].block_ptr == IFFTnorm_kernel && "mismatch block");
          bmap.map[7].kernel = clCreateKernel(bmap.program, "IFFTnorm", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = XY_transpose_kernel;
  bmap.map[1].block_ptr = XZ_transpose_kernel;
  bmap.map[2].block_ptr = spinFact_kernel;
  bmap.map[3].block_ptr = bitReverse_kernel;
  bmap.map[4].block_ptr = bitReverseAndXZ_transpose_kernel;
  bmap.map[5].block_ptr = butterfly_kernel;
  bmap.map[6].block_ptr = FFTnorm_kernel;
  bmap.map[7].block_ptr = IFFTnorm_kernel;
}

