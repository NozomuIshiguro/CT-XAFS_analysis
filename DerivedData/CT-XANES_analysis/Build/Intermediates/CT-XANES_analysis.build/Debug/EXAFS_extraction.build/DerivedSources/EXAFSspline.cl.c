/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "EXAFSspline.cl.h"

static void initBlocks(void);

// Initialize static data structures
static block_kernel_pair pair_map[4] = {
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL }
};

static block_kernel_map bmap = { 0, 4, initBlocks, pair_map };

// Block function
void (^Bspline_basis_zero_kernel)(const cl_ndrange *ndrange, cl_float* basis, cl_float h) =
^(const cl_ndrange *ndrange, cl_float* basis, cl_float h) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel Bspline_basis_zero does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, basis, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, sizeof(h), &h, &kargs);
  gcl_log_cl_fatal(err, "setting argument for Bspline_basis_zero failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing Bspline_basis_zero failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^Bspline_orderUpdatingMatrix_kernel)(const cl_ndrange *ndrange, cl_float* OUM, cl_float h, cl_int order) =
^(const cl_ndrange *ndrange, cl_float* OUM, cl_float h, cl_int order) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel Bspline_orderUpdatingMatrix does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, OUM, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, sizeof(h), &h, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(order), &order, &kargs);
  gcl_log_cl_fatal(err, "setting argument for Bspline_orderUpdatingMatrix failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing Bspline_orderUpdatingMatrix failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^Bspline_basis_updateOrder_kernel)(const cl_ndrange *ndrange, cl_float* basis_src, cl_float* basis_dest, cl_float* OUM) =
^(const cl_ndrange *ndrange, cl_float* basis_src, cl_float* basis_dest, cl_float* OUM) {
  int err = 0;
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel Bspline_basis_updateOrder does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, basis_src, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, basis_dest, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, OUM, &kargs);
  gcl_log_cl_fatal(err, "setting argument for Bspline_basis_updateOrder failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing Bspline_basis_updateOrder failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^Bspline_kernel)(const cl_ndrange *ndrange, cl_float* spline, cl_float* basis) =
^(const cl_ndrange *ndrange, cl_float* spline, cl_float* basis) {
  int err = 0;
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel Bspline does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, spline, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, basis, &kargs);
  gcl_log_cl_fatal(err, "setting argument for Bspline failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing Bspline failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/EXAFSspline.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == Bspline_basis_zero_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "Bspline_basis_zero", &err);
          assert(bmap.map[1].block_ptr == Bspline_orderUpdatingMatrix_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "Bspline_orderUpdatingMatrix", &err);
          assert(bmap.map[2].block_ptr == Bspline_basis_updateOrder_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "Bspline_basis_updateOrder", &err);
          assert(bmap.map[3].block_ptr == Bspline_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "Bspline", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = Bspline_basis_zero_kernel;
  bmap.map[1].block_ptr = Bspline_orderUpdatingMatrix_kernel;
  bmap.map[2].block_ptr = Bspline_basis_updateOrder_kernel;
  bmap.map[3].block_ptr = Bspline_kernel;
}

