/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "AART.cl.h"

static void initBlocks(void);

// Initialize static data structures
static block_kernel_pair pair_map[2] = {
      { NULL, NULL },
      { NULL, NULL }
};

static block_kernel_map bmap = { 0, 2, initBlocks, pair_map };

// Block function
void (^AART1_kernel)(const cl_ndrange *ndrange, cl_image reconst_img, cl_image prj_img, cl_image dprj_img, cl_float* angle, cl_int sub) =
^(const cl_ndrange *ndrange, cl_image reconst_img, cl_image prj_img, cl_image dprj_img, cl_float* angle, cl_int sub) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel AART1 does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, reconst_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, dprj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, angle, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(sub), &sub, &kargs);
  gcl_log_cl_fatal(err, "setting argument for AART1 failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing AART1 failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^AART2_kernel)(const cl_ndrange *ndrange, cl_image reconst_img, cl_image reconst_dest_img, cl_image dprj_img, cl_float* angle, cl_int sub, cl_float alpha) =
^(const cl_ndrange *ndrange, cl_image reconst_img, cl_image reconst_dest_img, cl_image dprj_img, cl_float* angle, cl_int sub, cl_float alpha) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel AART2 does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, reconst_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, reconst_dest_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, dprj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, angle, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(sub), &sub, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(alpha), &alpha, &kargs);
  gcl_log_cl_fatal(err, "setting argument for AART2 failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing AART2 failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/AART.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == AART1_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "AART1", &err);
          assert(bmap.map[1].block_ptr == AART2_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "AART2", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = AART1_kernel;
  bmap.map[1].block_ptr = AART2_kernel;
}

