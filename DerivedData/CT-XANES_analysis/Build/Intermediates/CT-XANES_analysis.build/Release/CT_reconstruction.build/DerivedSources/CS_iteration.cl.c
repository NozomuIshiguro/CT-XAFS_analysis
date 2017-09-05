/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "CS_iteration.cl.h"

static void initBlocks(void);

// Initialize static data structures
static block_kernel_pair pair_map[1] = {
      { NULL, NULL }
};

static block_kernel_map bmap = { 0, 1, initBlocks, pair_map };

// Block function
void (^partialDerivativeOfGradiant_kernel)(const cl_ndrange *ndrange, cl_image img_src, cl_image v_img, cl_float epsilon) =
^(const cl_ndrange *ndrange, cl_image img_src, cl_image v_img, cl_float epsilon) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel partialDerivativeOfGradiant does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, img_src, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, v_img, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(epsilon), &epsilon, &kargs);
  gcl_log_cl_fatal(err, "setting argument for partialDerivativeOfGradiant failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing partialDerivativeOfGradiant failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/CS_iteration.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == partialDerivativeOfGradiant_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "partialDerivativeOfGradiant", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = partialDerivativeOfGradiant_kernel;
}

