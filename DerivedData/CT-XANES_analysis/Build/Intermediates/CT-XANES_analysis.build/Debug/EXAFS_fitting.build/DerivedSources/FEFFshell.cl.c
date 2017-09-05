/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "FEFFshell.cl.h"

static void initBlocks(void);

// Initialize static data structures
static block_kernel_pair pair_map[1] = {
      { NULL, NULL }
};

static block_kernel_map bmap = { 0, 1, initBlocks, pair_map };

// Block function
void (^redimension_feffShellPara_kernel)(const cl_ndrange *ndrange, cl_image paraW, cl_image paraW_raw, cl_float* kw, cl_int numPnts, cl_int offset) =
^(const cl_ndrange *ndrange, cl_image paraW, cl_image paraW_raw, cl_float* kw, cl_int numPnts, cl_int offset) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel redimension_feffShellPara does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, paraW, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, paraW_raw, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, kw, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(numPnts), &numPnts, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(offset), &offset, &kargs);
  gcl_log_cl_fatal(err, "setting argument for redimension_feffShellPara failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing redimension_feffShellPara failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/FEFFshell.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == redimension_feffShellPara_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "redimension_feffShellPara", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = redimension_feffShellPara_kernel;
}

