/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "ISTA.cl.h"

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
void (^powerIter1_kernel)(const cl_ndrange *ndrange, cl_image reconst_img, cl_image prj_img, cl_float* angle) =
^(const cl_ndrange *ndrange, cl_image reconst_img, cl_image prj_img, cl_float* angle) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel powerIter1 does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, reconst_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, angle, &kargs);
  gcl_log_cl_fatal(err, "setting argument for powerIter1 failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing powerIter1 failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^powerIter2_kernel)(const cl_ndrange *ndrange, cl_image reconst_cnd_img, cl_image prj_img, cl_float* angle) =
^(const cl_ndrange *ndrange, cl_image reconst_cnd_img, cl_image prj_img, cl_float* angle) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel powerIter2 does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, reconst_cnd_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, angle, &kargs);
  gcl_log_cl_fatal(err, "setting argument for powerIter2 failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing powerIter2 failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^powerIter3_kernel)(const cl_ndrange *ndrange, cl_image reconst_cnd_img, cl_image reconst_new_img, cl_float* L2abs) =
^(const cl_ndrange *ndrange, cl_image reconst_cnd_img, cl_image reconst_new_img, cl_float* L2abs) {
  int err = 0;
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel powerIter3 does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, reconst_cnd_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, reconst_new_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, L2abs, &kargs);
  gcl_log_cl_fatal(err, "setting argument for powerIter3 failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing powerIter3 failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^imageL2AbsX_kernel)(const cl_ndrange *ndrange, cl_image img_src, size_t loc_mem, cl_float* L2absY) =
^(const cl_ndrange *ndrange, cl_image img_src, size_t loc_mem, cl_float* L2absY) {
  int err = 0;
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imageL2AbsX does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, img_src, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, loc_mem, NULL, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, L2absY, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imageL2AbsX failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imageL2AbsX failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^imageL2AbsY_kernel)(const cl_ndrange *ndrange, cl_float* L2absY, size_t loc_mem, cl_float* L2abs) =
^(const cl_ndrange *ndrange, cl_float* L2absY, size_t loc_mem, cl_float* L2abs) {
  int err = 0;
  cl_kernel k = bmap.map[4].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[4].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imageL2AbsY does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, L2absY, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, loc_mem, NULL, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, L2abs, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imageL2AbsY failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imageL2AbsY failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^ISTA1_kernel)(const cl_ndrange *ndrange, cl_image reconst_img, cl_image prj_img, cl_image vprj_img, cl_float* angle, cl_int sub) =
^(const cl_ndrange *ndrange, cl_image reconst_img, cl_image prj_img, cl_image vprj_img, cl_float* angle, cl_int sub) {
  int err = 0;
  cl_kernel k = bmap.map[5].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[5].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel ISTA1 does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, reconst_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, vprj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, angle, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(sub), &sub, &kargs);
  gcl_log_cl_fatal(err, "setting argument for ISTA1 failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing ISTA1 failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^ISTA2_kernel)(const cl_ndrange *ndrange, cl_image reconst_img, cl_image reconst_v_img, cl_image vprj_img, cl_float* angle, cl_int sub, cl_float* L) =
^(const cl_ndrange *ndrange, cl_image reconst_img, cl_image reconst_v_img, cl_image vprj_img, cl_float* angle, cl_int sub, cl_float* L) {
  int err = 0;
  cl_kernel k = bmap.map[6].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[6].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel ISTA2 does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, reconst_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, reconst_v_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, vprj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, angle, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(sub), &sub, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, L, &kargs);
  gcl_log_cl_fatal(err, "setting argument for ISTA2 failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing ISTA2 failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^ISTA3_kernel)(const cl_ndrange *ndrange, cl_image reconst_img, cl_image reconst_v_img, cl_image reconst_dest_img, cl_int sub, cl_float* L) =
^(const cl_ndrange *ndrange, cl_image reconst_img, cl_image reconst_v_img, cl_image reconst_dest_img, cl_int sub, cl_float* L) {
  int err = 0;
  cl_kernel k = bmap.map[7].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[7].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel ISTA3 does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, reconst_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, reconst_v_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, reconst_dest_img, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(sub), &sub, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, L, &kargs);
  gcl_log_cl_fatal(err, "setting argument for ISTA3 failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing ISTA3 failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/ISTA.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == powerIter1_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "powerIter1", &err);
          assert(bmap.map[1].block_ptr == powerIter2_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "powerIter2", &err);
          assert(bmap.map[2].block_ptr == powerIter3_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "powerIter3", &err);
          assert(bmap.map[3].block_ptr == imageL2AbsX_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "imageL2AbsX", &err);
          assert(bmap.map[4].block_ptr == imageL2AbsY_kernel && "mismatch block");
          bmap.map[4].kernel = clCreateKernel(bmap.program, "imageL2AbsY", &err);
          assert(bmap.map[5].block_ptr == ISTA1_kernel && "mismatch block");
          bmap.map[5].kernel = clCreateKernel(bmap.program, "ISTA1", &err);
          assert(bmap.map[6].block_ptr == ISTA2_kernel && "mismatch block");
          bmap.map[6].kernel = clCreateKernel(bmap.program, "ISTA2", &err);
          assert(bmap.map[7].block_ptr == ISTA3_kernel && "mismatch block");
          bmap.map[7].kernel = clCreateKernel(bmap.program, "ISTA3", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = powerIter1_kernel;
  bmap.map[1].block_ptr = powerIter2_kernel;
  bmap.map[2].block_ptr = powerIter3_kernel;
  bmap.map[3].block_ptr = imageL2AbsX_kernel;
  bmap.map[4].block_ptr = imageL2AbsY_kernel;
  bmap.map[5].block_ptr = ISTA1_kernel;
  bmap.map[6].block_ptr = ISTA2_kernel;
  bmap.map[7].block_ptr = ISTA3_kernel;
}

