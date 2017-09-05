/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "rotCenterSearch.cl.h"

static void initBlocks(void);

// Initialize static data structures
static block_kernel_pair pair_map[5] = {
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL }
};

static block_kernel_map bmap = { 0, 5, initBlocks, pair_map };

// Block function
void (^rotCenterShift_kernel)(const cl_ndrange *ndrange, cl_image prj_input_img, cl_image prj_output_img, cl_float rotCenterShift) =
^(const cl_ndrange *ndrange, cl_image prj_input_img, cl_image prj_output_img, cl_float rotCenterShift) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel rotCenterShift does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, prj_input_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_output_img, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(rotCenterShift), &rotCenterShift, &kargs);
  gcl_log_cl_fatal(err, "setting argument for rotCenterShift failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing rotCenterShift failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^setMask_kernel)(const cl_ndrange *ndrange, cl_image mask_img, cl_int offsetN, cl_float startShift, cl_float shiftStep, cl_float min_ang, cl_float max_ang) =
^(const cl_ndrange *ndrange, cl_image mask_img, cl_int offsetN, cl_float startShift, cl_float shiftStep, cl_float min_ang, cl_float max_ang) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel setMask does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mask_img, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, sizeof(offsetN), &offsetN, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(startShift), &startShift, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(shiftStep), &shiftStep, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(min_ang), &min_ang, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(max_ang), &max_ang, &kargs);
  gcl_log_cl_fatal(err, "setting argument for setMask failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing setMask failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^imgAVG_kernel)(const cl_ndrange *ndrange, cl_image img, cl_image mask_img, size_t loc_mem, cl_float* avg) =
^(const cl_ndrange *ndrange, cl_image img, cl_image mask_img, size_t loc_mem, cl_float* avg) {
  int err = 0;
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imgAVG does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mask_img, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, loc_mem, NULL, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, avg, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imgAVG failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imgAVG failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^imgSTDEV_kernel)(const cl_ndrange *ndrange, cl_image img, cl_image mask_img, size_t loc_mem, cl_float* avg, cl_float* stedev) =
^(const cl_ndrange *ndrange, cl_image img, cl_image mask_img, size_t loc_mem, cl_float* avg, cl_float* stedev) {
  int err = 0;
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imgSTDEV does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mask_img, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, loc_mem, NULL, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, avg, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, stedev, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imgSTDEV failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imgSTDEV failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^imgFocusIndex_kernel)(const cl_ndrange *ndrange, cl_image img, cl_image mask_img, size_t loc_mem, cl_float* Findex) =
^(const cl_ndrange *ndrange, cl_image img, cl_image mask_img, size_t loc_mem, cl_float* Findex) {
  int err = 0;
  cl_kernel k = bmap.map[4].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[4].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel imgFocusIndex does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mask_img, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, loc_mem, NULL, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, Findex, &kargs);
  gcl_log_cl_fatal(err, "setting argument for imgFocusIndex failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing imgFocusIndex failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/rotCenterSearch.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == rotCenterShift_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "rotCenterShift", &err);
          assert(bmap.map[1].block_ptr == setMask_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "setMask", &err);
          assert(bmap.map[2].block_ptr == imgAVG_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "imgAVG", &err);
          assert(bmap.map[3].block_ptr == imgSTDEV_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "imgSTDEV", &err);
          assert(bmap.map[4].block_ptr == imgFocusIndex_kernel && "mismatch block");
          bmap.map[4].kernel = clCreateKernel(bmap.program, "imgFocusIndex", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = rotCenterShift_kernel;
  bmap.map[1].block_ptr = setMask_kernel;
  bmap.map[2].block_ptr = imgAVG_kernel;
  bmap.map[3].block_ptr = imgSTDEV_kernel;
  bmap.map[4].block_ptr = imgFocusIndex_kernel;
}

