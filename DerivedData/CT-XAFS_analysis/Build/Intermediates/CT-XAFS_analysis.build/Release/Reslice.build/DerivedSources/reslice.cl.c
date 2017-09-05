/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "reslice.cl.h"

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
void (^reslice_kernel)(const cl_ndrange *ndrange, cl_image mt_img, cl_image prj_img, cl_float* Xshift, cl_float* Yshift, cl_float baseup, cl_int th, cl_int th_offset, cl_char correction) =
^(const cl_ndrange *ndrange, cl_image mt_img, cl_image prj_img, cl_float* Xshift, cl_float* Yshift, cl_float baseup, cl_int th, cl_int th_offset, cl_char correction) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel reslice does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, Xshift, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, Yshift, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(baseup), &baseup, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(th), &th, &kargs);
  err |= gclSetKernelArgAPPLE(k, 6, sizeof(th_offset), &th_offset, &kargs);
  err |= gclSetKernelArgAPPLE(k, 7, sizeof(correction), &correction, &kargs);
  gcl_log_cl_fatal(err, "setting argument for reslice failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing reslice failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^xprojection_kernel)(const cl_ndrange *ndrange, cl_image mt_img, cl_float* xproj, cl_int startX, cl_int endX, cl_int th) =
^(const cl_ndrange *ndrange, cl_image mt_img, cl_float* xproj, cl_int startX, cl_int endX, cl_int th) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel xprojection does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, xproj, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(startX), &startX, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(endX), &endX, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(th), &th, &kargs);
  gcl_log_cl_fatal(err, "setting argument for xprojection failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing xprojection failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^zcorrection_kernel)(const cl_ndrange *ndrange, cl_image xprj_img, cl_float* yshift, size_t target_xprj, size_t loc_mem, cl_int startY, cl_int endY) =
^(const cl_ndrange *ndrange, cl_image xprj_img, cl_float* yshift, size_t target_xprj, size_t loc_mem, cl_int startY, cl_int endY) {
  int err = 0;
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel zcorrection does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, xprj_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, yshift, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, target_xprj, NULL, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, loc_mem, NULL, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(startY), &startY, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(endY), &endY, &kargs);
  gcl_log_cl_fatal(err, "setting argument for zcorrection failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing zcorrection failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^denoiseSinogram_kernel)(const cl_ndrange *ndrange, cl_image prj_src, cl_image prj_dest, cl_float threshold) =
^(const cl_ndrange *ndrange, cl_image prj_src, cl_image prj_dest, cl_float threshold) {
  int err = 0;
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel denoiseSinogram does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, prj_src, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, prj_dest, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(threshold), &threshold, &kargs);
  gcl_log_cl_fatal(err, "setting argument for denoiseSinogram failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing denoiseSinogram failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/reslice.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == reslice_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "reslice", &err);
          assert(bmap.map[1].block_ptr == xprojection_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "xprojection", &err);
          assert(bmap.map[2].block_ptr == zcorrection_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "zcorrection", &err);
          assert(bmap.map[3].block_ptr == denoiseSinogram_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "denoiseSinogram", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = reslice_kernel;
  bmap.map[1].block_ptr = xprojection_kernel;
  bmap.map[2].block_ptr = zcorrection_kernel;
  bmap.map[3].block_ptr = denoiseSinogram_kernel;
}

