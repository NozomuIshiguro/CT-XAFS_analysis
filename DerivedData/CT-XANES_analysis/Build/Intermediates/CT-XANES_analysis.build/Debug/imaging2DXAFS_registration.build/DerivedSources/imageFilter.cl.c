/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "imageFilter.cl.h"

static void initBlocks(void);

// Initialize static data structures
static block_kernel_pair pair_map[11] = {
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL }
};

static block_kernel_map bmap = { 0, 11, initBlocks, pair_map };

// Block function
void (^meanImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_uint radius) =
^(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_uint radius) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel meanImg does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img1, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_img2, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(radius), &radius, &kargs);
  gcl_log_cl_fatal(err, "setting argument for meanImg failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing meanImg failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^addImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float val) =
^(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float val) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel addImg does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img1, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_img2, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(val), &val, &kargs);
  gcl_log_cl_fatal(err, "setting argument for addImg failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing addImg failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^subtractImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float val) =
^(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float val) {
  int err = 0;
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel subtractImg does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img1, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_img2, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(val), &val, &kargs);
  gcl_log_cl_fatal(err, "setting argument for subtractImg failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing subtractImg failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^MultiplyImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float val) =
^(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float val) {
  int err = 0;
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel MultiplyImg does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img1, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_img2, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(val), &val, &kargs);
  gcl_log_cl_fatal(err, "setting argument for MultiplyImg failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing MultiplyImg failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^DivideImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float val) =
^(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float val) {
  int err = 0;
  cl_kernel k = bmap.map[4].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[4].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel DivideImg does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img1, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_img2, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(val), &val, &kargs);
  gcl_log_cl_fatal(err, "setting argument for DivideImg failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing DivideImg failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^RemoveNANImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float val) =
^(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float val) {
  int err = 0;
  cl_kernel k = bmap.map[5].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[5].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel RemoveNANImg does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img1, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_img2, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(val), &val, &kargs);
  gcl_log_cl_fatal(err, "setting argument for RemoveNANImg failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing RemoveNANImg failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^MinImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float minval) =
^(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float minval) {
  int err = 0;
  cl_kernel k = bmap.map[6].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[6].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel MinImg does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img1, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_img2, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(minval), &minval, &kargs);
  gcl_log_cl_fatal(err, "setting argument for MinImg failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing MinImg failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^MaxImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float maxval) =
^(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2, cl_float maxval) {
  int err = 0;
  cl_kernel k = bmap.map[7].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[7].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel MaxImg does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img1, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_img2, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(maxval), &maxval, &kargs);
  gcl_log_cl_fatal(err, "setting argument for MaxImg failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing MaxImg failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^expImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2) =
^(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2) {
  int err = 0;
  cl_kernel k = bmap.map[8].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[8].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel expImg does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img1, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_img2, &kargs);
  gcl_log_cl_fatal(err, "setting argument for expImg failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing expImg failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^lnImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2) =
^(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2) {
  int err = 0;
  cl_kernel k = bmap.map[9].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[9].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel lnImg does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img1, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_img2, &kargs);
  gcl_log_cl_fatal(err, "setting argument for lnImg failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing lnImg failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^rapizoralImg_kernel)(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2) =
^(const cl_ndrange *ndrange, cl_image mt_img1, cl_image mt_img2) {
  int err = 0;
  cl_kernel k = bmap.map[10].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[10].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel rapizoralImg does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img1, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_img2, &kargs);
  gcl_log_cl_fatal(err, "setting argument for rapizoralImg failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing rapizoralImg failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/imageFilter.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == meanImg_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "meanImg", &err);
          assert(bmap.map[1].block_ptr == addImg_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "addImg", &err);
          assert(bmap.map[2].block_ptr == subtractImg_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "subtractImg", &err);
          assert(bmap.map[3].block_ptr == MultiplyImg_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "MultiplyImg", &err);
          assert(bmap.map[4].block_ptr == DivideImg_kernel && "mismatch block");
          bmap.map[4].kernel = clCreateKernel(bmap.program, "DivideImg", &err);
          assert(bmap.map[5].block_ptr == RemoveNANImg_kernel && "mismatch block");
          bmap.map[5].kernel = clCreateKernel(bmap.program, "RemoveNANImg", &err);
          assert(bmap.map[6].block_ptr == MinImg_kernel && "mismatch block");
          bmap.map[6].kernel = clCreateKernel(bmap.program, "MinImg", &err);
          assert(bmap.map[7].block_ptr == MaxImg_kernel && "mismatch block");
          bmap.map[7].kernel = clCreateKernel(bmap.program, "MaxImg", &err);
          assert(bmap.map[8].block_ptr == expImg_kernel && "mismatch block");
          bmap.map[8].kernel = clCreateKernel(bmap.program, "expImg", &err);
          assert(bmap.map[9].block_ptr == lnImg_kernel && "mismatch block");
          bmap.map[9].kernel = clCreateKernel(bmap.program, "lnImg", &err);
          assert(bmap.map[10].block_ptr == rapizoralImg_kernel && "mismatch block");
          bmap.map[10].kernel = clCreateKernel(bmap.program, "rapizoralImg", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = meanImg_kernel;
  bmap.map[1].block_ptr = addImg_kernel;
  bmap.map[2].block_ptr = subtractImg_kernel;
  bmap.map[3].block_ptr = MultiplyImg_kernel;
  bmap.map[4].block_ptr = DivideImg_kernel;
  bmap.map[5].block_ptr = RemoveNANImg_kernel;
  bmap.map[6].block_ptr = MinImg_kernel;
  bmap.map[7].block_ptr = MaxImg_kernel;
  bmap.map[8].block_ptr = expImg_kernel;
  bmap.map[9].block_ptr = lnImg_kernel;
  bmap.map[10].block_ptr = rapizoralImg_kernel;
}

