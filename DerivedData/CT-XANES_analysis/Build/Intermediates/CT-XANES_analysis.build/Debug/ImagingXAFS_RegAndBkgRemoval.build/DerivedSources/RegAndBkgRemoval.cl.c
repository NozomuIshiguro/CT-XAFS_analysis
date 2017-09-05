/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "RegAndBkgRemoval.cl.h"

static void initBlocks(void);

// Initialize static data structures
static block_kernel_pair pair_map[6] = {
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL },
      { NULL, NULL }
};

static block_kernel_map bmap = { 0, 6, initBlocks, pair_map };

// Block function
void (^estimateBkgImg_kernel)(const cl_ndrange *ndrange, cl_image bkg_img, cl_image mt_data_img, cl_image grid_img, cl_image sample_img, cl_float* para, cl_float* contrast, cl_float* weight) =
^(const cl_ndrange *ndrange, cl_image bkg_img, cl_image mt_data_img, cl_image grid_img, cl_image sample_img, cl_float* para, cl_float* contrast, cl_float* weight) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel estimateBkgImg does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, bkg_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_data_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, grid_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, sample_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, para, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 5, contrast, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 6, weight, &kargs);
  gcl_log_cl_fatal(err, "setting argument for estimateBkgImg failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing estimateBkgImg failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^estimateResidueImg_kernel)(const cl_ndrange *ndrange, cl_image residue_img, cl_image mt_data_img, cl_image bkg_img) =
^(const cl_ndrange *ndrange, cl_image residue_img, cl_image mt_data_img, cl_image bkg_img) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel estimateResidueImg does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, residue_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, mt_data_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, bkg_img, &kargs);
  gcl_log_cl_fatal(err, "setting argument for estimateResidueImg failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing estimateResidueImg failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^estimateGridImg_kernel)(const cl_ndrange *ndrange, cl_image grid_img, cl_image residue_reg_img, cl_image sample_img, cl_float* contrast, cl_float* weight) =
^(const cl_ndrange *ndrange, cl_image grid_img, cl_image residue_reg_img, cl_image sample_img, cl_float* contrast, cl_float* weight) {
  int err = 0;
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel estimateGridImg does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, grid_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, residue_reg_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, sample_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, contrast, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 4, weight, &kargs);
  gcl_log_cl_fatal(err, "setting argument for estimateGridImg failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing estimateGridImg failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^estimateSampleImg_kernel)(const cl_ndrange *ndrange, cl_image sample_img, cl_image residue_reg_img, cl_image grid_img) =
^(const cl_ndrange *ndrange, cl_image sample_img, cl_image residue_reg_img, cl_image grid_img) {
  int err = 0;
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel estimateSampleImg does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, sample_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, residue_reg_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, grid_img, &kargs);
  gcl_log_cl_fatal(err, "setting argument for estimateSampleImg failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing estimateSampleImg failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^estimateSampleContrast1_kernel)(const cl_ndrange *ndrange, cl_image residue_reg_img, cl_image grid_img, cl_image sample_img, cl_float* sum, size_t loc_mem) =
^(const cl_ndrange *ndrange, cl_image residue_reg_img, cl_image grid_img, cl_image sample_img, cl_float* sum, size_t loc_mem) {
  int err = 0;
  cl_kernel k = bmap.map[4].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[4].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel estimateSampleContrast1 does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, residue_reg_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, grid_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, sample_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, sum, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, loc_mem, NULL, &kargs);
  gcl_log_cl_fatal(err, "setting argument for estimateSampleContrast1 failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing estimateSampleContrast1 failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^estimateSampleContrast2_kernel)(const cl_ndrange *ndrange, cl_float* sum, cl_float* contrast, size_t loc_mem) =
^(const cl_ndrange *ndrange, cl_float* sum, cl_float* contrast, size_t loc_mem) {
  int err = 0;
  cl_kernel k = bmap.map[5].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[5].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel estimateSampleContrast2 does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, sum, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, contrast, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, loc_mem, NULL, &kargs);
  gcl_log_cl_fatal(err, "setting argument for estimateSampleContrast2 failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing estimateSampleContrast2 failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/RegAndBkgRemoval.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == estimateBkgImg_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "estimateBkgImg", &err);
          assert(bmap.map[1].block_ptr == estimateResidueImg_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "estimateResidueImg", &err);
          assert(bmap.map[2].block_ptr == estimateGridImg_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "estimateGridImg", &err);
          assert(bmap.map[3].block_ptr == estimateSampleImg_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "estimateSampleImg", &err);
          assert(bmap.map[4].block_ptr == estimateSampleContrast1_kernel && "mismatch block");
          bmap.map[4].kernel = clCreateKernel(bmap.program, "estimateSampleContrast1", &err);
          assert(bmap.map[5].block_ptr == estimateSampleContrast2_kernel && "mismatch block");
          bmap.map[5].kernel = clCreateKernel(bmap.program, "estimateSampleContrast2", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = estimateBkgImg_kernel;
  bmap.map[1].block_ptr = estimateResidueImg_kernel;
  bmap.map[2].block_ptr = estimateGridImg_kernel;
  bmap.map[3].block_ptr = estimateSampleImg_kernel;
  bmap.map[4].block_ptr = estimateSampleContrast1_kernel;
  bmap.map[5].block_ptr = estimateSampleContrast2_kernel;
}

