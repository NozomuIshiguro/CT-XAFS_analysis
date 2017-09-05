/***** GCL Generated File *********************/
/* Automatically generated file, do not edit! */
/**********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <dispatch/dispatch.h>
#include <OpenCL/opencl.h>
#include <OpenCL/gcl_priv.h>
#include "EXAFSextraction.cl.h"

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
void (^estimateBkg_kernel)(const cl_ndrange *ndrange, cl_float* bkg_img, cl_float* fp_img, cl_int funcmode, cl_float E0) =
^(const cl_ndrange *ndrange, cl_float* bkg_img, cl_float* fp_img, cl_int funcmode, cl_float E0) {
  int err = 0;
  cl_kernel k = bmap.map[0].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[0].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel estimateBkg does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, bkg_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, fp_img, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(funcmode), &funcmode, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(E0), &E0, &kargs);
  gcl_log_cl_fatal(err, "setting argument for estimateBkg failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing estimateBkg failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^estimateEJ_kernel)(const cl_ndrange *ndrange, cl_float* ej_img, cl_float* bkg_img, cl_float* fp_img) =
^(const cl_ndrange *ndrange, cl_float* ej_img, cl_float* bkg_img, cl_float* fp_img) {
  int err = 0;
  cl_kernel k = bmap.map[1].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[1].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel estimateEJ does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, ej_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, bkg_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, fp_img, &kargs);
  gcl_log_cl_fatal(err, "setting argument for estimateEJ failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing estimateEJ failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^redimension_mt2chi_kernel)(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* chi_img, cl_float* energy, cl_int numPnts, cl_int koffset, cl_int ksize) =
^(const cl_ndrange *ndrange, cl_float* mt_img, cl_float* chi_img, cl_float* energy, cl_int numPnts, cl_int koffset, cl_int ksize) {
  int err = 0;
  cl_kernel k = bmap.map[2].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[2].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel redimension_mt2chi does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, mt_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, chi_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, energy, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(numPnts), &numPnts, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(koffset), &koffset, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(ksize), &ksize, &kargs);
  gcl_log_cl_fatal(err, "setting argument for redimension_mt2chi failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing redimension_mt2chi failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^convert2realChi_kernel)(const cl_ndrange *ndrange, cl_float2* chi_cmplx_img, cl_float* chi_img) =
^(const cl_ndrange *ndrange, cl_float2* chi_cmplx_img, cl_float* chi_img) {
  int err = 0;
  cl_kernel k = bmap.map[3].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[3].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel convert2realChi does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, chi_cmplx_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, chi_img, &kargs);
  gcl_log_cl_fatal(err, "setting argument for convert2realChi failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing convert2realChi failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^Bspline_basis_zero_kernel)(const cl_ndrange *ndrange, cl_float* basis, cl_int N_ctrlP, cl_float zoneStart, cl_float zonePitch, cl_int basisOffset) =
^(const cl_ndrange *ndrange, cl_float* basis, cl_int N_ctrlP, cl_float zoneStart, cl_float zonePitch, cl_int basisOffset) {
  int err = 0;
  cl_kernel k = bmap.map[4].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[4].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel Bspline_basis_zero does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, basis, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, sizeof(N_ctrlP), &N_ctrlP, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(zoneStart), &zoneStart, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(zonePitch), &zonePitch, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(basisOffset), &basisOffset, &kargs);
  gcl_log_cl_fatal(err, "setting argument for Bspline_basis_zero failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing Bspline_basis_zero failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^Bspline_basis_updateOrder_kernel)(const cl_ndrange *ndrange, cl_float* basis_src, cl_float* basis_dest, cl_int N_ctrlP, cl_float zoneStart, cl_float zonePitch, cl_int basisOffset, cl_int order) =
^(const cl_ndrange *ndrange, cl_float* basis_src, cl_float* basis_dest, cl_int N_ctrlP, cl_float zoneStart, cl_float zonePitch, cl_int basisOffset, cl_int order) {
  int err = 0;
  cl_kernel k = bmap.map[5].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[5].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel Bspline_basis_updateOrder does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, basis_src, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, basis_dest, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(N_ctrlP), &N_ctrlP, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(zoneStart), &zoneStart, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(zonePitch), &zonePitch, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(basisOffset), &basisOffset, &kargs);
  err |= gclSetKernelArgAPPLE(k, 6, sizeof(order), &order, &kargs);
  gcl_log_cl_fatal(err, "setting argument for Bspline_basis_updateOrder failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing Bspline_basis_updateOrder failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^Bspline_kernel)(const cl_ndrange *ndrange, cl_float* spline, cl_float* basis, cl_float* ctrlP, cl_int N_ctrlP, cl_int basisOffset, cl_int order) =
^(const cl_ndrange *ndrange, cl_float* spline, cl_float* basis, cl_float* ctrlP, cl_int N_ctrlP, cl_int basisOffset, cl_int order) {
  int err = 0;
  cl_kernel k = bmap.map[6].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[6].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel Bspline does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, spline, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, basis, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, ctrlP, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(N_ctrlP), &N_ctrlP, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(basisOffset), &basisOffset, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(order), &order, &kargs);
  gcl_log_cl_fatal(err, "setting argument for Bspline failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing Bspline failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^BsplineRemoval_kernel)(const cl_ndrange *ndrange, cl_float2* chiData_img, cl_float2* chiFit_img, cl_float* basis, cl_float* ctrlP, cl_int N_ctrlP, cl_int basisOffset, cl_int order) =
^(const cl_ndrange *ndrange, cl_float2* chiData_img, cl_float2* chiFit_img, cl_float* basis, cl_float* ctrlP, cl_int N_ctrlP, cl_int basisOffset, cl_int order) {
  int err = 0;
  cl_kernel k = bmap.map[7].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[7].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel BsplineRemoval does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, chiData_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, chiFit_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, basis, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 3, ctrlP, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(N_ctrlP), &N_ctrlP, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(basisOffset), &basisOffset, &kargs);
  err |= gclSetKernelArgAPPLE(k, 6, sizeof(order), &order, &kargs);
  gcl_log_cl_fatal(err, "setting argument for BsplineRemoval failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing BsplineRemoval failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^Jacobian_BsplineRemoval_kernel)(const cl_ndrange *ndrange, cl_float2* J, cl_int paraID, cl_float* basis, cl_int N_ctrlP, cl_int basisOffset, cl_int order) =
^(const cl_ndrange *ndrange, cl_float2* J, cl_int paraID, cl_float* basis, cl_int N_ctrlP, cl_int basisOffset, cl_int order) {
  int err = 0;
  cl_kernel k = bmap.map[8].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[8].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel Jacobian_BsplineRemoval does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, J, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, sizeof(paraID), &paraID, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 2, basis, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(N_ctrlP), &N_ctrlP, &kargs);
  err |= gclSetKernelArgAPPLE(k, 4, sizeof(basisOffset), &basisOffset, &kargs);
  err |= gclSetKernelArgAPPLE(k, 5, sizeof(order), &order, &kargs);
  gcl_log_cl_fatal(err, "setting argument for Jacobian_BsplineRemoval failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing Jacobian_BsplineRemoval failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^kweight_kernel)(const cl_ndrange *ndrange, cl_float2* chi_img, cl_int kw) =
^(const cl_ndrange *ndrange, cl_float2* chi_img, cl_int kw) {
  int err = 0;
  cl_kernel k = bmap.map[9].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[9].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel kweight does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, chi_img, &kargs);
  err |= gclSetKernelArgAPPLE(k, 1, sizeof(kw), &kw, &kargs);
  gcl_log_cl_fatal(err, "setting argument for kweight failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing kweight failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

void (^initialCtrlP_kernel)(const cl_ndrange *ndrange, cl_float2* chiData_img, cl_float* ctrlP_img, cl_float knotPitch, cl_int ksize) =
^(const cl_ndrange *ndrange, cl_float2* chiData_img, cl_float* ctrlP_img, cl_float knotPitch, cl_int ksize) {
  int err = 0;
  cl_kernel k = bmap.map[10].kernel;
  if (!k) {
    initBlocks();
    k = bmap.map[10].kernel;
  }
  if (!k)
    gcl_log_fatal("kernel initialCtrlP does not exist for device");
  kargs_struct kargs;
  gclCreateArgsAPPLE(k, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 0, chiData_img, &kargs);
  err |= gclSetKernelArgMemAPPLE(k, 1, ctrlP_img, &kargs);
  err |= gclSetKernelArgAPPLE(k, 2, sizeof(knotPitch), &knotPitch, &kargs);
  err |= gclSetKernelArgAPPLE(k, 3, sizeof(ksize), &ksize, &kargs);
  gcl_log_cl_fatal(err, "setting argument for initialCtrlP failed");
  err = gclExecKernelAPPLE(k, ndrange, &kargs);
  gcl_log_cl_fatal(err, "Executing initialCtrlP failed");
  gclDeleteArgsAPPLE(k, &kargs);
};

// Initialization functions
static void initBlocks(void) {
  const char* build_opts = "";
  static dispatch_once_t once;
  dispatch_once(&once,
    ^{ int err = gclBuildProgramBinaryAPPLE("OpenCL/EXAFSextraction.cl", "", &bmap, build_opts);
       if (!err) {
          assert(bmap.map[0].block_ptr == estimateBkg_kernel && "mismatch block");
          bmap.map[0].kernel = clCreateKernel(bmap.program, "estimateBkg", &err);
          assert(bmap.map[1].block_ptr == estimateEJ_kernel && "mismatch block");
          bmap.map[1].kernel = clCreateKernel(bmap.program, "estimateEJ", &err);
          assert(bmap.map[2].block_ptr == redimension_mt2chi_kernel && "mismatch block");
          bmap.map[2].kernel = clCreateKernel(bmap.program, "redimension_mt2chi", &err);
          assert(bmap.map[3].block_ptr == convert2realChi_kernel && "mismatch block");
          bmap.map[3].kernel = clCreateKernel(bmap.program, "convert2realChi", &err);
          assert(bmap.map[4].block_ptr == Bspline_basis_zero_kernel && "mismatch block");
          bmap.map[4].kernel = clCreateKernel(bmap.program, "Bspline_basis_zero", &err);
          assert(bmap.map[5].block_ptr == Bspline_basis_updateOrder_kernel && "mismatch block");
          bmap.map[5].kernel = clCreateKernel(bmap.program, "Bspline_basis_updateOrder", &err);
          assert(bmap.map[6].block_ptr == Bspline_kernel && "mismatch block");
          bmap.map[6].kernel = clCreateKernel(bmap.program, "Bspline", &err);
          assert(bmap.map[7].block_ptr == BsplineRemoval_kernel && "mismatch block");
          bmap.map[7].kernel = clCreateKernel(bmap.program, "BsplineRemoval", &err);
          assert(bmap.map[8].block_ptr == Jacobian_BsplineRemoval_kernel && "mismatch block");
          bmap.map[8].kernel = clCreateKernel(bmap.program, "Jacobian_BsplineRemoval", &err);
          assert(bmap.map[9].block_ptr == kweight_kernel && "mismatch block");
          bmap.map[9].kernel = clCreateKernel(bmap.program, "kweight", &err);
          assert(bmap.map[10].block_ptr == initialCtrlP_kernel && "mismatch block");
          bmap.map[10].kernel = clCreateKernel(bmap.program, "initialCtrlP", &err);
       }
     });
}

__attribute__((constructor))
static void RegisterMap(void) {
  gclRegisterBlockKernelMap(&bmap);
  bmap.map[0].block_ptr = estimateBkg_kernel;
  bmap.map[1].block_ptr = estimateEJ_kernel;
  bmap.map[2].block_ptr = redimension_mt2chi_kernel;
  bmap.map[3].block_ptr = convert2realChi_kernel;
  bmap.map[4].block_ptr = Bspline_basis_zero_kernel;
  bmap.map[5].block_ptr = Bspline_basis_updateOrder_kernel;
  bmap.map[6].block_ptr = Bspline_kernel;
  bmap.map[7].block_ptr = BsplineRemoval_kernel;
  bmap.map[8].block_ptr = Jacobian_BsplineRemoval_kernel;
  bmap.map[9].block_ptr = kweight_kernel;
  bmap.map[10].block_ptr = initialCtrlP_kernel;
}

