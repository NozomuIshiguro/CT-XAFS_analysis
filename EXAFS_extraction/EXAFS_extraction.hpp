//
//  EXAFS_extraction.hpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2017/08/09.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#ifndef EXAFS_extraction_hpp
#define EXAFS_extraction_hpp

#include "EXAFS.hpp"
#include "XANES_fitting.hpp"
#define EFF     0.262468426103175
#define PI      3.14159265358979323846


int EXAFS_extraction_ocl(input_parameter inp, OCL_platform_device plat_dev_list);
int PreEdgeRemoval(cl::CommandQueue queue, cl::Program program,
                   cl::Buffer mt_img, cl::Buffer energy, cl::Buffer bkg_img,
                   int imagesizeX, int imagesizeY, float E0, int startEn, int endEn,
                   int fitmode, float lambda, int numTrial);
int PostEdgeEstimation(cl::CommandQueue queue, cl::Program program,
                       cl::Buffer mt_img, cl::Buffer energy,
                       cl::Buffer bkg_img, cl::Buffer edgeJ_img,
                       int imagesizeX, int imagesizeY,int startEn, int endEn,
                       float lambda, int numTrial);
int SplineBkgRemoval(cl::CommandQueue queue, cl::Program program,
                     cl::Buffer mt_img, cl::Buffer energy, cl::Buffer w_factor, cl::Buffer chi_img,
                     int imagesizeX, int imagesizeY, int FFTimageSizeY, int num_energy,
                     float kstart, float kend, float Rbkg, int kw, float lambda, int numTrial, bool kendClamp);

#endif /* EXAFS_extraction_hpp */
