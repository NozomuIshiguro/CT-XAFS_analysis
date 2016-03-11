//
//  mask_class.cpp
//  Image registration share
//
//  Created by Nozomu Ishiguro on 2015/06/09.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "imageReg.hpp"

mask::mask(input_parameter inp){
    refMask_shape=inp.getRefMask_shape();
    refMask_x=inp.getRefMask_x();
    refMask_y=inp.getRefMask_y();
    refMask_width=inp.getRefMask_width();
    refMask_height=inp.getRefMask_height();
    refMask_angle=inp.getRefMask_angle();
    sampleMask_shape=inp.getSampleMask_shape();
    sampleMask_x=inp.getSampleMask_x();
    sampleMask_y=inp.getSampleMask_y();
    sampleMask_width=inp.getSampleMask_width();
    sampleMask_height=inp.getSampleMask_height();
    sampleMask_angle=inp.getSampleMask_angle();
}
