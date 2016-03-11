//
//  XYshift_rotation_cl.hpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/03/21.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#ifndef CT_XANES_analysis_XYshift_rotation_cl_hpp
#define CT_XANES_analysis_XYshift_rotation_cl_hpp

string kernel_src_XYrot = {
    0x23, 0x64, 0x65, 0x66, 0x69, 0x6e, 0x65, 0x20, 0x4e, 0x55, 0x4d, 0x5f,
    0x54, 0x52, 0x41, 0x4e, 0x53, 0x50, 0x41, 0x52, 0x41, 0x20, 0x33, 0x0a,
    0x23, 0x64, 0x65, 0x66, 0x69, 0x6e, 0x65, 0x20, 0x4e, 0x55, 0x4d, 0x5f,
    0x52, 0x45, 0x44, 0x55, 0x43, 0x54, 0x49, 0x4f, 0x4e, 0x20, 0x31, 0x30,
    0x0a, 0x0a, 0x0a, 0x0a, 0x2f, 0x2f, 0x63, 0x6f, 0x70, 0x79, 0x20, 0x64,
    0x61, 0x74, 0x61, 0x20, 0x66, 0x72, 0x6f, 0x6d, 0x20, 0x67, 0x6c, 0x6f,
    0x62, 0x61, 0x6c, 0x20, 0x74, 0x72, 0x61, 0x6e, 0x73, 0x70, 0x61, 0x72,
    0x61, 0x20, 0x74, 0x6f, 0x20, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x74,
    0x72, 0x61, 0x6e, 0x73, 0x70, 0x61, 0x72, 0x61, 0x5f, 0x61, 0x74, 0x45,
    0x0a, 0x23, 0x64, 0x65, 0x66, 0x69, 0x6e, 0x65, 0x20, 0x54, 0x52, 0x41,
    0x4e, 0x53, 0x50, 0x41, 0x52, 0x41, 0x5f, 0x4c, 0x4f, 0x43, 0x5f, 0x43,
    0x4f, 0x50, 0x59, 0x28, 0x74, 0x70, 0x2c, 0x74, 0x70, 0x67, 0x2c, 0x6c,
    0x69, 0x64, 0x2c, 0x67, 0x69, 0x64, 0x29, 0x5c, 0x0a, 0x7b, 0x5c, 0x0a,
    0x69, 0x66, 0x28, 0x28, 0x6c, 0x69, 0x64, 0x29, 0x3d, 0x3d, 0x30, 0x29,
    0x7b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c,
    0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x74, 0x70, 0x29,
    0x29, 0x5b, 0x30, 0x5d, 0x3d, 0x28, 0x28, 0x5f, 0x5f, 0x67, 0x6c, 0x6f,
    0x62, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28,
    0x74, 0x70, 0x67, 0x29, 0x29, 0x5b, 0x67, 0x69, 0x64, 0x2a, 0x33, 0x5d,
    0x3b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c,
    0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x74, 0x70, 0x29,
    0x29, 0x5b, 0x31, 0x5d, 0x3d, 0x28, 0x28, 0x5f, 0x5f, 0x67, 0x6c, 0x6f,
    0x62, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28,
    0x74, 0x70, 0x67, 0x29, 0x29, 0x5b, 0x67, 0x69, 0x64, 0x2a, 0x33, 0x2b,
    0x31, 0x5d, 0x3b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63,
    0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x74,
    0x70, 0x29, 0x29, 0x5b, 0x32, 0x5d, 0x3d, 0x28, 0x28, 0x5f, 0x5f, 0x67,
    0x6c, 0x6f, 0x62, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a,
    0x29, 0x28, 0x74, 0x70, 0x67, 0x29, 0x29, 0x5b, 0x67, 0x69, 0x64, 0x2a,
    0x33, 0x2b, 0x32, 0x5d, 0x3b, 0x5c, 0x0a, 0x7d, 0x5c, 0x0a, 0x62, 0x61,
    0x72, 0x72, 0x69, 0x65, 0x72, 0x28, 0x43, 0x4c, 0x4b, 0x5f, 0x4c, 0x4f,
    0x43, 0x41, 0x4c, 0x5f, 0x4d, 0x45, 0x4d, 0x5f, 0x46, 0x45, 0x4e, 0x43,
    0x45, 0x7c, 0x43, 0x4c, 0x4b, 0x5f, 0x47, 0x4c, 0x4f, 0x42, 0x41, 0x4c,
    0x5f, 0x4d, 0x45, 0x4d, 0x5f, 0x46, 0x45, 0x4e, 0x43, 0x45, 0x29, 0x3b,
    0x5c, 0x0a, 0x7d, 0x0a, 0x0a, 0x0a, 0x0a, 0x2f, 0x2f, 0x74, 0x72, 0x61,
    0x6e, 0x73, 0x66, 0x6f, 0x72, 0x6d, 0x20, 0x58, 0x59, 0x0a, 0x23, 0x64,
    0x65, 0x66, 0x69, 0x6e, 0x65, 0x20, 0x54, 0x52, 0x41, 0x4e, 0x53, 0x5f,
    0x58, 0x59, 0x28, 0x58, 0x59, 0x2c, 0x74, 0x2c, 0x78, 0x2c, 0x79, 0x29,
    0x5c, 0x0a, 0x28, 0x58, 0x59, 0x29, 0x20, 0x3d, 0x20, 0x28, 0x66, 0x6c,
    0x6f, 0x61, 0x74, 0x32, 0x29, 0x28, 0x63, 0x6f, 0x73, 0x28, 0x28, 0x28,
    0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61,
    0x74, 0x2a, 0x29, 0x28, 0x74, 0x29, 0x29, 0x5b, 0x32, 0x5d, 0x29, 0x2a,
    0x28, 0x78, 0x29, 0x5c, 0x0a, 0x2d, 0x73, 0x69, 0x6e, 0x28, 0x28, 0x28,
    0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61,
    0x74, 0x2a, 0x29, 0x28, 0x74, 0x29, 0x29, 0x5b, 0x32, 0x5d, 0x29, 0x2a,
    0x28, 0x79, 0x29, 0x5c, 0x0a, 0x2b, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f,
    0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28,
    0x74, 0x29, 0x29, 0x5b, 0x30, 0x5d, 0x2c, 0x5c, 0x0a, 0x73, 0x69, 0x6e,
    0x28, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66,
    0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x74, 0x29, 0x29, 0x5b, 0x32,
    0x5d, 0x29, 0x2a, 0x28, 0x78, 0x29, 0x5c, 0x0a, 0x2b, 0x63, 0x6f, 0x73,
    0x28, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66,
    0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x74, 0x29, 0x29, 0x5b, 0x32,
    0x5d, 0x29, 0x2a, 0x28, 0x79, 0x29, 0x5c, 0x0a, 0x2b, 0x28, 0x28, 0x5f,
    0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74,
    0x2a, 0x29, 0x28, 0x74, 0x29, 0x29, 0x5b, 0x31, 0x5d, 0x29, 0x0a, 0x0a,
    0x0a, 0x0a, 0x2f, 0x2f, 0x4c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x6d, 0x65,
    0x6d, 0x6f, 0x72, 0x79, 0x20, 0x28, 0x66, 0x6f, 0x72, 0x20, 0x72, 0x65,
    0x64, 0x75, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x29, 0x20, 0x72, 0x65, 0x73,
    0x65, 0x74, 0x0a, 0x23, 0x64, 0x65, 0x66, 0x69, 0x6e, 0x65, 0x20, 0x4c,
    0x4f, 0x43, 0x4d, 0x45, 0x4d, 0x5f, 0x52, 0x45, 0x53, 0x45, 0x54, 0x28,
    0x6c, 0x2c, 0x6c, 0x69, 0x64, 0x2c, 0x6c, 0x73, 0x7a, 0x29, 0x5c, 0x0a,
    0x7b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c,
    0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6c, 0x29, 0x29,
    0x5b, 0x6c, 0x69, 0x64, 0x5d, 0x3d, 0x30, 0x3b, 0x5c, 0x0a, 0x28, 0x28,
    0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61,
    0x74, 0x2a, 0x29, 0x28, 0x6c, 0x29, 0x29, 0x5b, 0x6c, 0x69, 0x64, 0x2b,
    0x6c, 0x73, 0x7a, 0x5d, 0x3d, 0x30, 0x3b, 0x5c, 0x0a, 0x28, 0x28, 0x5f,
    0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74,
    0x2a, 0x29, 0x28, 0x6c, 0x29, 0x29, 0x5b, 0x6c, 0x69, 0x64, 0x2b, 0x6c,
    0x73, 0x7a, 0x2a, 0x32, 0x5d, 0x3d, 0x30, 0x3b, 0x5c, 0x0a, 0x28, 0x28,
    0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61,
    0x74, 0x2a, 0x29, 0x28, 0x6c, 0x29, 0x29, 0x5b, 0x6c, 0x69, 0x64, 0x2b,
    0x6c, 0x73, 0x7a, 0x2a, 0x33, 0x5d, 0x3d, 0x30, 0x3b, 0x5c, 0x0a, 0x28,
    0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f,
    0x61, 0x74, 0x2a, 0x29, 0x28, 0x6c, 0x29, 0x29, 0x5b, 0x6c, 0x69, 0x64,
    0x2b, 0x6c, 0x73, 0x7a, 0x2a, 0x34, 0x5d, 0x3d, 0x30, 0x3b, 0x5c, 0x0a,
    0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c,
    0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6c, 0x29, 0x29, 0x5b, 0x6c, 0x69,
    0x64, 0x2b, 0x6c, 0x73, 0x7a, 0x2a, 0x35, 0x5d, 0x3d, 0x30, 0x3b, 0x5c,
    0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66,
    0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6c, 0x29, 0x29, 0x5b, 0x6c,
    0x69, 0x64, 0x2b, 0x6c, 0x73, 0x7a, 0x2a, 0x36, 0x5d, 0x3d, 0x30, 0x3b,
    0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20,
    0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6c, 0x29, 0x29, 0x5b,
    0x6c, 0x69, 0x64, 0x2b, 0x6c, 0x73, 0x7a, 0x2a, 0x37, 0x5d, 0x3d, 0x30,
    0x3b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c,
    0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6c, 0x29, 0x29,
    0x5b, 0x6c, 0x69, 0x64, 0x2b, 0x6c, 0x73, 0x7a, 0x2a, 0x38, 0x5d, 0x3d,
    0x30, 0x3b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61,
    0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6c, 0x29,
    0x29, 0x5b, 0x6c, 0x69, 0x64, 0x2b, 0x6c, 0x73, 0x7a, 0x2a, 0x39, 0x5d,
    0x3d, 0x30, 0x3b, 0x5c, 0x0a, 0x7d, 0x0a, 0x0a, 0x0a, 0x2f, 0x2f, 0x72,
    0x65, 0x73, 0x65, 0x74, 0x20, 0x72, 0x65, 0x64, 0x75, 0x63, 0x74, 0x69,
    0x6f, 0x6e, 0x20, 0x72, 0x65, 0x73, 0x75, 0x6c, 0x74, 0x73, 0x0a, 0x23,
    0x64, 0x65, 0x66, 0x69, 0x6e, 0x65, 0x20, 0x52, 0x45, 0x44, 0x55, 0x43,
    0x54, 0x50, 0x41, 0x52, 0x41, 0x5f, 0x52, 0x45, 0x53, 0x45, 0x54, 0x28,
    0x72, 0x70, 0x29, 0x5c, 0x0a, 0x7b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f,
    0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a,
    0x29, 0x28, 0x72, 0x70, 0x29, 0x29, 0x5b, 0x30, 0x5d, 0x3d, 0x30, 0x3b,
    0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20,
    0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x72, 0x70, 0x29, 0x29,
    0x5b, 0x31, 0x5d, 0x3d, 0x30, 0x3b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f,
    0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a,
    0x29, 0x28, 0x72, 0x70, 0x29, 0x29, 0x5b, 0x32, 0x5d, 0x3d, 0x30, 0x3b,
    0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20,
    0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x72, 0x70, 0x29, 0x29,
    0x5b, 0x33, 0x5d, 0x3d, 0x30, 0x3b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f,
    0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a,
    0x29, 0x28, 0x72, 0x70, 0x29, 0x29, 0x5b, 0x34, 0x5d, 0x3d, 0x30, 0x3b,
    0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20,
    0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x72, 0x70, 0x29, 0x29,
    0x5b, 0x35, 0x5d, 0x3d, 0x30, 0x3b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f,
    0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a,
    0x29, 0x28, 0x72, 0x70, 0x29, 0x29, 0x5b, 0x36, 0x5d, 0x3d, 0x30, 0x3b,
    0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20,
    0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x72, 0x70, 0x29, 0x29,
    0x5b, 0x37, 0x5d, 0x3d, 0x30, 0x3b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f,
    0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a,
    0x29, 0x28, 0x72, 0x70, 0x29, 0x29, 0x5b, 0x38, 0x5d, 0x3d, 0x30, 0x3b,
    0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20,
    0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x72, 0x70, 0x29, 0x29,
    0x5b, 0x39, 0x5d, 0x3d, 0x30, 0x3b, 0x5c, 0x0a, 0x7d, 0x0a, 0x0a, 0x0a,
    0x0a, 0x2f, 0x2f, 0x63, 0x61, 0x6c, 0x63, 0x75, 0x6c, 0x61, 0x74, 0x65,
    0x20, 0x4a, 0x61, 0x63, 0x6f, 0x62, 0x69, 0x61, 0x6e, 0x0a, 0x23, 0x64,
    0x65, 0x66, 0x69, 0x6e, 0x65, 0x20, 0x4a, 0x41, 0x43, 0x4f, 0x42, 0x49,
    0x41, 0x4e, 0x28, 0x74, 0x2c, 0x78, 0x2c, 0x79, 0x2c, 0x6a, 0x2c, 0x64,
    0x78, 0x2c, 0x64, 0x79, 0x2c, 0x6d, 0x73, 0x6b, 0x2c, 0x6d, 0x73, 0x2c,
    0x6d, 0x6c, 0x29, 0x5c, 0x0a, 0x7b, 0x5c, 0x0a, 0x66, 0x6c, 0x6f, 0x61,
    0x74, 0x20, 0x44, 0x66, 0x78, 0x44, 0x74, 0x68, 0x20, 0x3d, 0x20, 0x28,
    0x2d, 0x28, 0x78, 0x29, 0x2a, 0x73, 0x69, 0x6e, 0x28, 0x28, 0x28, 0x5f,
    0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74,
    0x2a, 0x29, 0x28, 0x74, 0x29, 0x29, 0x5b, 0x32, 0x5d, 0x29, 0x2d, 0x28,
    0x79, 0x29, 0x2a, 0x63, 0x6f, 0x73, 0x28, 0x28, 0x28, 0x5f, 0x5f, 0x6c,
    0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29,
    0x28, 0x74, 0x29, 0x29, 0x5b, 0x32, 0x5d, 0x29, 0x29, 0x3b, 0x5c, 0x0a,
    0x66, 0x6c, 0x6f, 0x61, 0x74, 0x20, 0x44, 0x66, 0x79, 0x44, 0x74, 0x68,
    0x20, 0x3d, 0x20, 0x28, 0x28, 0x78, 0x29, 0x2a, 0x63, 0x6f, 0x73, 0x28,
    0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c,
    0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x74, 0x29, 0x29, 0x5b, 0x32, 0x5d,
    0x29, 0x2d, 0x28, 0x79, 0x29, 0x2a, 0x73, 0x69, 0x6e, 0x28, 0x28, 0x28,
    0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61,
    0x74, 0x2a, 0x29, 0x28, 0x74, 0x29, 0x29, 0x5b, 0x32, 0x5d, 0x29, 0x29,
    0x3b, 0x5c, 0x0a, 0x62, 0x61, 0x72, 0x72, 0x69, 0x65, 0x72, 0x28, 0x43,
    0x4c, 0x4b, 0x5f, 0x4c, 0x4f, 0x43, 0x41, 0x4c, 0x5f, 0x4d, 0x45, 0x4d,
    0x5f, 0x46, 0x45, 0x4e, 0x43, 0x45, 0x29, 0x3b, 0x5c, 0x0a, 0x5c, 0x0a,
    0x28, 0x28, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6a, 0x29,
    0x29, 0x5b, 0x30, 0x5d, 0x20, 0x3d, 0x20, 0x28, 0x64, 0x78, 0x29, 0x2a,
    0x28, 0x6d, 0x73, 0x6b, 0x29, 0x2f, 0x28, 0x6d, 0x73, 0x29, 0x2f, 0x28,
    0x6d, 0x73, 0x29, 0x2a, 0x28, 0x6d, 0x6c, 0x29, 0x3b, 0x5c, 0x0a, 0x28,
    0x28, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6a, 0x29, 0x29,
    0x5b, 0x31, 0x5d, 0x20, 0x3d, 0x20, 0x28, 0x64, 0x79, 0x29, 0x2a, 0x28,
    0x6d, 0x73, 0x6b, 0x29, 0x2f, 0x28, 0x6d, 0x73, 0x29, 0x2f, 0x28, 0x6d,
    0x73, 0x29, 0x2a, 0x28, 0x6d, 0x6c, 0x29, 0x3b, 0x5c, 0x0a, 0x28, 0x28,
    0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6a, 0x29, 0x29, 0x5b,
    0x32, 0x5d, 0x20, 0x3d, 0x20, 0x28, 0x28, 0x64, 0x78, 0x29, 0x2a, 0x44,
    0x66, 0x78, 0x44, 0x74, 0x68, 0x2b, 0x28, 0x64, 0x79, 0x29, 0x2a, 0x44,
    0x66, 0x78, 0x44, 0x74, 0x68, 0x29, 0x2a, 0x28, 0x6d, 0x6c, 0x29, 0x3b,
    0x5c, 0x0a, 0x7d, 0x0a, 0x0a, 0x0a, 0x0a, 0x2f, 0x2f, 0x63, 0x6f, 0x70,
    0x79, 0x20, 0x4a, 0x61, 0x63, 0x6f, 0x62, 0x69, 0x61, 0x6e, 0x20, 0x26,
    0x20, 0x64, 0x69, 0x6d, 0x67, 0x20, 0x74, 0x6f, 0x20, 0x6c, 0x6f, 0x63,
    0x61, 0x6c, 0x20, 0x6d, 0x65, 0x6d, 0x6f, 0x72, 0x79, 0x0a, 0x23, 0x64,
    0x65, 0x66, 0x69, 0x6e, 0x65, 0x20, 0x4c, 0x4f, 0x43, 0x4d, 0x45, 0x4d,
    0x5f, 0x43, 0x4f, 0x50, 0x59, 0x28, 0x6c, 0x2c, 0x6a, 0x2c, 0x64, 0x69,
    0x6d, 0x67, 0x2c, 0x70, 0x2c, 0x6c, 0x69, 0x64, 0x2c, 0x6c, 0x73, 0x7a,
    0x29, 0x5c, 0x0a, 0x7b, 0x5c, 0x0a, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x20,
    0x65, 0x5b, 0x39, 0x5d, 0x3b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c,
    0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29,
    0x28, 0x6c, 0x29, 0x29, 0x5b, 0x6c, 0x69, 0x64, 0x5d, 0x2b, 0x3d, 0x28,
    0x28, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6a, 0x29, 0x29,
    0x5b, 0x30, 0x5d, 0x2a, 0x28, 0x28, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a,
    0x29, 0x28, 0x6a, 0x29, 0x29, 0x5b, 0x30, 0x5d, 0x3b, 0x5c, 0x0a, 0x28,
    0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f,
    0x61, 0x74, 0x2a, 0x29, 0x28, 0x6c, 0x29, 0x29, 0x5b, 0x6c, 0x69, 0x64,
    0x2b, 0x6c, 0x73, 0x7a, 0x5d, 0x2b, 0x3d, 0x28, 0x28, 0x66, 0x6c, 0x6f,
    0x61, 0x74, 0x2a, 0x29, 0x28, 0x6a, 0x29, 0x29, 0x5b, 0x31, 0x5d, 0x2a,
    0x28, 0x28, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6a, 0x29,
    0x29, 0x5b, 0x30, 0x5d, 0x3b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c,
    0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29,
    0x28, 0x6c, 0x29, 0x29, 0x5b, 0x6c, 0x69, 0x64, 0x2b, 0x6c, 0x73, 0x7a,
    0x2a, 0x32, 0x5d, 0x2b, 0x3d, 0x28, 0x28, 0x66, 0x6c, 0x6f, 0x61, 0x74,
    0x2a, 0x29, 0x28, 0x6a, 0x29, 0x29, 0x5b, 0x32, 0x5d, 0x2a, 0x28, 0x28,
    0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6a, 0x29, 0x29, 0x5b,
    0x30, 0x5d, 0x3b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63,
    0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6c,
    0x29, 0x29, 0x5b, 0x6c, 0x69, 0x64, 0x2b, 0x6c, 0x73, 0x7a, 0x2a, 0x33,
    0x5d, 0x2b, 0x3d, 0x28, 0x28, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29,
    0x28, 0x6a, 0x29, 0x29, 0x5b, 0x31, 0x5d, 0x2a, 0x28, 0x28, 0x66, 0x6c,
    0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6a, 0x29, 0x29, 0x5b, 0x30, 0x5d,
    0x3b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c,
    0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6c, 0x29, 0x29,
    0x5b, 0x6c, 0x69, 0x64, 0x2b, 0x6c, 0x73, 0x7a, 0x2a, 0x34, 0x5d, 0x2b,
    0x3d, 0x28, 0x28, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6a,
    0x29, 0x29, 0x5b, 0x32, 0x5d, 0x2a, 0x28, 0x28, 0x66, 0x6c, 0x6f, 0x61,
    0x74, 0x2a, 0x29, 0x28, 0x6a, 0x29, 0x29, 0x5b, 0x30, 0x5d, 0x3b, 0x5c,
    0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66,
    0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6c, 0x29, 0x29, 0x5b, 0x6c,
    0x69, 0x64, 0x2b, 0x6c, 0x73, 0x7a, 0x2a, 0x35, 0x5d, 0x2b, 0x3d, 0x28,
    0x28, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6a, 0x29, 0x29,
    0x5b, 0x32, 0x5d, 0x2a, 0x28, 0x28, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a,
    0x29, 0x28, 0x6a, 0x29, 0x29, 0x5b, 0x32, 0x5d, 0x3b, 0x5c, 0x0a, 0x28,
    0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f,
    0x61, 0x74, 0x2a, 0x29, 0x28, 0x6c, 0x29, 0x29, 0x5b, 0x6c, 0x69, 0x64,
    0x2b, 0x6c, 0x73, 0x7a, 0x2a, 0x36, 0x5d, 0x2b, 0x3d, 0x28, 0x28, 0x66,
    0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6a, 0x29, 0x29, 0x5b, 0x30,
    0x5d, 0x2a, 0x28, 0x64, 0x69, 0x6d, 0x67, 0x29, 0x3b, 0x5c, 0x0a, 0x28,
    0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f,
    0x61, 0x74, 0x2a, 0x29, 0x28, 0x6c, 0x29, 0x29, 0x5b, 0x6c, 0x69, 0x64,
    0x2b, 0x6c, 0x73, 0x7a, 0x2a, 0x37, 0x5d, 0x2b, 0x3d, 0x28, 0x28, 0x66,
    0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6a, 0x29, 0x29, 0x5b, 0x31,
    0x5d, 0x2a, 0x28, 0x64, 0x69, 0x6d, 0x67, 0x29, 0x3b, 0x5c, 0x0a, 0x28,
    0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f,
    0x61, 0x74, 0x2a, 0x29, 0x28, 0x6c, 0x29, 0x29, 0x5b, 0x6c, 0x69, 0x64,
    0x2b, 0x6c, 0x73, 0x7a, 0x2a, 0x38, 0x5d, 0x2b, 0x3d, 0x28, 0x28, 0x66,
    0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6a, 0x29, 0x29, 0x5b, 0x32,
    0x5d, 0x2a, 0x28, 0x64, 0x69, 0x6d, 0x67, 0x29, 0x3b, 0x5c, 0x0a, 0x28,
    0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f,
    0x61, 0x74, 0x2a, 0x29, 0x28, 0x6c, 0x29, 0x29, 0x5b, 0x6c, 0x69, 0x64,
    0x2b, 0x6c, 0x73, 0x7a, 0x2a, 0x39, 0x5d, 0x2b, 0x3d, 0x28, 0x70, 0x29,
    0x3b, 0x5c, 0x0a, 0x62, 0x61, 0x72, 0x72, 0x69, 0x65, 0x72, 0x28, 0x43,
    0x4c, 0x4b, 0x5f, 0x4c, 0x4f, 0x43, 0x41, 0x4c, 0x5f, 0x4d, 0x45, 0x4d,
    0x5f, 0x46, 0x45, 0x4e, 0x43, 0x45, 0x29, 0x3b, 0x5c, 0x0a, 0x7d, 0x0a,
    0x0a, 0x0a, 0x0a, 0x2f, 0x2f, 0x72, 0x65, 0x64, 0x75, 0x63, 0x74, 0x69,
    0x6f, 0x6e, 0x0a, 0x23, 0x64, 0x65, 0x66, 0x69, 0x6e, 0x65, 0x20, 0x52,
    0x45, 0x44, 0x55, 0x43, 0x54, 0x49, 0x4f, 0x4e, 0x28, 0x6c, 0x2c, 0x72,
    0x70, 0x29, 0x20, 0x72, 0x65, 0x64, 0x75, 0x63, 0x74, 0x69, 0x6f, 0x6e,
    0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c,
    0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x6c, 0x29, 0x2c, 0x28, 0x5f, 0x5f,
    0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a,
    0x29, 0x28, 0x72, 0x70, 0x29, 0x2c, 0x28, 0x31, 0x30, 0x29, 0x29, 0x3b,
    0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x2f, 0x2f, 0x63, 0x61, 0x6c, 0x63, 0x75,
    0x6c, 0x61, 0x74, 0x65, 0x20, 0x64, 0x65, 0x6c, 0x74, 0x61, 0x5f, 0x74,
    0x72, 0x61, 0x6e, 0x73, 0x70, 0x61, 0x72, 0x61, 0x0a, 0x23, 0x64, 0x65,
    0x66, 0x69, 0x6e, 0x65, 0x20, 0x43, 0x41, 0x4c, 0x43, 0x5f, 0x44, 0x45,
    0x4c, 0x54, 0x41, 0x5f, 0x54, 0x52, 0x41, 0x4e, 0x53, 0x50, 0x41, 0x52,
    0x41, 0x28, 0x72, 0x70, 0x2c, 0x6c, 0x2c, 0x64, 0x74, 0x70, 0x2c, 0x64,
    0x72, 0x2c, 0x6c, 0x69, 0x64, 0x29, 0x20, 0x5c, 0x0a, 0x7b, 0x5c, 0x0a,
    0x66, 0x6c, 0x6f, 0x61, 0x74, 0x20, 0x74, 0x4a, 0x4a, 0x5b, 0x33, 0x5d,
    0x5b, 0x33, 0x5d, 0x3b, 0x5c, 0x0a, 0x74, 0x4a, 0x4a, 0x5b, 0x30, 0x5d,
    0x5b, 0x30, 0x5d, 0x3d, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61,
    0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x72, 0x70,
    0x29, 0x29, 0x5b, 0x30, 0x5d, 0x2b, 0x28, 0x6c, 0x29, 0x3b, 0x5c, 0x0a,
    0x74, 0x4a, 0x4a, 0x5b, 0x30, 0x5d, 0x5b, 0x31, 0x5d, 0x3d, 0x28, 0x28,
    0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61,
    0x74, 0x2a, 0x29, 0x28, 0x72, 0x70, 0x29, 0x29, 0x5b, 0x31, 0x5d, 0x3b,
    0x5c, 0x0a, 0x74, 0x4a, 0x4a, 0x5b, 0x30, 0x5d, 0x5b, 0x32, 0x5d, 0x3d,
    0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c,
    0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x72, 0x70, 0x29, 0x29, 0x5b, 0x32,
    0x5d, 0x3b, 0x5c, 0x0a, 0x74, 0x4a, 0x4a, 0x5b, 0x31, 0x5d, 0x5b, 0x30,
    0x5d, 0x3d, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20,
    0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x72, 0x70, 0x29, 0x29,
    0x5b, 0x31, 0x5d, 0x3b, 0x5c, 0x0a, 0x74, 0x4a, 0x4a, 0x5b, 0x31, 0x5d,
    0x5b, 0x31, 0x5d, 0x3d, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61,
    0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x72, 0x70,
    0x29, 0x29, 0x5b, 0x33, 0x5d, 0x2b, 0x28, 0x6c, 0x29, 0x3b, 0x5c, 0x0a,
    0x74, 0x4a, 0x4a, 0x5b, 0x31, 0x5d, 0x5b, 0x32, 0x5d, 0x3d, 0x28, 0x28,
    0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61,
    0x74, 0x2a, 0x29, 0x28, 0x72, 0x70, 0x29, 0x29, 0x5b, 0x34, 0x5d, 0x2b,
    0x28, 0x6c, 0x29, 0x3b, 0x5c, 0x0a, 0x74, 0x4a, 0x4a, 0x5b, 0x32, 0x5d,
    0x5b, 0x30, 0x5d, 0x3d, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61,
    0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x72, 0x70,
    0x29, 0x29, 0x5b, 0x32, 0x5d, 0x2b, 0x28, 0x6c, 0x29, 0x3b, 0x5c, 0x0a,
    0x74, 0x4a, 0x4a, 0x5b, 0x32, 0x5d, 0x5b, 0x31, 0x5d, 0x3d, 0x28, 0x28,
    0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61,
    0x74, 0x2a, 0x29, 0x28, 0x72, 0x70, 0x29, 0x29, 0x5b, 0x34, 0x5d, 0x2b,
    0x28, 0x6c, 0x29, 0x3b, 0x5c, 0x0a, 0x74, 0x4a, 0x4a, 0x5b, 0x32, 0x5d,
    0x5b, 0x32, 0x5d, 0x3d, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61,
    0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x72, 0x70,
    0x29, 0x29, 0x5b, 0x35, 0x5d, 0x2b, 0x28, 0x6c, 0x29, 0x3b, 0x5c, 0x0a,
    0x62, 0x61, 0x72, 0x72, 0x69, 0x65, 0x72, 0x28, 0x43, 0x4c, 0x4b, 0x5f,
    0x4c, 0x4f, 0x43, 0x41, 0x4c, 0x5f, 0x4d, 0x45, 0x4d, 0x5f, 0x46, 0x45,
    0x4e, 0x43, 0x45, 0x29, 0x3b, 0x5c, 0x0a, 0x5c, 0x0a, 0x66, 0x6c, 0x6f,
    0x61, 0x74, 0x20, 0x64, 0x65, 0x74, 0x5f, 0x74, 0x4a, 0x4a, 0x3b, 0x5c,
    0x0a, 0x64, 0x65, 0x74, 0x5f, 0x74, 0x4a, 0x4a, 0x20, 0x3d, 0x20, 0x74,
    0x4a, 0x4a, 0x5b, 0x30, 0x5d, 0x5b, 0x30, 0x5d, 0x2a, 0x74, 0x4a, 0x4a,
    0x5b, 0x31, 0x5d, 0x5b, 0x31, 0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x32,
    0x5d, 0x5b, 0x32, 0x5d, 0x20, 0x2b, 0x74, 0x4a, 0x4a, 0x5b, 0x30, 0x5d,
    0x5b, 0x31, 0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x31, 0x5d, 0x5b, 0x32,
    0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x32, 0x5d, 0x5b, 0x30, 0x5d, 0x20,
    0x2b, 0x74, 0x4a, 0x4a, 0x5b, 0x30, 0x5d, 0x5b, 0x32, 0x5d, 0x2a, 0x74,
    0x4a, 0x4a, 0x5b, 0x31, 0x5d, 0x5b, 0x30, 0x5d, 0x2a, 0x74, 0x4a, 0x4a,
    0x5b, 0x32, 0x5d, 0x5b, 0x31, 0x5d, 0x5c, 0x0a, 0x2d, 0x74, 0x4a, 0x4a,
    0x5b, 0x30, 0x5d, 0x5b, 0x30, 0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x31,
    0x5d, 0x5b, 0x32, 0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x32, 0x5d, 0x5b,
    0x31, 0x5d, 0x20, 0x2d, 0x74, 0x4a, 0x4a, 0x5b, 0x30, 0x5d, 0x5b, 0x31,
    0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x31, 0x5d, 0x5b, 0x30, 0x5d, 0x2a,
    0x74, 0x4a, 0x4a, 0x5b, 0x32, 0x5d, 0x5b, 0x32, 0x5d, 0x20, 0x2d, 0x74,
    0x4a, 0x4a, 0x5b, 0x30, 0x5d, 0x5b, 0x32, 0x5d, 0x2a, 0x74, 0x4a, 0x4a,
    0x5b, 0x31, 0x5d, 0x5b, 0x31, 0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x32,
    0x5d, 0x5b, 0x30, 0x5d, 0x3b, 0x5c, 0x0a, 0x5c, 0x0a, 0x66, 0x6c, 0x6f,
    0x61, 0x74, 0x20, 0x69, 0x6e, 0x76, 0x5f, 0x74, 0x4a, 0x4a, 0x5b, 0x33,
    0x5d, 0x5b, 0x33, 0x5d, 0x3b, 0x5c, 0x0a, 0x69, 0x6e, 0x76, 0x5f, 0x74,
    0x4a, 0x4a, 0x5b, 0x30, 0x5d, 0x5b, 0x30, 0x5d, 0x20, 0x3d, 0x20, 0x20,
    0x28, 0x74, 0x4a, 0x4a, 0x5b, 0x31, 0x5d, 0x5b, 0x31, 0x5d, 0x2a, 0x74,
    0x4a, 0x4a, 0x5b, 0x32, 0x5d, 0x5b, 0x32, 0x5d, 0x2d, 0x74, 0x4a, 0x4a,
    0x5b, 0x31, 0x5d, 0x5b, 0x32, 0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x32,
    0x5d, 0x5b, 0x31, 0x5d, 0x29, 0x2f, 0x28, 0x64, 0x65, 0x74, 0x5f, 0x74,
    0x4a, 0x4a, 0x29, 0x3b, 0x5c, 0x0a, 0x69, 0x6e, 0x76, 0x5f, 0x74, 0x4a,
    0x4a, 0x5b, 0x30, 0x5d, 0x5b, 0x31, 0x5d, 0x20, 0x3d, 0x20, 0x2d, 0x28,
    0x74, 0x4a, 0x4a, 0x5b, 0x31, 0x5d, 0x5b, 0x30, 0x5d, 0x2a, 0x74, 0x4a,
    0x4a, 0x5b, 0x32, 0x5d, 0x5b, 0x32, 0x5d, 0x2d, 0x74, 0x4a, 0x4a, 0x5b,
    0x31, 0x5d, 0x5b, 0x32, 0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x32, 0x5d,
    0x5b, 0x30, 0x5d, 0x29, 0x2f, 0x28, 0x64, 0x65, 0x74, 0x5f, 0x74, 0x4a,
    0x4a, 0x29, 0x3b, 0x5c, 0x0a, 0x69, 0x6e, 0x76, 0x5f, 0x74, 0x4a, 0x4a,
    0x5b, 0x30, 0x5d, 0x5b, 0x32, 0x5d, 0x20, 0x3d, 0x20, 0x20, 0x28, 0x74,
    0x4a, 0x4a, 0x5b, 0x31, 0x5d, 0x5b, 0x30, 0x5d, 0x2a, 0x74, 0x4a, 0x4a,
    0x5b, 0x32, 0x5d, 0x5b, 0x31, 0x5d, 0x2d, 0x74, 0x4a, 0x4a, 0x5b, 0x31,
    0x5d, 0x5b, 0x31, 0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x32, 0x5d, 0x5b,
    0x30, 0x5d, 0x29, 0x2f, 0x28, 0x64, 0x65, 0x74, 0x5f, 0x74, 0x4a, 0x4a,
    0x29, 0x3b, 0x5c, 0x0a, 0x69, 0x6e, 0x76, 0x5f, 0x74, 0x4a, 0x4a, 0x5b,
    0x31, 0x5d, 0x5b, 0x30, 0x5d, 0x20, 0x3d, 0x20, 0x2d, 0x28, 0x74, 0x4a,
    0x4a, 0x5b, 0x30, 0x5d, 0x5b, 0x31, 0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b,
    0x32, 0x5d, 0x5b, 0x32, 0x5d, 0x2d, 0x74, 0x4a, 0x4a, 0x5b, 0x30, 0x5d,
    0x5b, 0x32, 0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x32, 0x5d, 0x5b, 0x31,
    0x5d, 0x29, 0x2f, 0x28, 0x64, 0x65, 0x74, 0x5f, 0x74, 0x4a, 0x4a, 0x29,
    0x3b, 0x5c, 0x0a, 0x69, 0x6e, 0x76, 0x5f, 0x74, 0x4a, 0x4a, 0x5b, 0x31,
    0x5d, 0x5b, 0x31, 0x5d, 0x20, 0x3d, 0x20, 0x20, 0x28, 0x74, 0x4a, 0x4a,
    0x5b, 0x30, 0x5d, 0x5b, 0x30, 0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x32,
    0x5d, 0x5b, 0x32, 0x5d, 0x2d, 0x74, 0x4a, 0x4a, 0x5b, 0x30, 0x5d, 0x5b,
    0x32, 0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x32, 0x5d, 0x5b, 0x30, 0x5d,
    0x29, 0x2f, 0x28, 0x64, 0x65, 0x74, 0x5f, 0x74, 0x4a, 0x4a, 0x29, 0x3b,
    0x5c, 0x0a, 0x69, 0x6e, 0x76, 0x5f, 0x74, 0x4a, 0x4a, 0x5b, 0x31, 0x5d,
    0x5b, 0x32, 0x5d, 0x20, 0x3d, 0x20, 0x2d, 0x28, 0x74, 0x4a, 0x4a, 0x5b,
    0x30, 0x5d, 0x5b, 0x30, 0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x32, 0x5d,
    0x5b, 0x31, 0x5d, 0x2d, 0x74, 0x4a, 0x4a, 0x5b, 0x30, 0x5d, 0x5b, 0x31,
    0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x32, 0x5d, 0x5b, 0x30, 0x5d, 0x29,
    0x2f, 0x28, 0x64, 0x65, 0x74, 0x5f, 0x74, 0x4a, 0x4a, 0x29, 0x3b, 0x5c,
    0x0a, 0x69, 0x6e, 0x76, 0x5f, 0x74, 0x4a, 0x4a, 0x5b, 0x32, 0x5d, 0x5b,
    0x30, 0x5d, 0x20, 0x3d, 0x20, 0x20, 0x28, 0x74, 0x4a, 0x4a, 0x5b, 0x30,
    0x5d, 0x5b, 0x31, 0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x31, 0x5d, 0x5b,
    0x32, 0x5d, 0x2d, 0x74, 0x4a, 0x4a, 0x5b, 0x30, 0x5d, 0x5b, 0x32, 0x5d,
    0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x31, 0x5d, 0x5b, 0x31, 0x5d, 0x29, 0x2f,
    0x28, 0x64, 0x65, 0x74, 0x5f, 0x74, 0x4a, 0x4a, 0x29, 0x3b, 0x5c, 0x0a,
    0x69, 0x6e, 0x76, 0x5f, 0x74, 0x4a, 0x4a, 0x5b, 0x32, 0x5d, 0x5b, 0x31,
    0x5d, 0x20, 0x3d, 0x20, 0x2d, 0x28, 0x74, 0x4a, 0x4a, 0x5b, 0x30, 0x5d,
    0x5b, 0x30, 0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x31, 0x5d, 0x5b, 0x32,
    0x5d, 0x2d, 0x74, 0x4a, 0x4a, 0x5b, 0x30, 0x5d, 0x5b, 0x32, 0x5d, 0x2a,
    0x74, 0x4a, 0x4a, 0x5b, 0x31, 0x5d, 0x5b, 0x30, 0x5d, 0x29, 0x2f, 0x28,
    0x64, 0x65, 0x74, 0x5f, 0x74, 0x4a, 0x4a, 0x29, 0x3b, 0x5c, 0x0a, 0x69,
    0x6e, 0x76, 0x5f, 0x74, 0x4a, 0x4a, 0x5b, 0x32, 0x5d, 0x5b, 0x32, 0x5d,
    0x20, 0x3d, 0x20, 0x20, 0x28, 0x74, 0x4a, 0x4a, 0x5b, 0x30, 0x5d, 0x5b,
    0x30, 0x5d, 0x2a, 0x74, 0x4a, 0x4a, 0x5b, 0x31, 0x5d, 0x5b, 0x31, 0x5d,
    0x2d, 0x74, 0x4a, 0x4a, 0x5b, 0x30, 0x5d, 0x5b, 0x31, 0x5d, 0x2a, 0x74,
    0x4a, 0x4a, 0x5b, 0x31, 0x5d, 0x5b, 0x30, 0x5d, 0x29, 0x2f, 0x28, 0x64,
    0x65, 0x74, 0x5f, 0x74, 0x4a, 0x4a, 0x29, 0x3b, 0x5c, 0x0a, 0x5c, 0x0a,
    0x69, 0x66, 0x28, 0x28, 0x6c, 0x69, 0x64, 0x29, 0x3d, 0x3d, 0x30, 0x29,
    0x7b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c,
    0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x64, 0x74, 0x70,
    0x29, 0x29, 0x5b, 0x30, 0x5d, 0x3d, 0x69, 0x6e, 0x76, 0x5f, 0x74, 0x4a,
    0x4a, 0x5b, 0x30, 0x5d, 0x5b, 0x30, 0x5d, 0x2a, 0x28, 0x28, 0x5f, 0x5f,
    0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a,
    0x29, 0x28, 0x72, 0x70, 0x29, 0x29, 0x5b, 0x36, 0x5d, 0x2b, 0x69, 0x6e,
    0x76, 0x5f, 0x74, 0x4a, 0x4a, 0x5b, 0x30, 0x5d, 0x5b, 0x31, 0x5d, 0x2a,
    0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c,
    0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x72, 0x70, 0x29, 0x29, 0x5b, 0x37,
    0x5d, 0x2b, 0x69, 0x6e, 0x76, 0x5f, 0x74, 0x4a, 0x4a, 0x5b, 0x30, 0x5d,
    0x5b, 0x32, 0x5d, 0x2a, 0x28, 0x28, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a,
    0x29, 0x28, 0x72, 0x70, 0x29, 0x29, 0x5b, 0x38, 0x5d, 0x3b, 0x5c, 0x0a,
    0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c,
    0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x64, 0x74, 0x70, 0x29, 0x29, 0x5b,
    0x31, 0x5d, 0x3d, 0x69, 0x6e, 0x76, 0x5f, 0x74, 0x4a, 0x4a, 0x5b, 0x31,
    0x5d, 0x5b, 0x30, 0x5d, 0x2a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63,
    0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x72,
    0x70, 0x29, 0x29, 0x5b, 0x36, 0x5d, 0x2b, 0x69, 0x6e, 0x76, 0x5f, 0x74,
    0x4a, 0x4a, 0x5b, 0x31, 0x5d, 0x5b, 0x31, 0x5d, 0x2a, 0x28, 0x28, 0x5f,
    0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74,
    0x2a, 0x29, 0x28, 0x72, 0x70, 0x29, 0x29, 0x5b, 0x37, 0x5d, 0x2b, 0x69,
    0x6e, 0x76, 0x5f, 0x74, 0x4a, 0x4a, 0x5b, 0x31, 0x5d, 0x5b, 0x32, 0x5d,
    0x2a, 0x28, 0x28, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x72,
    0x70, 0x29, 0x29, 0x5b, 0x38, 0x5d, 0x3b, 0x5c, 0x0a, 0x28, 0x28, 0x5f,
    0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74,
    0x2a, 0x29, 0x28, 0x64, 0x74, 0x70, 0x29, 0x29, 0x5b, 0x32, 0x5d, 0x3d,
    0x69, 0x6e, 0x76, 0x5f, 0x74, 0x4a, 0x4a, 0x5b, 0x32, 0x5d, 0x5b, 0x30,
    0x5d, 0x2a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20,
    0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x72, 0x70, 0x29, 0x29,
    0x5b, 0x36, 0x5d, 0x2b, 0x69, 0x6e, 0x76, 0x5f, 0x74, 0x4a, 0x4a, 0x5b,
    0x32, 0x5d, 0x5b, 0x31, 0x5d, 0x2a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f,
    0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28,
    0x72, 0x70, 0x29, 0x29, 0x5b, 0x37, 0x5d, 0x2b, 0x69, 0x6e, 0x76, 0x5f,
    0x74, 0x4a, 0x4a, 0x5b, 0x32, 0x5d, 0x5b, 0x32, 0x5d, 0x2a, 0x28, 0x28,
    0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x72, 0x70, 0x29, 0x29,
    0x5b, 0x38, 0x5d, 0x3b, 0x5c, 0x0a, 0x5c, 0x0a, 0x28, 0x64, 0x72, 0x29,
    0x20, 0x3d, 0x20, 0x28, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61,
    0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x64, 0x74,
    0x70, 0x29, 0x29, 0x5b, 0x30, 0x5d, 0x2a, 0x28, 0x28, 0x6c, 0x29, 0x2a,
    0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c,
    0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x64, 0x74, 0x70, 0x29, 0x29, 0x5b,
    0x30, 0x5d, 0x2d, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c,
    0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x72, 0x70, 0x29,
    0x29, 0x5b, 0x36, 0x5d, 0x29, 0x5c, 0x0a, 0x2b, 0x28, 0x28, 0x5f, 0x5f,
    0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a,
    0x29, 0x28, 0x64, 0x74, 0x70, 0x29, 0x29, 0x5b, 0x31, 0x5d, 0x2a, 0x28,
    0x28, 0x6c, 0x29, 0x2a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61,
    0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x64, 0x74,
    0x70, 0x29, 0x29, 0x5b, 0x31, 0x5d, 0x2d, 0x28, 0x28, 0x5f, 0x5f, 0x6c,
    0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29,
    0x28, 0x72, 0x70, 0x29, 0x29, 0x5b, 0x37, 0x5d, 0x29, 0x5c, 0x0a, 0x2b,
    0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c,
    0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x64, 0x74, 0x70, 0x29, 0x29, 0x5b,
    0x32, 0x5d, 0x2a, 0x28, 0x28, 0x6c, 0x29, 0x2a, 0x28, 0x28, 0x5f, 0x5f,
    0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a,
    0x29, 0x28, 0x64, 0x74, 0x70, 0x29, 0x29, 0x5b, 0x32, 0x5d, 0x2d, 0x28,
    0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f,
    0x61, 0x74, 0x2a, 0x29, 0x28, 0x72, 0x70, 0x29, 0x29, 0x5b, 0x38, 0x5d,
    0x29, 0x29, 0x2f, 0x32, 0x3b, 0x5c, 0x0a, 0x7d, 0x5c, 0x0a, 0x62, 0x61,
    0x72, 0x72, 0x69, 0x65, 0x72, 0x28, 0x43, 0x4c, 0x4b, 0x5f, 0x4c, 0x4f,
    0x43, 0x41, 0x4c, 0x5f, 0x4d, 0x45, 0x4d, 0x5f, 0x46, 0x45, 0x4e, 0x43,
    0x45, 0x29, 0x3b, 0x5c, 0x0a, 0x5c, 0x0a, 0x7d, 0x0a, 0x0a, 0x0a, 0x0a,
    0x2f, 0x2f, 0x75, 0x70, 0x64, 0x61, 0x74, 0x65, 0x20, 0x74, 0x72, 0x61,
    0x6e, 0x73, 0x70, 0x61, 0x72, 0x61, 0x0a, 0x23, 0x64, 0x65, 0x66, 0x69,
    0x6e, 0x65, 0x20, 0x55, 0x50, 0x44, 0x41, 0x54, 0x45, 0x5f, 0x54, 0x52,
    0x41, 0x4e, 0x53, 0x50, 0x41, 0x52, 0x41, 0x28, 0x64, 0x74, 0x70, 0x2c,
    0x74, 0x70, 0x2c, 0x6d, 0x2c, 0x6c, 0x69, 0x64, 0x29, 0x5c, 0x0a, 0x7b,
    0x5c, 0x0a, 0x69, 0x66, 0x28, 0x28, 0x6c, 0x69, 0x64, 0x29, 0x3d, 0x3d,
    0x30, 0x29, 0x7b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63,
    0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x74,
    0x70, 0x29, 0x29, 0x5b, 0x30, 0x5d, 0x20, 0x2b, 0x3d, 0x20, 0x28, 0x6d,
    0x29, 0x2a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20,
    0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x64, 0x74, 0x70, 0x29,
    0x29, 0x5b, 0x30, 0x5d, 0x3b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x6c,
    0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29,
    0x28, 0x74, 0x70, 0x29, 0x29, 0x5b, 0x31, 0x5d, 0x20, 0x2b, 0x3d, 0x20,
    0x28, 0x6d, 0x29, 0x2a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61,
    0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x64, 0x74,
    0x70, 0x29, 0x29, 0x5b, 0x31, 0x5d, 0x3b, 0x5c, 0x0a, 0x28, 0x28, 0x5f,
    0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74,
    0x2a, 0x29, 0x28, 0x74, 0x70, 0x29, 0x29, 0x5b, 0x32, 0x5d, 0x20, 0x2b,
    0x3d, 0x20, 0x28, 0x6d, 0x29, 0x2a, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f,
    0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28,
    0x64, 0x74, 0x70, 0x29, 0x29, 0x5b, 0x32, 0x5d, 0x3b, 0x5c, 0x0a, 0x7d,
    0x5c, 0x0a, 0x62, 0x61, 0x72, 0x72, 0x69, 0x65, 0x72, 0x28, 0x43, 0x4c,
    0x4b, 0x5f, 0x4c, 0x4f, 0x43, 0x41, 0x4c, 0x5f, 0x4d, 0x45, 0x4d, 0x5f,
    0x46, 0x45, 0x4e, 0x43, 0x45, 0x29, 0x3b, 0x5c, 0x0a, 0x5c, 0x0a, 0x7d,
    0x0a, 0x0a, 0x0a, 0x0a, 0x2f, 0x2f, 0x63, 0x6f, 0x70, 0x79, 0x20, 0x64,
    0x61, 0x74, 0x61, 0x20, 0x66, 0x72, 0x6f, 0x6d, 0x20, 0x6c, 0x6f, 0x63,
    0x61, 0x6c, 0x20, 0x74, 0x72, 0x61, 0x6e, 0x73, 0x70, 0x61, 0x72, 0x61,
    0x5f, 0x61, 0x74, 0x45, 0x20, 0x74, 0x6f, 0x20, 0x67, 0x6c, 0x6f, 0x62,
    0x61, 0x6c, 0x20, 0x74, 0x72, 0x61, 0x6e, 0x73, 0x70, 0x61, 0x72, 0x61,
    0x0a, 0x23, 0x64, 0x65, 0x66, 0x69, 0x6e, 0x65, 0x20, 0x54, 0x52, 0x41,
    0x4e, 0x53, 0x50, 0x41, 0x52, 0x41, 0x5f, 0x47, 0x4c, 0x4f, 0x42, 0x5f,
    0x43, 0x4f, 0x50, 0x59, 0x28, 0x74, 0x70, 0x2c, 0x74, 0x70, 0x67, 0x2c,
    0x6c, 0x69, 0x64, 0x2c, 0x67, 0x69, 0x64, 0x29, 0x5c, 0x0a, 0x7b, 0x5c,
    0x0a, 0x69, 0x66, 0x28, 0x28, 0x6c, 0x69, 0x64, 0x29, 0x3d, 0x3d, 0x30,
    0x29, 0x7b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x67, 0x6c, 0x6f, 0x62,
    0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x74,
    0x70, 0x67, 0x29, 0x29, 0x5b, 0x67, 0x69, 0x64, 0x2a, 0x33, 0x5d, 0x3d,
    0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20, 0x66, 0x6c,
    0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x74, 0x70, 0x29, 0x29, 0x5b, 0x30,
    0x5d, 0x3b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x67, 0x6c, 0x6f, 0x62,
    0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x74,
    0x70, 0x67, 0x29, 0x29, 0x5b, 0x67, 0x69, 0x64, 0x2a, 0x33, 0x2b, 0x31,
    0x5d, 0x3d, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x20,
    0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x74, 0x70, 0x29, 0x29,
    0x5b, 0x31, 0x5d, 0x3b, 0x5c, 0x0a, 0x28, 0x28, 0x5f, 0x5f, 0x67, 0x6c,
    0x6f, 0x62, 0x61, 0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29,
    0x28, 0x74, 0x70, 0x67, 0x29, 0x29, 0x5b, 0x67, 0x69, 0x64, 0x2a, 0x33,
    0x2b, 0x32, 0x5d, 0x3d, 0x28, 0x28, 0x5f, 0x5f, 0x6c, 0x6f, 0x63, 0x61,
    0x6c, 0x20, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x2a, 0x29, 0x28, 0x74, 0x70,
    0x29, 0x29, 0x5b, 0x32, 0x5d, 0x3b, 0x5c, 0x0a, 0x7d, 0x5c, 0x0a, 0x62,
    0x61, 0x72, 0x72, 0x69, 0x65, 0x72, 0x28, 0x43, 0x4c, 0x4b, 0x5f, 0x4c,
    0x4f, 0x43, 0x41, 0x4c, 0x5f, 0x4d, 0x45, 0x4d, 0x5f, 0x46, 0x45, 0x4e,
    0x43, 0x45, 0x7c, 0x43, 0x4c, 0x4b, 0x5f, 0x47, 0x4c, 0x4f, 0x42, 0x41,
    0x4c, 0x5f, 0x4d, 0x45, 0x4d, 0x5f, 0x46, 0x45, 0x4e, 0x43, 0x45, 0x29,
    0x3b, 0x5c, 0x0a, 0x7d, 0x0a, 0x0a, 0x0a
};

#endif
