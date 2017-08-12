//
//  imageFilter.c
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2016/05/16.
//  Copyright © 2016年 Nozomu Ishiguro. All rights reserved.
//


#ifndef IMAGESIZE_X
#define IMAGESIZE_X 2048
#endif

#ifndef IMAGESIZE_Y
#define IMAGESIZE_Y 2048
#endif

#ifndef IMAGESIZE_M
#define IMAGESIZE_M 4194304 //2048*2048
#endif

//sampler
__constant sampler_t s_linear = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;
__constant sampler_t s_linear_cEdge = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE;


__kernel void meanImg(__read_only image2d_array_t mt_img1, __write_only image2d_array_t mt_img2, uint radius)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_group_id(0);
    const size_t Y = get_global_id(1);
    int X,ID,IDxy;
    int4 XYZ;
    
    
    for(size_t i=0;i<IMAGESIZE_X/localsize;i++){
        X=local_ID+i*localsize;
        ID=X+IMAGESIZE_X*Y+Z*IMAGESIZE_M;
        IDxy=X+IMAGESIZE_X*Y;
        XYZ=(int4)(X,Y,Z,0);
        float mt_mean=0;
        float4 mt_f1 = read_imagef(mt_img1,s_linear_cEdge,XYZ);
        for (int k=-(int)radius; k<=radius; k++) {
            int Y1 = Y + k;
            for (int j=-(int)radius; j<=radius; j++) {
                int X1 = X+j;
                int4 XYZ1 =(int4)(X1,Y1,Z,0);
                mt_mean +=read_imagef(mt_img1,s_linear_cEdge,XYZ1).x;
            }
        }
        mt_mean /= (2*radius+1)*(2*radius+1);
        
        mt_f1.x = mt_mean;
        write_imagef(mt_img2,XYZ,mt_f1);
    }
}


__kernel void addImg(__read_only image2d_array_t mt_img1, __write_only image2d_array_t mt_img2, float val)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_group_id(0);
    const size_t Y = get_global_id(1);
    int X,ID,IDxy;
    int4 XYZ;
    
    
    for(size_t i=0;i<IMAGESIZE_X/localsize;i++){
        float4 mt_f1 = read_imagef(mt_img1,s_linear_cEdge,XYZ);
        
        mt_f1.x += val;
        write_imagef(mt_img2,XYZ,mt_f1);
    }
    
}


__kernel void subtractImg(__read_only image2d_array_t mt_img1, __write_only image2d_array_t mt_img2, float val)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_group_id(0);
    const size_t Y = get_global_id(1);
    int X,ID,IDxy;
    int4 XYZ;
    
    
    for(size_t i=0;i<IMAGESIZE_X/localsize;i++){
        float4 mt_f1 = read_imagef(mt_img1,s_linear_cEdge,XYZ);
        
        mt_f1.x -= val;
        write_imagef(mt_img2,XYZ,mt_f1);
    }
    
}


__kernel void MultiplyImg(__read_only image2d_array_t mt_img1, __write_only image2d_array_t mt_img2, float val)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_group_id(0);
    const size_t Y = get_global_id(1);
    int X,ID,IDxy;
    int4 XYZ;
    
    
    for(size_t i=0;i<IMAGESIZE_X/localsize;i++){
        float4 mt_f1 = read_imagef(mt_img1,s_linear_cEdge,XYZ);
        
        mt_f1.x *= val;
        write_imagef(mt_img2,XYZ,mt_f1);
    }
    
}


__kernel void DivideImg(__read_only image2d_array_t mt_img1, __write_only image2d_array_t mt_img2, float val)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_group_id(0);
    const size_t Y = get_global_id(1);
    int X,ID,IDxy;
    int4 XYZ;
    
    
    for(size_t i=0;i<IMAGESIZE_X/localsize;i++){
        float4 mt_f1 = read_imagef(mt_img1,s_linear_cEdge,XYZ);
        
        mt_f1.x /= val;
        write_imagef(mt_img2,XYZ,mt_f1);
    }
    
}


__kernel void RemoveNANImg(__read_only image2d_array_t mt_img1, __write_only image2d_array_t mt_img2,float val)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_group_id(0);
    const size_t Y = get_global_id(1);
    int X,ID,IDxy;
    int4 XYZ;
    
    
    for(size_t i=0;i<IMAGESIZE_X/localsize;i++){
        float4 mt_f1 = read_imagef(mt_img1,s_linear_cEdge,XYZ);
        
        mt_f1.x = isnan(mt_f1.x)? val:mt_f1.x;
        write_imagef(mt_img2,XYZ,mt_f1);
    }
}


__kernel void MinImg(__read_only image2d_array_t mt_img1, __write_only image2d_array_t mt_img2,float minval)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_group_id(0);
    const size_t Y = get_global_id(1);
    int X,ID,IDxy;
    int4 XYZ;
    
    
    for(size_t i=0;i<IMAGESIZE_X/localsize;i++){
        float4 mt_f1 = read_imagef(mt_img1,s_linear_cEdge,XYZ);
        
        mt_f1.x = (mt_f1.x<minval)? minval:mt_f1.x;
        write_imagef(mt_img2,XYZ,mt_f1);
    }
}


__kernel void MaxImg(__read_only image2d_array_t mt_img1, __write_only image2d_array_t mt_img2,float maxval)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_group_id(0);
    const size_t Y = get_global_id(1);
    int X,ID,IDxy;
    int4 XYZ;
    
    
    for(size_t i=0;i<IMAGESIZE_X/localsize;i++){
        float4 mt_f1 = read_imagef(mt_img1,s_linear_cEdge,XYZ);
        
        mt_f1.x = (mt_f1.x>maxval)? maxval:mt_f1.x;
        write_imagef(mt_img2,XYZ,mt_f1);
    }
}


__kernel void expImg(__read_only image2d_array_t mt_img1, __write_only image2d_array_t mt_img2)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_group_id(0);
    const size_t Y = get_global_id(1);
    int X,ID,IDxy;
    int4 XYZ;
    
    
    for(size_t i=0;i<IMAGESIZE_X/localsize;i++){
        float4 mt_f1 = read_imagef(mt_img1,s_linear_cEdge,XYZ);
        
        mt_f1.x = exp(mt_f1.x);
        write_imagef(mt_img2,XYZ,mt_f1);
    }
}

__kernel void lnImg(__read_only image2d_array_t mt_img1, __write_only image2d_array_t mt_img2)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_group_id(0);
    const size_t Y = get_global_id(1);
    int X,ID,IDxy;
    int4 XYZ;
    
    
    for(size_t i=0;i<IMAGESIZE_X/localsize;i++){
        float4 mt_f1 = read_imagef(mt_img1,s_linear_cEdge,XYZ);
        
        mt_f1.x = ln(mt_f1.x);
        write_imagef(mt_img2,XYZ,mt_f1);
    }
}


__kernel void rapizoralImg(__read_only image2d_array_t mt_img1, __write_only image2d_array_t mt_img2)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_group_id(0);
    const size_t Y = get_global_id(1);
    int X,ID,IDxy;
    int4 XYZ;
    
    
    for(size_t i=0;i<IMAGESIZE_X/localsize;i++){
        float4 mt_f1 = read_imagef(mt_img1,s_linear_cEdge,XYZ);
        
        mt_f1.x = 1.0f/mt_f1.x;
        write_imagef(mt_img2,XYZ,mt_f1);
    }
}