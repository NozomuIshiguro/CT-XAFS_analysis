#define PI 3.14159265358979323846f
#define PI_2 1.57079632679489661923f

#ifndef IMAGESIZE_X
#define IMAGESIZE_X 2048
#endif

#ifndef IMAGESIZE_Y
#define IMAGESIZE_Y 2048
#endif

#ifndef IMAGESIZE_M
#define IMAGESIZE_M 4194304 //2048*2048
#endif

#ifndef ZP_SIZE
#define ZP_SIZE 2048
#endif

#ifndef E_SIZE
#define E_SIZE 100
#endif

//Image Reg mode
#ifndef REGMODE
#define REGMODE 0
#endif
//XYshift
#if REGMODE == 0
#define NUM_PARA 2
#define TRANS_XY(p,xyz,bin) DxDy((float*)(p),xyz,bin)
#define INV_TRANS_XY(p,xyz,bin) inv_DxDy((float*)(p),xyz,bin)
#define JACOBIAN(j,p,XYZ,dfdx,dfdy,bin,msk) Jacob_DxDy((float*)(j),(float*)(p),XYZ,bin,dfdx,dfdy,msk)

//XYshift+rotation
#elif REGMODE == 1
#define NUM_PARA 3
#define TRANS_XY(p,xyz,bin) RotDxDy((float*)(p),xyz,bin)
#define INV_TRANS_XY(p,xyz,bin) inv_RotDxDy((float*)(p),xyz,bin)
#define JACOBIAN(j,p,XYZ,dfdx,dfdy,bin,msk) Jacob_RotDxDy((float*)(j),(float*)(p),(float4)XYZ,bin,dfdx,dfdy,msk)

//XYshift+scale
#elif REGMODE == 2
#define NUM_PARA 3
#define TRANS_XY(p,xyz,bin) ScaleDxDy((float*)(p),xyz,bin)
#define INV_TRANS_XY(p,xyz,bin) inv_ScaleDxDy((float*)(p),xyz,bin)
#define JACOBIAN(j,p,XYZ,dfdx,dfdy,bin,msk) Jacob_ScaleDxDy((float*)(j),(float*)(p),(float4)XYZ,bin,dfdx,dfdy,msk)

//XYshift+ratation+scale
#elif REGMODE == 3
#define NUM_PARA 4
#define TRANS_XY(p,xyz,bin) RotScaleDxDy((float*)(p),xyz,bin)
#define INV_TRANS_XY(p,xyz,bin) inv_RotScaleDxDy((float*)(p),xyz,bin)
#define JACOBIAN(j,p,XYZ,dfdx,dfdy,bin,msk) RotScaleJacob_DxDy((float*)(j),(float*)(p),(float4)XYZ,bin,dfdx,dfdy,msk)

//XYshift+affine
#elif REGMODE == 4
#define NUM_PARA 6
#define TRANS_XY(p,xyz,bin) AffineDxDy((float*)(p),xyz,bin)
#define INV_TRANS_XY(p,xyz,bin) inv_AffineDxDy((float*)(p),xyz,bin)
#define JACOBIAN(j,p,XYZ,dfdx,dfdy,bin,msk) Jacob_AffineDxDy((float*)(j),(float*)(p),(float4)XYZ,bin,dfdx,dfdy,msk)

#endif

extern inline float4 DxDy(float* para, float4 XYZ, int bin);
extern inline float4 RotDxDy(float* para, float4 XYZ, int bin);
extern inline float4 ScaleDxDy(float* para, float4 XYZ, int bin);
extern inline float4 RotScaleDxDy(float* para, float4 XYZ, int bin);
extern inline float4 AffineDxDy(float* para, float4 XYZ,int bin);
extern inline float4 inv_DxDy(float* para, float4 XYZ, int bin);
extern inline float4 inv_RotDxDy(float* para, float4 XYZ, int bin);
extern inline float4 inv_ScaleDxDy(float* para, float4 XYZ, int bin);
extern inline float4 inv_RotScaleDxDy(float* para, float4 XYZ, int bin);
extern inline float4 inv_AffineDxDy(float* para, float4 XYZ,int bin);
extern inline float reduction(__local float *loc_mem, const size_t local_ID, const size_t localsize);

//sampler
__constant sampler_t s_linear = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;
__constant sampler_t s_linear_cEdge = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE;

__kernel void estimateBkgImg(__write_only image2d_t bkg_img,
                          __read_only image2d_array_t mt_data_img,
                          __read_only image2d_t grid_img,__read_only image2d_t sample_img,
                          __constant float* para, __constant float* contrast,
                          __constant float* weight
                          ){
    
    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    int2 XY_i = (int2)(X,Y);
    float4 XYZ;
    float4 inv_XYZ;
    float img1=0.0f;
    float img2;
    float mask =1.0f;
    float p[NUM_PARA];
    float weightSum = 0.0f;
    
    for(int E=0;E<E_SIZE;E++){
        for(int i=0;i<NUM_PARA;i++){
            p[i] = para[i+E*NUM_PARA];
        }
        XYZ=(float4)(X,Y,E,0.0f);
        inv_XYZ = INV_TRANS_XY(p,XYZ,1);
        img2 = read_imagef(mt_data_img,s_linear,XYZ).x;
        mask *= read_imagef(mt_data_img,s_linear,XYZ).y;
        
        img2 -= read_imagef(grid_img,s_linear,inv_XYZ.xy).x;
        mask *= read_imagef(grid_img,s_linear,inv_XYZ.xy).y;
        
        img2 -= contrast[E]*read_imagef(sample_img,s_linear,inv_XYZ.xy).x;
        
        img1 += weight[E]*img2;
        weightSum += weight[E];
    }
    
    img1 *= mask/weightSum;
    write_imagef(bkg_img,XY_i,(float4)(img1,mask,0.0f,0.0f));
}

__kernel void estimateResidueImg(__write_only image2d_array_t residue_img,
                                 __read_only image2d_array_t mt_data_img,
                                 __read_only image2d_t bkg_img){
    
    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    const size_t E = get_global_id(2);
    int4 XYZ;
    float mt;
    float mask;
    float4 img;

    
    XYZ=(int4)(X,Y,E,0.0f);
    img = read_imagef(mt_data_img,s_linear,XYZ);
    mt = img.x;
    mask = img.y;
        
    img   = read_imagef(bkg_img,s_linear,XYZ.xy);
    mt   -= img.x;
    mask *= img.y;
        
    mt *= mask;
        
    write_imagef(residue_img,XYZ,(float4)(mt,mask,0.0f,0.0f));
}

__kernel void estimateGridImg(__write_only image2d_t grid_img,
                              __read_only image2d_array_t residue_reg_img,
                              __read_only image2d_t sample_img,
                              __constant float* contrast, __constant float* weight
                              ){
    
    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    int2 XY_i = (int2)(X,Y);
    float4 XYZ;
    float img1=0.0f;
    float img2;
    float mask =1.0f;
    float weightSum = 0.0f;
    
    for(int E=0;E<E_SIZE;E++){
        XYZ=(float4)(X,Y,E,0.0f);
        
        img2  = read_imagef(residue_reg_img,s_linear,XYZ).x;
        mask *= read_imagef(residue_reg_img,s_linear,XYZ).y;
        
        img2 -= contrast[E]*read_imagef(sample_img,s_linear,XYZ.xy).x;
        mask *= read_imagef(sample_img,s_linear,XYZ.xy).y;
        
        img1 *= weight[E]*img2;
        weightSum += weight[E];
    }
    
    img1 *= mask/weightSum;
    write_imagef(grid_img,XY_i,(float4)(img1,mask,0.0f,0.0f));
}

__kernel void estimateSampleImg(__write_only image2d_t sample_img,
                                __read_only image2d_array_t residue_reg_img,
                                __read_only image2d_t grid_img){
    
    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    int4 XYZ;
    
    
    XYZ=(int4)(X,Y,0.0f,0.0f);
    float img = read_imagef(residue_reg_img,s_linear,XYZ).x;
    float mask = read_imagef(residue_reg_img,s_linear,XYZ).y;
        
    img  -= read_imagef(grid_img,s_linear,XYZ.xy).x;
    mask *= read_imagef(grid_img,s_linear,XYZ.xy).y;
        
    img *= mask;
        
    write_imagef(sample_img,XYZ.xy,(float4)(img,mask,0.0f,0.0f));

}

__kernel void estimateSampleContrast1(__read_only image2d_array_t residue_reg_img,
                                      __read_only image2d_t grid_img,
                                      __read_only image2d_t sample_img,
                                      __global float* sum, __local float* loc_mem){
    
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Y = get_global_id(1);
    const size_t E = get_global_id(2);
    int4 XYZ;
    float4 img;
    float mask;
    float cnt;
    
    //initialize loc_mem
    loc_mem[local_ID]=0.0f;
    
    for(int i=0; i<IMAGESIZE_X; i+=localsize){
        XYZ=(int4)(local_ID+i,Y,E,0.0f);
        img  = read_imagef(residue_reg_img,s_linear,XYZ);
        cnt  = img.x;
        mask = img.y;
        img  = read_imagef(grid_img,s_linear,XYZ.xy);
        cnt -= img.x;
        img  = read_imagef(grid_img,s_linear,XYZ.xy);
        cnt /= fmax(img.x,1E-5f);
        loc_mem[local_ID] += cnt*mask;
    }
        
    
    
    float res = reduction(loc_mem, local_ID, localsize);
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    if(local_ID==0){
        sum[Y+E*IMAGESIZE_Y]=res;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void estimateSampleContrast2(__global float* sum, __global float* contrast,
                                      __local float* loc_mem){
    
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t E = get_global_id(2);
    
    //initialize loc_mem
    loc_mem[local_ID]=0.0f;
    
    for(int i=0; i<IMAGESIZE_Y; i+=localsize){
        loc_mem[local_ID] += sum[local_ID+i+E*IMAGESIZE_Y];
    }
    
    float res = reduction(loc_mem, local_ID, localsize);
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    if(local_ID==0){
        contrast[E]=res;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
}
                    
