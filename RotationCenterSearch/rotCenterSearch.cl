#ifndef IMAGESIZE_X
#define IMAGESIZE_X 2048
#endif

#ifndef IMAGESIZE_Y
#define IMAGESIZE_Y 2048
#endif

#ifndef IMAGESIZE_M
#define IMAGESIZE_M 4194304 //2048*2048
#endif

#define PI 3.14159265358979323846
#define PI_2 1.57079632679489661923

__constant sampler_t s_linear = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;


//reduction
inline float2 reduction(__local float2 *loc_mem, const size_t local_ID, const size_t localsize)
{
    for(size_t s=localsize/2;s>0;s>>=1){
        if(local_ID<s){
            loc_mem[local_ID]+=loc_mem[local_ID+s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    float2 res = loc_mem[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    return res;
}

__kernel void rotCenterShift(__read_only image2d_t prj_input_img,
                             __write_only image2d_t prj_output_img,
                             float rotCenterShift){
    
    
    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    float2 XY_in = (float2)(X+rotCenterShift+0.5f,Y+0.5f);
    int2 XY_out = (int2)(X,Y);
    
    float4 value = read_imagef(prj_input_img,s_linear,XY_in);
    write_imagef(prj_output_img,XY_out,value);
    
}

__kernel void setMask(__write_only image2d_array_t mask_img,
                      int offsetN, float startShift, float shiftStep,
                      float min_ang, float max_ang){
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    
    float shift = startShift+shiftStep*(group_ID+offsetN);
    
    float2 a_min = {min_ang*PI/180,min_ang*PI/180-PI_2};
    float2 a_max = {max_ang*PI/180,max_ang*PI/180-PI_2};
    float2 p_cnt = {IMAGESIZE_X/2,IMAGESIZE_Y/2};
    
    float rp = fmin(IMAGESIZE_X/2+shift,IMAGESIZE_X/2);
    float rm = fmax(-IMAGESIZE_X/2+shift,-IMAGESIZE_X/2);
    
    float2 p1 = rp*cos(a_max);
    float2 p2 = rp*cos(a_min);
    float2 p3 = rm*cos(a_max);
    float2 p4 = rm*cos(a_min);
    
    float p_div = (rp*rp<rm*rm) ? p1.x:p3.x;
    
    
    int4 XYZ;
    float2 XY;
    float4 mask = {1.0f,0.0f,0.0f,0.0f};
    for(int j=0;j<IMAGESIZE_Y;j++){
        for(int i=0;i<IMAGESIZE_X/localsize;i++){
            XY=(float2)(local_ID+i*localsize,j);
            XY -= p_cnt;
            XYZ=(int4)(local_ID+i*localsize,j,group_ID,0);
            
            bool b1 = dot(XY,p1)<=rp*rp;
            bool b2 = dot(XY,p2)<=rp*rp;
            bool b3 = dot(XY,p3)<=rm*rm;
            bool b4 = dot(XY,p4)<=rm*rm;
            
            bool b5 = dot(XY,XY)<=rp*rp && XY.x>p_div;
            bool b6 = dot(XY,XY)<=rm*rm && XY.x<=p_div;
            
            mask.x = (b1*b2*b3*b4*(b5+b6)) ? 1.0f:0.0f;
            write_imagef(mask_img,XYZ,mask);
        }
    }
}


__kernel void imgAVG(__read_only image2d_array_t img,__read_only image2d_array_t mask_img,
                     __local float2 *loc_mem,__global float* avg){
    
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    
    float4 XYZ;
    float2 sum = {0.0f,0.0f};
    for(int j=0;j<IMAGESIZE_Y;j++){
        for(int i=0;i<IMAGESIZE_X/localsize;i++){
            XYZ=(float4)(local_ID+i*localsize+0.5f,j+0.5f,group_ID,0);
            float mask = read_imagef(mask_img,s_linear,XYZ).x;
            float val = (mask==1.0f) ? read_imagef(img,s_linear,XYZ).x:0.0f;
            sum += (float2)(val,mask);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    loc_mem[local_ID]=sum;
    sum = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);

    
    if(local_ID==0){
        avg[group_ID]=sum.x/sum.y;
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    
}

__kernel void imgSTDEV(__read_only image2d_array_t img,__read_only image2d_array_t mask_img,
                       __local float2 *loc_mem,__constant float* avg, __global float* stedev){
    
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    
    float2 var = {0.0f,0.0f};
    for(int j=0;j<IMAGESIZE_Y;j++){
        for(int i=0;i<IMAGESIZE_X/localsize;i++){
            float4 XYZ=(float4)(local_ID+i*localsize+0.5f,j+0.5f,group_ID,0);
            float mask = read_imagef(mask_img,s_linear,XYZ).x;
            float a = (mask==1.0f) ? read_imagef(img,s_linear,XYZ).x - avg[group_ID]:0.0f;
            var += (float2)(a*a,mask);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    loc_mem[local_ID]=var;
    var = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    if(local_ID==0){
        stedev[group_ID]=sqrt(var.x/var.y);
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    
}

__kernel void imgFocusIndex(__read_only image2d_array_t img,__read_only image2d_array_t mask_img,
                       __local float2 *loc_mem,__global float* Findex){
    
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    
    float2 var = {0.0f,0.0f};
    for(int j=0;j<IMAGESIZE_Y;j++){
        for(int i=0;i<IMAGESIZE_X/localsize;i++){
            float4 XYZ=(float4)(local_ID+i*localsize+0.5f,j+0.5f,group_ID,0);
            float4 XYZu=(float4)(local_ID+i*localsize+0.5f,j+1.5f,group_ID,0);
            float4 XYZd=(float4)(local_ID+i*localsize+0.5f,j-0.5f,group_ID,0);
            float4 XYZr=(float4)(local_ID+i*localsize+1.5f,j+0.5f,group_ID,0);
            float4 XYZl=(float4)(local_ID+i*localsize-0.5f,j+0.5f,group_ID,0);
            float4 XYZur=(float4)(local_ID+i*localsize+1.5f,j+1.5f,group_ID,0);
            float4 XYZdr=(float4)(local_ID+i*localsize+1.5f,j-0.5f,group_ID,0);
            float4 XYZul=(float4)(local_ID+i*localsize-0.5f,j+1.5f,group_ID,0);
            float4 XYZdl=(float4)(local_ID+i*localsize-0.5f,j-0.5f,group_ID,0);
            
            float mask = read_imagef(mask_img,s_linear,XYZ).x;
            float mask_u = read_imagef(mask_img,s_linear,XYZu).x;
            float mask_d = read_imagef(mask_img,s_linear,XYZd).x;
            float mask_r = read_imagef(mask_img,s_linear,XYZr).x;
            float mask_l = read_imagef(mask_img,s_linear,XYZl).x;
            float mask_ur = read_imagef(mask_img,s_linear,XYZur).x;
            float mask_dr = read_imagef(mask_img,s_linear,XYZdr).x;
            float mask_ul = read_imagef(mask_img,s_linear,XYZul).x;
            float mask_dl = read_imagef(mask_img,s_linear,XYZdl).x;
            
            float a = read_imagef(img,s_linear,XYZ).x;
            float a_u = read_imagef(img,s_linear,XYZu).x;
            float a_d = read_imagef(img,s_linear,XYZd).x;
            float a_r = read_imagef(img,s_linear,XYZr).x;
            float a_l = read_imagef(img,s_linear,XYZl).x;
            float a_ur = read_imagef(img,s_linear,XYZur).x;
            float a_dr = read_imagef(img,s_linear,XYZdr).x;
            float a_ul = read_imagef(img,s_linear,XYZul).x;
            float a_dl = read_imagef(img,s_linear,XYZdl).x;
            
            float mask_tot = mask*mask_u*mask_d*mask_r*mask_l*mask_ur*mask_dr*mask_ul*mask_dl;
            
            float sq_delta = 0.0f;
            sq_delta += (mask_tot==1.0) ? (a_u-a)*(a_u-a):0.0f;
            sq_delta += (mask_tot==1.0) ? (a_d-a)*(a_d-a):0.0f;
            sq_delta += (mask_tot==1.0) ? (a_r-a)*(a_r-a):0.0f;
            sq_delta += (mask_tot==1.0) ? (a_l-a)*(a_l-a):0.0f;
            sq_delta += (mask_tot==1.0) ? (a_ur-a)*(a_ur-a):0.0f;
            sq_delta += (mask_tot==1.0) ? (a_dr-a)*(a_dr-a):0.0f;
            sq_delta += (mask_tot==1.0) ? (a_ul-a)*(a_ul-a):0.0f;
            sq_delta += (mask_tot==1.0) ? (a_dl-a)*(a_dl-a):0.0f;
            sq_delta /= (mask_tot==1.0) ? a*a:1.0f;
            
            var += (float2)(sq_delta,mask_tot);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    loc_mem[local_ID]=var;
    var = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    if(local_ID==0){
        Findex[group_ID]=sqrt(var.x/var.y);
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    
}
