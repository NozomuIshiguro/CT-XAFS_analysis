#define PI 3.14159265358979323846
#define PI_2 1.57079632679489661923

#ifndef IMAGESIZE_X
#define IMAGESIZE_X 2048
#endif

#ifndef IMAGESIZE_Y
#define IMAGESIZE_Y 2048
#endif

#ifndef DEPTHSIZE
#define DEPTHSIZE 2048
#endif

#ifndef IMAGESIZE_M
#define IMAGESIZE_M 4194304 //2048*2048
#endif

#ifndef PRJ_IMAGESIZE
#define PRJ_IMAGESIZE 2048
#endif

#ifndef PRJ_ANGLESIZE
#define PRJ_ANGLESIZE 160
#endif

#ifndef SS
#define SS 16
#endif

__constant sampler_t s_linear = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;
__constant sampler_t s_nearest = CLK_FILTER_NEAREST|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;

__kernel void sinogramCorrection(__read_only image2d_array_t prj_img_src,
                                 __write_only image2d_array_t prj_img_dst,
                                 __constant float *angle, int mode, float a, float b){
    
    const int X = get_global_id(0);
    const int th = get_global_id(1);
    const int Z = get_global_id(2);
    
    float4 img;
    float angle_pr = angle[th]/180*PI;
    float sinfactor = a + b*cos(angle_pr);
    
    float4 XthZ_f = (float4)(X,th,Z,0.0f);
    int4 XthZ_i = (int4)(X,th,Z,0);
    //update assumed img
    img = read_imagef(prj_img_src,s_nearest,XthZ_f);
    float theta = 2.0f*fabs((float)X-PRJ_IMAGESIZE/2.0f)/PRJ_IMAGESIZE;
        
    switch(mode){
        case 4: //mask ab
            img.x = (theta*theta < sinfactor*sinfactor) ? img.x:0.0f;
            break;
        case 3: //intensity correction for x + theta
            img.x *= cos(asin(theta))*cos(angle_pr);
            break;
        case 2: //intensity correction for theta
            img.x *= cos(angle_pr);
            break;
        case 1: //intensity correction for x
            img.x *= cos(asin(theta));
            break;
        default:
            break;
    }
    write_imagef(prj_img_dst, XthZ_i, img);
}

__kernel void setThreshold(__read_only image2d_array_t img_src,
                           __write_only image2d_array_t img_dest, float threshold){
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    
    float4 img;
    float4 xyz_f = (float4)(X,Y,Z,0.0f);
    int4 xyz_i = (int4)(X,Y,Z,0);
    img = read_imagef(img_src,s_nearest,xyz_f);
    img.x = (img.x<threshold) ? threshold:img.x;
    
    write_imagef(img_dest, xyz_i, img);
}

__kernel void baseUp(__read_only image2d_array_t img_src,
                     __write_only image2d_array_t img_dest,
                     __constant float *baseup, int order){
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    
    
    float4 xyz_f = (float4)(X,Y,Z,0.0f);
    int4 xyz_i = (int4)(X,Y,Z,0);
    float4 img = read_imagef(img_src,s_nearest,xyz_f);
    float a=baseup[Z],g=0.0f;
    float b1 = (a!=0) ? (img.x-a)/a:0.0f;
    float b2 = 1.0f;
    float flag = -1.0f;
    for(int i=1;i<=order;i++){
        b2 *= b1;
        flag *= -1.0f;
        g += flag*b2/i;
    }
    img.x -= a*exp(g);
    write_imagef(img_dest, xyz_i, img);
}

__kernel void findMinimumX(__read_only image2d_array_t img_src, __local float *loc_mem,
                           __global float *minimumY){
    const int local_ID = get_local_id(0);
    const int localsize = get_local_size(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    
    float4 img;
    float4 xyz_f;
    int4 xyz_i;
    int X,i;
    
    //initialize loc_mem
    loc_mem[local_ID]=INFINITY;
    
    for(i=0;i<IMAGESIZE_X/localsize;i++){
        X=local_ID+i*localsize;
        xyz_f = (float4)(X,Y,Z,0.0f);
        loc_mem[local_ID] = fmin(loc_mem[local_ID],read_imagef(img_src,s_nearest,xyz_f).x);
    }
    barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
    
    size_t s;
    for(s=localsize/2;s>0;s>>=1){
        if(local_ID<s){
            loc_mem[local_ID] = fmin(loc_mem[local_ID],loc_mem[local_ID+s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if(local_ID==0){
        minimumY[Y+Z*IMAGESIZE_Y]=loc_mem[0];
        barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void findMinimumY(__global float *minimumY, __local float *loc_mem,
                           __global float *minimum){
    const int local_ID = get_local_id(0);
    const int localsize = get_local_size(0);
    const int Z = get_global_id(1);
    
    int Y,i;
    
    //initialize loc_mem
    loc_mem[local_ID]=INFINITY;
    
    for(i=0;i<IMAGESIZE_Y/localsize;i++){
        Y=local_ID+i*localsize;
        loc_mem[local_ID] = fmin(loc_mem[local_ID],minimumY[Y+Z*IMAGESIZE_Y]);
    }
    barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
    
    size_t s;
    for(s=localsize/2;s>0;s>>=1){
        if(local_ID<s){
            loc_mem[local_ID] = fmin(loc_mem[local_ID],loc_mem[local_ID+s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if(local_ID==0){
        minimum[Z]=loc_mem[0];
        barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
    }
    
}

__kernel void partialDerivativeOfGradiant(__read_only image2d_array_t original_img_src,
                                          __read_only image2d_array_t img_src,
                                          __write_only image2d_array_t img_dest,
                                          float epsilon, float alpha){
    
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    
    int4 xyz_i = (int4)(X,Y,Z,0);
    float f_o = read_imagef(original_img_src,s_nearest,(float4)(X,Y,Z,0.0f)).x;
    float f_i_j = read_imagef(img_src,s_nearest,(float4)(X,Y,Z,0.0f)).x;
    float f_ip_j = read_imagef(img_src,s_nearest,(float4)(X+1.0f,Y,Z,0.0f)).x;
    float f_im_j = read_imagef(img_src,s_nearest,(float4)(X-1.0f,Y,Z,0.0f)).x;
    float f_im_jp = read_imagef(img_src,s_nearest,(float4)(X-1.0f,Y+1.0f,Z,0.0f)).x;
    float f_i_jp = read_imagef(img_src,s_nearest,(float4)(X,Y+1.0f,Z,0.0f)).x;
    float f_i_jm = read_imagef(img_src,s_nearest,(float4)(X,Y-1.0f,Z,0.0f)).x;
    float f_ip_jm = read_imagef(img_src,s_nearest,(float4)(X+1.0f,Y-1.0f,Z,0.0f)).x;
    
    float v=0.0f;
    float grad = epsilon + (f_ip_j-f_i_j)*(f_ip_j-f_i_j) + (f_i_jp-f_i_j)*(f_i_jp-f_i_j);
    v += (2.0f*f_i_j - f_ip_j - f_i_jp)/sqrt(grad);
    grad = epsilon + (f_i_j-f_im_j)*(f_i_j-f_im_j) + (f_im_jp-f_im_j)*(f_im_jp-f_im_j);
    v += (f_i_j - f_im_j)/sqrt(grad);
    grad = epsilon + (f_ip_jm-f_i_jm)*(f_ip_jm-f_i_jm) + (f_i_j-f_i_jm)*(f_i_j-f_i_jm);
    v += (f_i_j - f_i_jm)/sqrt(grad);
    
    f_i_j -= (v>=0.0f) ? alpha*fabs(f_o):-alpha*fabs(f_o);
    //f_i_j -= alpha*v;
    
    write_imagef(img_dest, xyz_i, (float4)(f_i_j,0.0f,0.0f,1.0f));
}

//projection of trial image and calculate delta between projected image data
__kernel void Profection(__read_only image2d_array_t reconst_img,
                           __write_only image2d_array_t prj_img,
                           __constant float *angle, int sub){
    
    const int X = get_global_id(0);
    const int th = sub + get_global_id(1)*SS;
    const int Z = get_global_id(2);
    
    float4 xyz;
    xyz.z=Z;
    float prj=0.0f;
    float angle_pr;
    int Y;
    
    //projection from assumed image & calculate delta
    // reconst_img: lambda(k)
    // prj_img:     y_i
    // dprj_img:    y(k) = C x lambda(k) => y'(k) = y_i - y(k)
    int4 XthZ = (int4)(X,th,Z,0);
    for(Y=0; Y < DEPTHSIZE; Y++){
        angle_pr = angle[th]*PI/180.0f;
        xyz.x =  (X-PRJ_IMAGESIZE*0.5f)*cos(angle_pr)+(Y-DEPTHSIZE*0.5f)*sin(angle_pr) + IMAGESIZE_X*0.5f+0.5f;
        xyz.y = -(X-PRJ_IMAGESIZE*0.5f)*sin(angle_pr)+(Y-DEPTHSIZE*0.5f)*cos(angle_pr) + IMAGESIZE_Y*0.5f+0.5f;
        
        prj += read_imagef(reconst_img,s_linear,xyz).x;
    }
    
    write_imagef(prj_img, XthZ, (float4)(prj,0.0f,0.0f,1.0f));
}


//back-projection of projection img
__kernel void backProjection(__write_only image2d_array_t reconst_dest_img,
                               __read_only image2d_array_t prj_img,
                               __constant float *angle, int sub){
    
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    
    float4 XthZ;
    XthZ.z = Z;
    float bprj=0.0f;
    float img;
    
    //back projection
    float4 xyz_f = (float4)(X,Y,Z,0.0f);
    int4 xyz_i = (int4)(X,Y,Z,0);
    float angle_pr;
    for(int th=sub;th<PRJ_ANGLESIZE;th+=SS){
        angle_pr = angle[th]*PI/180.0f;
        XthZ.x =  (X-IMAGESIZE_X/2)*cos(angle_pr)-(Y-IMAGESIZE_Y/2)*sin(angle_pr) + PRJ_IMAGESIZE/2 + 0.5f;
        XthZ.y = th + 0.5f;
        
        bprj += read_imagef(prj_img,s_nearest,XthZ).x;
    }
    
    //update assumed img
    img = bprj*SS/PRJ_ANGLESIZE/DEPTHSIZE;
    write_imagef(reconst_dest_img, xyz_i, (float4)(img,0.0f,0.0f,1.0f));
}


