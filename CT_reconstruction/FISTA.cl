//
//  ISTA.cl
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2017/05/05.
//  Copyright (c) 2017 Nozomu Ishiguro. All rights reserved.
//

#ifndef PI
#define PI 3.14159265358979323846
#endif

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

#ifndef LAMBDA_FISTA
#define LAMBDA_FISTA 1.0e-5f
#endif

extern __constant sampler_t s_linear;
extern __constant sampler_t s_nearest;

//find max eigenvalue of projection-back-projection matrix by Power iteration (1)
//foward projection
__kernel void powerIter1(__read_only image2d_array_t reconst_img,
                         __write_only image2d_array_t prj_img,
                         __constant float *angle){
    
    const int X = get_global_id(0);
    const int thN = get_global_id(1);
    const int Z = get_global_id(2);
    const int sub = Z;
    const int th = sub + get_global_id(1)*SS;
    
    float4 xyz;
    int4 XthZ= (int4)(X,thN,Z,0);
    int Y;
    float prj=0.0f, rprj;
    float angle_pr;
    //projection from assumed image
    for(Y=0; Y < DEPTHSIZE; Y++){
        angle_pr = angle[th]*PI/180;
        xyz.x =  (X-PRJ_IMAGESIZE/2)*cos(angle_pr)+(Y-DEPTHSIZE/2)*sin(angle_pr) + IMAGESIZE_X/2;
        xyz.y = -(X-PRJ_IMAGESIZE/2)*sin(angle_pr)+(Y-DEPTHSIZE/2)*cos(angle_pr) + IMAGESIZE_Y/2;
        
        prj += read_imagef(reconst_img,s_linear,xyz).x;
    }
    
    write_imagef(prj_img, XthZ, (float4)(prj,0.0f,0.0f,1.0f));
}

//find max eigenvalue of projection-back-projection matrix by Power iteration (2)
//back-projection of projection ratio
__kernel void powerIter2(__write_only image2d_array_t reconst_cnd_img,
                         __read_only image2d_array_t prj_img,
                         __constant float *angle){
    
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    const int sub = Z;
    
    float4 XthZ;
    XthZ.z = Z;
    float bprj=0.0f;
    float img;
    
    //back projection from ratio
    float4 xyz_f = (float4)(X,Y,Z,0.0f);
    int4 xyz_i = (int4)(X,Y,Z,0);
    float angle_pr;
    int th;
    for(int i=0;i<PRJ_ANGLESIZE/SS;i++){
        th = sub + i*SS;
        angle_pr = angle[th]*PI/180;
        XthZ.x =  (X-IMAGESIZE_X/2)*cos(angle_pr)-(Y-IMAGESIZE_Y/2)*sin(angle_pr) + PRJ_IMAGESIZE/2;
        XthZ.y = i;
        
        bprj += read_imagef(prj_img,s_nearest,XthZ).x;
    }
    bprj *= (float)SS/PRJ_ANGLESIZE/DEPTHSIZE;
    
    //update assumed img
    write_imagef(reconst_cnd_img, xyz_i, (float4)(bprj,0.0f,0.0f,1.0f));
}

//find max eigenvalue of projection-back-projection matrix by Power iteration (3)
//update image
__kernel void powerIter3(__read_only image2d_array_t reconst_cnd_img,
                         __write_only image2d_array_t reconst_new_img,
                         __constant float* L2abs){
    
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    
    int4 xyz= (int4)(X,Y,Z,0);
    float img = read_imagef(reconst_cnd_img,s_linear,xyz).x;
    float L2abs_pr = L2abs[Z];
    img = img/sqrt(L2abs_pr);
    
    write_imagef(reconst_new_img, xyz, (float4)(img,0.0f,0.0f,1.0f));
}


//L2-norm of image (1)
//reduction of X
__kernel void imageL2AbsX(__read_only image2d_array_t img_src, __local float *loc_mem,
                          __global float *L2absY){
    const int local_ID = get_local_id(0);
    const int localsize = get_local_size(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    
    float4 img;
    int4 xyz;
    int X,i;
    
    loc_mem[local_ID]=0.0f;
    
    //copy to local memory
    float val;
    for(i=0;i<IMAGESIZE_X/localsize;i++){
        X=local_ID+i*localsize;
        xyz = (int4)(X,Y,Z,0);
        val = read_imagef(img_src,s_nearest,xyz).x;
        loc_mem[local_ID] += val*val;
    }
    barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
    
    //reduction
    size_t s;
    for(s=localsize/2;s>0;s>>=1){
        if(local_ID<s){
            loc_mem[local_ID] += loc_mem[local_ID+s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    //const size_t global_ID = Y + Z*IMAGESIZE_Y + local_ID*IMAGESIZE_Y*SS;
    //L2absY[global_ID]=loc_mem[local_ID];
    if(local_ID==0){
        L2absY[Y+Z*IMAGESIZE_Y]=loc_mem[0];
    }
    barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
}

//L2-norm of image (2)
//reduction of Y
__kernel void imageL2AbsY(__global float *L2absY, __local float *loc_mem,
                          __global float *L2abs){
    const int local_ID = get_local_id(0);
    const int localsize = get_local_size(0);
    const int Z = get_global_id(1);
    
    int Y,i;
    
    loc_mem[local_ID]=0.0f;
    
    //copy to local memory
    for(i=0;i<IMAGESIZE_Y/localsize;i++){
        Y=local_ID+i*localsize;
        loc_mem[local_ID] += L2absY[Y+Z*IMAGESIZE_Y];
    }
    barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
    
    //reduction
    size_t s;
    for(s=localsize/2;s>0;s>>=1){
        if(local_ID<s){
            loc_mem[local_ID] += loc_mem[local_ID+s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    //const size_t global_ID = Z + local_ID*SS;
    //L2abs[global_ID]=loc_mem[local_ID];
    if(local_ID==0){
        L2abs[Z]=loc_mem[0];
    }
    barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
    
}

//back-projection of projection delta
__kernel void FISTAbackProjection(__read_only image2d_array_t reconst_img,
                                   __write_only image2d_array_t reconst_dest_img,
                                   __read_only image2d_array_t dprj_img,
                                   __constant float *angle, __constant float *L,
                                   int sub, float alpha){
    
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    
    float4 XthZ;
    XthZ.z = Z;
    float bprj=0.0f;
    float img;
    int th;
    float angle_pr;
    
    //back projection from delta
    // reconst_img:         lambda(k)
    // dprj_img:            y'(k)
    // bprj:                lambda' = y'(k) x C_pinv
    // reconst_dest_img:    lambda(k+1) = lambda' x lambda(k)
    float4 xyz_f = (float4)(X,Y,Z,0.0f);
    int4 xyz_i = (int4)(X,Y,Z,0);
    for(th=sub;th<PRJ_ANGLESIZE;th+=SS){
        angle_pr = angle[th]*PI/180.0f;
        XthZ.x =  (X-IMAGESIZE_X*0.5f)*cos(angle_pr)-(Y-IMAGESIZE_Y*0.5f)*sin(angle_pr) + PRJ_IMAGESIZE*0.5f;
        XthZ.y = th;
        
        bprj += read_imagef(dprj_img,s_nearest,XthZ).x;
    }
    
    //update assumed img
    img = read_imagef(reconst_img,s_nearest,xyz_f).x + bprj*alpha*SS/PRJ_ANGLESIZE/DEPTHSIZE/L[sub]*0.5f;
    write_imagef(reconst_dest_img, xyz_i, (float4)(img,0.0f,0.0f,1.0f));
}


//ISTA update image (only for 0th cycle for FISTA)
__kernel void ISTA(__read_only image2d_array_t reconst_v_img,
                       __write_only image2d_array_t reconst_x_new_img,
                       int sub, __constant float *L){
    
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    const float4 xyz = (float4)(X+0.5f,Y+0.5f,Z,0.0f);
    const int4 xyz_i = (int4)(X,Y,Z,0);
    const float4 xpyz = (float4)(X+1.5f,Y,Z,0.0f);
    const float4 xmyz = (float4)(X-0.5f,Y,Z,0.0f);
    const float4 xypz = (float4)(X,Y+1.5f,Z,0.0f);
    const float4 xymz = (float4)(X,Y-0.5f,Z,0.0f);
    const float lambda = LAMBDA_FISTA/L[sub]*0.5f;
    
    
    float img_neibor[4];
    img_neibor[0] = read_imagef(reconst_v_img,s_nearest,xpyz).x;
    img_neibor[1] = read_imagef(reconst_v_img,s_nearest,xmyz).x;
    img_neibor[2] = read_imagef(reconst_v_img,s_nearest,xypz).x;
    img_neibor[3] = read_imagef(reconst_v_img,s_nearest,xymz).x;
    float v_img = read_imagef(reconst_v_img,s_nearest,xyz).x;
    
    //change order of img_neibor[4]
    float img1, img2;
    int biggerN;
    for(int i=0;i<4;i++){
        img1 = img_neibor[i];
        img2 =img1;
        biggerN = i;
        for(int j=i+1;j<4;j++){
            biggerN = (img_neibor[j]>img2) ? j:biggerN;
            img2 = (img_neibor[j]>img2) ? img_neibor[j]:img2;
        }
        img_neibor[i] = img2;
        img_neibor[biggerN] = img1;
    }
    
    float img_cnd[9];
    img_cnd[0] = v_img - 4.0f*lambda;
    img_cnd[1] = img_neibor[0];
    img_cnd[2] = v_img - 2.0f*lambda;
    img_cnd[3] = img_neibor[1];
    img_cnd[4] = v_img;
    img_cnd[5] = img_neibor[2];
    img_cnd[6] = v_img + 2.0f*lambda;
    img_cnd[7] = img_neibor[3];
    img_cnd[8] = v_img + 4.0f*lambda;
    
    float v_img_range[8];
    v_img_range[0] = img_neibor[0] + 4.0f*lambda;
    v_img_range[1] = img_neibor[0] + 2.0f*lambda;
    v_img_range[2] = img_neibor[1] + 2.0f*lambda;
    v_img_range[3] = img_neibor[1];
    v_img_range[4] = img_neibor[2];
    v_img_range[5] = img_neibor[2] - 2.0f*lambda;
    v_img_range[6] = img_neibor[3] - 2.0f*lambda;
    v_img_range[7] = img_neibor[3] - 4.0f*lambda;
    
    float x_img_new=img_cnd[0];
    for(int i=0;i<8;i++){
        x_img_new = (v_img<v_img_range[i]) ? img_cnd[i+1]:x_img_new;
    }
    x_img_new = fmax(1.0e-6f, x_img_new);
    x_img_new = (isnan(x_img_new)) ? 1.0e-6f:x_img_new;
    
    //update of x img
    write_imagef(reconst_x_new_img, xyz_i, (float4)(x_img_new,0.0f,0.0f,1.0f));
}


//FISTA update image (more than 1st cycle)
__kernel void FISTA(__read_only image2d_array_t reconst_x_img,
                    __read_only image2d_array_t reconst_v_img,
                    __read_only image2d_array_t reconst_b_img,
                    __write_only image2d_array_t reconst_w_new_img,
                    __write_only image2d_array_t reconst_x_new_img,
                    __write_only image2d_array_t reconst_b_new_img,
                    int sub, __constant float *L){
    
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    const float4 xyz = (float4)(X+0.5f,Y+0.5f,Z,0.0f);
    const int4 xyz_i = (int4)(X,Y,Z,0);
    const float4 xpyz = (float4)(X+1.5f,Y,Z,0.0f);
    const float4 xmyz = (float4)(X-0.5f,Y,Z,0.0f);
    const float4 xypz = (float4)(X,Y+1.5f,Z,0.0f);
    const float4 xymz = (float4)(X,Y-0.5f,Z,0.0f);
    const float lambda = LAMBDA_FISTA/L[sub]*0.5f;
    
    
    float x_img = read_imagef(reconst_x_img,s_nearest,xyz).x;
    float img_neibor[4];
    img_neibor[0] = read_imagef(reconst_v_img,s_nearest,xpyz).x;
    img_neibor[1] = read_imagef(reconst_v_img,s_nearest,xmyz).x;
    img_neibor[2] = read_imagef(reconst_v_img,s_nearest,xypz).x;
    img_neibor[3] = read_imagef(reconst_v_img,s_nearest,xymz).x;
    float v_img = read_imagef(reconst_v_img,s_nearest,xyz).x;
    float beta = read_imagef(reconst_b_img,s_nearest,xyz).x;
    
    //change order of img_neibor[4]
    float img1, img2;
    int biggerN;
    for(int i=0;i<4;i++){
        img1 = img_neibor[i];
        img2 =img1;
        biggerN = i;
        for(int j=i+1;j<4;j++){
            biggerN = (img_neibor[j]>img2) ? j:biggerN;
            img2 = (img_neibor[j]>img2) ? img_neibor[j]:img2;
        }
        img_neibor[i] = img2;
        img_neibor[biggerN] = img1;
    }
    
    float img_cnd[9];
    img_cnd[0] = v_img - 4.0f*lambda;
    img_cnd[1] = img_neibor[0];
    img_cnd[2] = v_img - 2.0f*lambda;
    img_cnd[3] = img_neibor[1];
    img_cnd[4] = v_img;
    img_cnd[5] = img_neibor[2];
    img_cnd[6] = v_img + 2.0f*lambda;
    img_cnd[7] = img_neibor[3];
    img_cnd[8] = v_img + 4.0f*lambda;
    
    float v_img_range[8];
    v_img_range[0] = img_neibor[0] + 4.0f*lambda;
    v_img_range[1] = img_neibor[0] + 2.0f*lambda;
    v_img_range[2] = img_neibor[1] + 2.0f*lambda;
    v_img_range[3] = img_neibor[1];
    v_img_range[4] = img_neibor[2];
    v_img_range[5] = img_neibor[2] - 2.0f*lambda;
    v_img_range[6] = img_neibor[3] - 2.0f*lambda;
    v_img_range[7] = img_neibor[3] - 4.0f*lambda;
    
    float x_img_new=img_cnd[0];
    for(int i=0;i<8;i++){
        x_img_new = (v_img<v_img_range[i]) ? img_cnd[i+1]:x_img_new;
    }
    
    //update of x img
    write_imagef(reconst_x_new_img, xyz_i, (float4)(x_img_new,0.0f,0.0f,1.0f));
    
    //update of beta image
    float beta_new = beta*beta*4.0f + 1.0f;
    beta_new = (sqrt(beta_new) + 1.0f)*0.5f;
    float gamma=(beta-1.0f)/beta_new;
    write_imagef(reconst_b_new_img, xyz_i, (float4)(beta_new,0.0f,0.0f,1.0f));
    
    //update of w img
    float w_new = gamma*x_img_new + (1.0f - gamma)*x_img;
    w_new = (isnan(w_new)) ? 1.0e-6f:w_new;
    write_imagef(reconst_w_new_img, xyz_i, (float4)(w_new,0.0f,0.0f,1.0f));
}


//ISTA update image (only for 0th cycle for FISTA)
__kernel void ISTA3D(__read_only image2d_array_t reconst_v_img,
                     __write_only image2d_array_t reconst_x_new_img,
                     int sub, __constant float *L){
    
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    const int Zsize = get_global_size(2);
    
    const float4 xyz = (float4)(X+0.5f,Y+0.5f,Z,0.0f);
    const int4 xyz_i = (int4)(X,Y,Z,0);
    const float4 xpyz = (float4)(X+1.5f,Y,Z,0.0f);
    const float4 xmyz = (float4)(X-0.5f,Y,Z,0.0f);
    const float4 xypz = (float4)(X,Y+1.5f,Z,0.0f);
    const float4 xymz = (float4)(X,Y-0.5f,Z,0.0f);
    const float4 xyzp = (float4)(X+0.5f,Y+0.5f,fmin(Z+1.0f,Zsize-1.0f),0.0f);
    const float4 xyzm = (float4)(X+0.5f,Y+0.5f,fmin(Z-1.0f,0.0f),0.0f);
    const float lambda = LAMBDA_FISTA/L[sub]*0.5f;
    
    
    float img_neibor[6];
    img_neibor[0] = read_imagef(reconst_v_img,s_nearest,xpyz).x;
    img_neibor[1] = read_imagef(reconst_v_img,s_nearest,xmyz).x;
    img_neibor[2] = read_imagef(reconst_v_img,s_nearest,xypz).x;
    img_neibor[3] = read_imagef(reconst_v_img,s_nearest,xymz).x;
    img_neibor[4] = read_imagef(reconst_v_img,s_nearest,xyzp).x;
    img_neibor[5] = read_imagef(reconst_v_img,s_nearest,xyzm).x;
    float v_img = read_imagef(reconst_v_img,s_nearest,xyz).x;
    
    //change order of img_neibor[4]
    float img1, img2;
    int biggerN;
    for(int i=0;i<6;i++){
        img1 = img_neibor[i];
        img2 =img1;
        biggerN = i;
        for(int j=i+1;j<6;j++){
            biggerN = (img_neibor[j]>img2) ? j:biggerN;
            img2 = (img_neibor[j]>img2) ? img_neibor[j]:img2;
        }
        img_neibor[i] = img2;
        img_neibor[biggerN] = img1;
    }
    
    float img_cnd[13];
    img_cnd[0] = v_img - 6.0f*lambda;
    img_cnd[1] = img_neibor[0];
    img_cnd[2] = v_img - 4.0f*lambda;
    img_cnd[3] = img_neibor[1];
    img_cnd[4] = v_img - 2.0f*lambda;
    img_cnd[5] = img_neibor[2];
    img_cnd[6] = v_img;
    img_cnd[7] = img_neibor[3];
    img_cnd[8] = v_img + 2.0f*lambda;
    img_cnd[9] = img_neibor[4];
    img_cnd[10]= v_img + 4.0f*lambda;
    img_cnd[11]= img_neibor[5];
    img_cnd[12]= v_img + 6.0f*lambda;
    
    
    float v_img_range[12];
    v_img_range[0] = img_neibor[0] + 6.0f*lambda;
    v_img_range[1] = img_neibor[0] + 4.0f*lambda;
    v_img_range[2] = img_neibor[1] + 4.0f*lambda;
    v_img_range[3] = img_neibor[1] + 2.0f*lambda;
    v_img_range[4] = img_neibor[2] + 2.0f*lambda;
    v_img_range[5] = img_neibor[2];
    v_img_range[6] = img_neibor[3];
    v_img_range[7] = img_neibor[3] - 2.0f*lambda;
    v_img_range[8] = img_neibor[4] - 2.0f*lambda;
    v_img_range[9] = img_neibor[4] - 4.0f*lambda;
    v_img_range[10]= img_neibor[5] - 4.0f*lambda;
    v_img_range[11]= img_neibor[5] - 6.0f*lambda;
    
    
    float x_img_new=img_cnd[0];
    for(int i=0;i<12;i++){
        x_img_new = (v_img<v_img_range[i]) ? img_cnd[i+1]:x_img_new;
    }
    x_img_new = fmax(1.0e-6f, x_img_new);
    x_img_new = (isnan(x_img_new)) ? 1.0e-6f:x_img_new;
    
    //update of x img
    write_imagef(reconst_x_new_img, xyz_i, (float4)(x_img_new,0.0f,0.0f,1.0f));
}


//FISTA update image (more than 1st cycle)
__kernel void FISTA3D(__read_only image2d_array_t reconst_x_img,
                    __read_only image2d_array_t reconst_v_img,
                    __read_only image2d_array_t reconst_b_img,
                    __write_only image2d_array_t reconst_w_new_img,
                    __write_only image2d_array_t reconst_x_new_img,
                    __write_only image2d_array_t reconst_b_new_img,
                    int sub, __constant float *L){
    
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    const int Zsize = get_global_size(2);
    
    const float4 xyz = (float4)(X+0.5f,Y+0.5f,Z,0.0f);
    const int4 xyz_i = (int4)(X,Y,Z,0);
    const float4 xpyz = (float4)(X+1.5f,Y,Z,0.0f);
    const float4 xmyz = (float4)(X-0.5f,Y,Z,0.0f);
    const float4 xypz = (float4)(X,Y+1.5f,Z,0.0f);
    const float4 xymz = (float4)(X,Y-0.5f,Z,0.0f);
    const float4 xyzp = (float4)(X+0.5f,Y+0.5f,fmin(Z+1.0f,Zsize-1.0f),0.0f);
    const float4 xyzm = (float4)(X+0.5f,Y+0.5f,fmin(Z-1.0f,0.0f),0.0f);
    const float lambda = LAMBDA_FISTA/L[sub]*0.5f;
    
    
    float x_img = read_imagef(reconst_x_img,s_nearest,xyz).x;
    float img_neibor[6];
    img_neibor[0] = read_imagef(reconst_v_img,s_nearest,xpyz).x;
    img_neibor[1] = read_imagef(reconst_v_img,s_nearest,xmyz).x;
    img_neibor[2] = read_imagef(reconst_v_img,s_nearest,xypz).x;
    img_neibor[3] = read_imagef(reconst_v_img,s_nearest,xymz).x;
    img_neibor[4] = read_imagef(reconst_v_img,s_nearest,xyzp).x;
    img_neibor[5] = read_imagef(reconst_v_img,s_nearest,xyzm).x;
    float v_img = read_imagef(reconst_v_img,s_nearest,xyz).x;
    float beta = read_imagef(reconst_b_img,s_nearest,xyz).x;
    
    //change order of img_neibor[4]
    float img1, img2;
    int biggerN;
    for(int i=0;i<6;i++){
        img1 = img_neibor[i];
        img2 =img1;
        biggerN = i;
        for(int j=i+1;j<6;j++){
            biggerN = (img_neibor[j]>img2) ? j:biggerN;
            img2 = (img_neibor[j]>img2) ? img_neibor[j]:img2;
        }
        img_neibor[i] = img2;
        img_neibor[biggerN] = img1;
    }
    
    float img_cnd[13];
    img_cnd[0] = v_img - 6.0f*lambda;
    img_cnd[1] = img_neibor[0];
    img_cnd[2] = v_img - 4.0f*lambda;
    img_cnd[3] = img_neibor[1];
    img_cnd[4] = v_img - 2.0f*lambda;
    img_cnd[5] = img_neibor[2];
    img_cnd[6] = v_img;
    img_cnd[7] = img_neibor[3];
    img_cnd[8] = v_img + 2.0f*lambda;
    img_cnd[9] = img_neibor[4];
    img_cnd[10]= v_img + 4.0f*lambda;
    img_cnd[11]= img_neibor[5];
    img_cnd[12]= v_img + 6.0f*lambda;
    
    float v_img_range[12];
    v_img_range[0] = img_neibor[0] + 6.0f*lambda;
    v_img_range[1] = img_neibor[0] + 4.0f*lambda;
    v_img_range[2] = img_neibor[1] + 4.0f*lambda;
    v_img_range[3] = img_neibor[1] + 2.0f*lambda;
    v_img_range[4] = img_neibor[2] + 2.0f*lambda;
    v_img_range[5] = img_neibor[2];
    v_img_range[6] = img_neibor[3];
    v_img_range[7] = img_neibor[3] - 2.0f*lambda;
    v_img_range[8] = img_neibor[4] - 2.0f*lambda;
    v_img_range[9] = img_neibor[4] - 4.0f*lambda;
    v_img_range[10]= img_neibor[5] - 4.0f*lambda;
    v_img_range[11]= img_neibor[5] - 6.0f*lambda;
    
    float x_img_new=img_cnd[0];
    for(int i=0;i<12;i++){
        x_img_new = (v_img<v_img_range[i]) ? img_cnd[i+1]:x_img_new;
    }
    
    //update of x img
    write_imagef(reconst_x_new_img, xyz_i, (float4)(x_img_new,0.0f,0.0f,1.0f));
    
    //update of beta image
    float beta_new = beta*beta*4.0f + 1.0f;
    beta_new = (sqrt(beta_new) + 1.0f)*0.5f;
    float gamma=(beta-1.0f)/beta_new;
    write_imagef(reconst_b_new_img, xyz_i, (float4)(beta_new,0.0f,0.0f,1.0f));
    
    //update of w img
    float w_new = gamma*x_img_new + (1.0f - gamma)*x_img;
    w_new = (isnan(w_new)) ? 1.0e-6f:w_new;
    write_imagef(reconst_w_new_img, xyz_i, (float4)(w_new,0.0f,0.0f,1.0f));
}
