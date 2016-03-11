//
//  imageregistration_kernel_src.cl
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2014/12/22.
//  Copyright (c) 2014 Nozomu Ishiguro. All rights reserved.
//



const sampler_t s_nearest = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_REPEAT;
const sampler_t s_linear = CLK_FILTER_LINEAR | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE;
const sampler_t s_repeat = CLK_FILTER_NEAREST | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_REPEAT;

/*Atomic float add*/
float atom_add_float(__global float* const address, const float value)
{
    uint oldval, newval, readback;
    
    *(float*)&oldval = *address;
    *(float*)&newval = (*(float*)&oldval + value);
    while ((readback = atom_cmpxchg((__global uint*)address, oldval, newval)) != oldval) {
        oldval = readback;
        *(float*)&newval = (*(float*)&oldval + value);
    }
    return *(float*)&oldval;
}

/* 0. mt transform*/
__kernel void convert2mt_img(__global float *dark,__global float *I0,__global float *It,
                             __write_only image2d_t mt)
{
    int basex = get_global_id(0);
    int basey = get_global_id(1);
    
    int basesizex = get_global_size(0);
    int basesizey = get_global_size(1);
    
    long base = basex + basesizex*basey;
    int2 baseXY = {basex, basey};
    
    float trans_val = (I0[base] - dark[base])/(It[base] - dark[base]);
    float mt_val = log(trans_val);
    //float mt_val = I0[base];
    write_imagef(mt,baseXY,(float4)(mt_val,0.0f,0.0f,1.0f));
    
}


/* 1. target convlution/shift */
__kernel void imageconvshift1(__read_only image2d_t im, __write_only image2d_t im_out, __global float* transpara) //for target
{
    int basex = get_global_id(0);
    int basey = get_global_id(1);
    
    int localx = get_local_id(0);
    int localy = get_local_id(1);
    
    int groupx = get_group_id(0);
    int groupy = get_group_id(1);
    
    int basesizex = get_global_size(0);
    int basesizey = get_global_size(1);
    
    int groupsize = get_local_size(0)*get_local_size(1);
    
    float x1 = cos(transpara[0])*basex-sin(transpara[0])*basey+transpara[1];
    float y1 = sin(transpara[0])*basex+cos(transpara[0])*basey+transpara[2];
    
    int2 baseXY = {basex, basey};
    int2 localXY = {localx, localy};
    int2 groupXY = {groupx, groupy};
    float2 transXY = {x1,y1};
    
    
    int datatype = get_image_channel_data_type(im_out);
    
    if((x1*(x1-basesizex)<=0)&&(y1*(y1-basesizey)<=0)){
        float4 pixel = read_imagef(im,s_linear, transXY);
        write_imagef(im_out,groupXY,pixel);
    } else {
        write_imagef(im_out,groupXY,(float4)(0.0f,0.0f,0.0f,0.0f));
    }
        
}


/* 2. sample convlution/shift */
__kernel void imageconvshift2(__read_only image2d_t im, __write_only image2d_t im_out, __global float* transpara) //for sample
{
    int basex = get_global_id(0);
    int basey = get_global_id(1);
    
    int localx = get_local_id(0);
    int localy = get_local_id(1);
    
    int groupx = get_group_id(0);
    int groupy = get_group_id(1);
    
    int basesizex = get_global_size(0);
    int basesizey = get_global_size(1);
    
    int groupsize = get_local_size(0)*get_local_size(1);
    
    float x1 = cos(transpara[0])*basex-sin(transpara[0])*basey+transpara[1];
    float y1 = sin(transpara[0])*basex+cos(transpara[0])*basey+transpara[2];
    
    int2 baseXY = {basex, basey};
    int2 localXY = {localx, localy};
    int2 groupXY = {groupx, groupy};
    float2 transXY = {x1,y1};
    
    
    int datatype = get_image_channel_data_type(im_out);
    
    if((x1*(x1-basesizex)<=0)&&(y1*(y1-basesizey)<=0)){
        float4 pixel = read_imagef(im,s_linear, transXY);
        write_imagef(im_out,groupXY,pixel);
    } else {
        write_imagef(im_out,groupXY,(float4)(0.0f,0.0f,0.0f,0.0f));
    }
    
}



/* 3. sample convlution/shift image integration */
__kernel void Sum_Image(__read_only image2d_t image, __global float *Fx, int imagesizeX, int imagesizeY) {
    
    Fx[0] = 0;
    for(int i=0;i<imagesizeX;i++){
        for(int j=0;j<imagesizeY;j++){
            Fx[0] += read_imagef(image,s_linear,(int2)(i,j)).x*read_imagef(image,s_linear,(int2)(i,j)).w;
        }
    }
    
}



/* 4. sample convlution/shift image integration and Lambda update */
__kernel void SumImage_RenewLambda(__read_only image2d_t image, int imagesizeX, int imagesizeY,
                                   __constant float* Fx_old, __global float* Fx_new,
                                   __global float* lambda, __constant float* delta_rho) {
    
    
    Fx_new[0] = 0;
    for(int i=0;i<imagesizeX;i++){
        for(int j=0;j<imagesizeY;j++){
            Fx_new[0] += read_imagef(image,s_linear,(int2)(i,j)).x*read_imagef(image,s_linear,(int2)(i,j)).w;
        }
    }
    
    float rho = (Fx_old[0]-Fx_new[0])/delta_rho[0];
    if(rho>0){
        float a = 1.0f/3.0f;
        float b = 1.0f-(2.0f*rho-1.0f)*(2.0f*rho-1.0f)*(2.0f*rho-1.0f);
        lambda[0] = lambda[0]*fmax(a,b);
    }else{
        lambda[0] = lambda[0]*2.0f;
    }
}



/* 5. resistration Jacobian matrix create */
__kernel void makeJacobian(__read_only image2d_t image_ref,
                           __read_only image2d_t image_test,
                           __global float *transpara, __global float *Jacob, __global float *delta_Img,
                           __global float *dummy, int rotation)
{
    int basex = get_global_id(0);
    int basey = get_global_id(1);
    
    int basesizex = get_global_size(0);
    int basesizey = get_global_size(1);
    
    long base = basex + basesizex*basey;
    int2 baseXY = {basex, basey};
    
    float Dimg_Dx, Dimg_Dy;
    if(basex==0){
        Dimg_Dx = (read_imagef(image_test,s_linear,(int2)(basex+1,basey)) - read_imagef(image_test,s_linear,baseXY)).x;
    } else if(basex==basesizex-1) {
        Dimg_Dx = (read_imagef(image_test,s_linear,baseXY) - read_imagef(image_test,s_linear,(int2)(basex-1,basey))).x;
    }else{
        Dimg_Dx = ((read_imagef(image_test,s_linear,(int2)(basex+1,basey)) - read_imagef(image_test,s_linear,(int2)(basex-1,basey)))/2).x;
    }
    
    if(basey==0){
        Dimg_Dy = (read_imagef(image_test,s_linear,(int2)(basex,basey+1)) - read_imagef(image_test,s_linear,baseXY)).x;
    } else if(basey==basesizey-1) {
        Dimg_Dy = (read_imagef(image_test,s_linear,baseXY) - read_imagef(image_test,s_linear,(int2)(basex,basey-1))).x;
    } else{
        Dimg_Dy = ((read_imagef(image_test,s_linear,(int2)(basex,basey+1)) - read_imagef(image_test,s_linear,(int2)(basex,basey-1)))/2).x;
    }
    
    int Mask = (int)(read_imagef(image_ref,s_linear,(int2)(basex,basey)).w*read_imagef(image_test,s_linear,(int2)(basex,basey)).w);
    
    float Dfx_Dtheta, Dfy_Dtheta;
    Dfx_Dtheta = -basex*sin(transpara[0])-basey*cos(transpara[0]);
    Dfy_Dtheta = basex*cos(transpara[0])-basey*sin(transpara[0]);
    
    if (rotation==1){
        Jacob[base] = (Dimg_Dx*Dfx_Dtheta + Dimg_Dy*Dfy_Dtheta)*Mask;
    }else{
        Jacob[base] = 0;
    }
    Jacob[base+basesizex*basesizey] = Dimg_Dx*Mask;
    Jacob[base+2*basesizex*basesizey] = Dimg_Dy*Mask;
    
    delta_Img[base] = (read_imagef(image_ref,s_linear,baseXY)-read_imagef(image_test,s_linear,baseXY)).x*Mask;
                       
    
    dummy[base]=1;// J_M[base]*J_M[base]; //tJ*J(0,0)
    dummy[base+basesizex*basesizey] = Jacob[base]*Jacob[base+basesizex*basesizey];//tJ*J(0,1),(1,0)
    dummy[base+2*basesizex*basesizey] = Jacob[base]*Jacob[base+2*basesizex*basesizey];//tJ*J(0,2),(2,0)
    dummy[base+3*basesizex*basesizey] = Jacob[base+basesizex*basesizey]*Jacob[base+basesizex*basesizey];//tJ*J(1,1)
    dummy[base+4*basesizex*basesizey] = Jacob[base+basesizex*basesizey]*Jacob[base+2*basesizex*basesizey];//tJ*J(2,1),(1,2)
    dummy[base+5*basesizex*basesizey] = Jacob[base+2*basesizex*basesizey]*Jacob[base+2*basesizex*basesizey];//tJ*J(2,2)
    dummy[base+6*basesizex*basesizey] = Jacob[base]*delta_Img[base]; //tJ*deltaImg(0)
    dummy[base+7*basesizex*basesizey] = Jacob[base+basesizex*basesizey]*delta_Img[base];//tJdeltaImg(1)
    dummy[base+8*basesizex*basesizey] = Jacob[base+2*basesizex*basesizey]*delta_Img[base];//tJ*deltaImg(2)
    
}



/* 7. tJJ reduction1*/
__kernel void reduction_tJJ_tJDeltaImg1(__global float *dummy, int basesizey)
{
    int basex = get_global_id(0);
    
    int basesizex = get_global_size(0);
    
    int sizeXY = basesizex*basesizey;
    
    for(int i=0;i<basesizey;i++){
        for(int s=basesizex/2;s>0;s>>=1){
            if(basex<s){
                for(int j=0;j<9;j++){
                    atom_add_float(dummy+basex+i*basesizex+j*sizeXY, dummy[basex+s+i*basesizex+j*sizeXY]);
                }
            }
            barrier(CLK_GLOBAL_MEM_FENCE);
        }
    }
}



/* 8. tJJ reduction2*/
__kernel void reduction_tJJ_tJDeltaImg2(__global float *dummy, int basesizex)
{
    int basey = get_global_id(0);
    
    int basesizey = get_global_size(0);
    
    int sizeXY = basesizex*basesizey;
    
    for(int s=basesizey/2;s>0;s>>=1){
        if(basey<s){
            for(int j=0;j<9;j++){
                atom_add_float(dummy+basey*basesizex+j*sizeXY, dummy[(basey+s)*basesizex+j*sizeXY]);
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

}


/* 9. transpara update*/
__kernel void renew_taranspara(__global float *transpara, __global float *dummy,
                               __global float *lambda, __global float *delta_rho,
                               int imagesize,int mergesize)
{
    
    //float lambda[0]=0;
    
    float tJJ[3][3];
    float inv_tJJ[3][3];
    
    if(lambda[0]<0){
        lambda[0] = 0.2;//fmax(fmax(dummy[0],dummy[imagesize]),dummy[imagesize*5]);
        while(lambda[0]>1){
            lambda[0]/=10.0f;
        }
    }
    
    
    tJJ[0][0]=dummy[0]+lambda[0];
    tJJ[0][1]=dummy[imagesize];
    tJJ[0][2]=dummy[imagesize*2];
    tJJ[1][0]=dummy[imagesize];
    tJJ[1][1]=dummy[imagesize*3]+lambda[0];
    tJJ[1][2]=dummy[imagesize*4];
    tJJ[2][0]=dummy[imagesize*2];
    tJJ[2][1]=dummy[imagesize*4];
    tJJ[2][2]=dummy[imagesize*5]+lambda[0];
    
    
    float det_tJJ = tJJ[0][0]*tJJ[1][1]*tJJ[2][2] + tJJ[0][1]*tJJ[1][2]*tJJ[2][0]
                    +tJJ[0][2]*tJJ[1][0]*tJJ[2][1] - tJJ[0][0]*tJJ[1][2]*tJJ[2][1]
    -tJJ[0][1]*tJJ[1][0]*tJJ[2][2] - tJJ[0][2]*tJJ[1][1]*tJJ[2][0];
    
    inv_tJJ[0][0] =  (tJJ[1][1]*tJJ[2][2] - tJJ[1][2]*tJJ[2][1])/det_tJJ;
    inv_tJJ[0][1] = -(tJJ[1][0]*tJJ[2][2] - tJJ[1][2]*tJJ[2][0])/det_tJJ;
    inv_tJJ[0][2] =  (tJJ[1][0]*tJJ[2][1] - tJJ[1][1]*tJJ[2][0])/det_tJJ;
    inv_tJJ[1][0] = -(tJJ[0][1]*tJJ[2][2] - tJJ[0][2]*tJJ[2][1])/det_tJJ;
    inv_tJJ[1][1] =  (tJJ[0][0]*tJJ[2][2] - tJJ[0][2]*tJJ[2][0])/det_tJJ;
    inv_tJJ[1][2] = -(tJJ[0][0]*tJJ[2][1] - tJJ[0][1]*tJJ[2][0])/det_tJJ;
    inv_tJJ[2][0] =  (tJJ[0][1]*tJJ[1][2] - tJJ[0][2]*tJJ[1][1])/det_tJJ;
    inv_tJJ[2][1] = -(tJJ[0][0]*tJJ[1][2] - tJJ[0][2]*tJJ[1][0])/det_tJJ;
    inv_tJJ[2][2] =  (tJJ[0][0]*tJJ[1][1] - tJJ[0][1]*tJJ[1][0])/det_tJJ;
    
    float delta_transpara[3];
    delta_transpara[0] = inv_tJJ[0][0]*dummy[imagesize*6] + inv_tJJ[0][1]*dummy[imagesize*7] + inv_tJJ[0][2]*dummy[imagesize*8];
    delta_transpara[1] = inv_tJJ[1][0]*dummy[imagesize*6] + inv_tJJ[1][1]*dummy[imagesize*7] + inv_tJJ[1][2]*dummy[imagesize*8];
    delta_transpara[2] = inv_tJJ[2][0]*dummy[imagesize*6] + inv_tJJ[2][1]*dummy[imagesize*7] + inv_tJJ[2][2]*dummy[imagesize*8];
    
    transpara[0] += delta_transpara[0];
    transpara[1] += mergesize*delta_transpara[1];
    transpara[2] += mergesize*delta_transpara[2];
    
    delta_rho[0] = (delta_transpara[0]*(lambda[0]*delta_transpara[0]-dummy[imagesize*6])
                      +delta_transpara[1]*(lambda[0]*delta_transpara[1]-dummy[imagesize*7])
                      +delta_transpara[2]*(lambda[0]*delta_transpara[2]-dummy[imagesize*8]))/2;
}


/* 10. registration image transfer*/
__kernel void convertImage2Buffer(__read_only image2d_t image, __global float *buffer) {
    
    int basex = get_global_id(0);
    int basey = get_global_id(1);
    
    int basesizex = get_global_size(0);
    int basesizey = get_global_size(1);
    
    long base = basex + basesizex*basey;
    int2 baseXY = {basex, basey};
    
    buffer[base] = read_imagef(image,s_linear,baseXY).x;
}
