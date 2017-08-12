//
//  imageregistration_kernel_src.cl
//  Image registration share
//
//  Created by Nozomu Ishiguro on 2015/01/07.
//  Copyright (c) 2014-2015 Nozomu Ishiguro. All rights reserved.
//

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


//Image Reg mode
#ifndef REGMODE
#define REGMODE 0
#endif
//XYshift
#if REGMODE == 0

#define NUM_PARA 2

#define TRANS_XY(XYZ,p,x,y,z,mr)\
(XYZ) = (float4)((x)+((float*)(p))[0]/(mr),(y)+((float*)(p))[1]/(mr),(z),0)

#define JACOBIAN_P(j,p,x,y,dfdx,dfdy,msk,mr)\
{\
((float*)(j))[0] = (dfdx)*(msk)/(mr);\
((float*)(j))[1] = (dfdy)*(msk)/(mr);\
}

//XYshift+rotation
#elif REGMODE == 1

#define NUM_PARA 3

#define TRANS_XY(XYZ,p,x,y,z,mr)\
(XYZ) = (float4)(\
cos(((float*)(p))[2])*(x)-sin(((float*)(p))[2])*(y)+((float*)(p))[0]/(mr),\
sin(((float*)(p))[2])*(x)+cos(((float*)(p))[2])*(y)+((float*)(p))[1]/(mr),\
(z),0)

#define JACOBIAN_P(j,p,x,y,dfdx,dfdy,msk,mr)\
{\
float DxDth = (-(x)*sin(((float*)(p))[2])-(y)*cos(((float*)(p))[2]));\
float DyDth = ( (x)*cos(((float*)(p))[2])-(y)*sin(((float*)(p))[2]));\
\
((float*)(j))[0] = (dfdx)*(msk)/(mr);\
((float*)(j))[1] = (dfdy)*(msk)/(mr);\
((float*)(j))[2] = ((dfdx)*DxDth+(dfdy)*DyDth)*(msk);\
}


//XYshift+scale
#elif REGMODE == 2

#define NUM_PARA 3

#define TRANS_XY(XYZ,p,x,y,z,mr)\
(XYZ) = (float4)(\
exp(((float*)(p))[2])*(x)+((float*)(p))[0]/(mr),\
exp(((float*)(p))[2])*(y)+((float*)(p))[1]/(mr),\
(z),0)

#define JACOBIAN_P(j,p,x,y,dfdx,dfdy,msk,mr)\
{\
float g = ((float*)(p))[2];\
\
((float*)(j))[0] = (dfdx)*(msk)/(mr);\
((float*)(j))[1] = (dfdy)*(msk)/(mr);\
((float*)(j))[2] = exp(g)*((dfdx)*(x)+(dfdy)*(y))*(msk);\
}


//XYshift+ratation+scale
#elif REGMODE == 3

#define NUM_PARA 4

#define TRANS_XY(XYZ,p,x,y,z,mr)\
(XYZ) = (float4)(\
(cos(((float*)(p))[2])*(x)-sin(((float*)(p))[2])*(y))*exp(((float*)(p))[3])+((float*)(p))[0]/(mr),\
(sin(((float*)(p))[2])*(x)+cos(((float*)(p))[2])*(y))*exp(((float*)(p))[3])+((float*)(p))[1]/(mr),\
(z),0)

#define JACOBIAN_P(j,p,x,y,dfdx,dfdy,msk,mr)\
{\
float g = ((float*)(p))[3]\
float DxDth = (-(x)*sin(((float*)(p))[2])-(y)*cos(((float*)(p))[2]))*exp(g);\
float DyDth = ( (x)*cos(((float*)(p))[2])-(y)*sin(((float*)(p))[2]))*exp(g);\
float DxDs  = ( (x)*cos(((float*)(p))[2])-(y)*sin(((float*)(p))[2]))*exp(g);\
float DyDs  = ( (x)*sin(((float*)(p))[2])+(y)*cos(((float*)(p))[2]))*exp(g);\
\
((float*)(j))[0] = (dfdx)*(msk)/(mr);\
((float*)(j))[1] = (dfdy)*(msk)/(mr);\
((float*)(j))[2] = ((dfdx)*DxDth+(dfdy)*DxDth)*(msk);\
((float*)(j))[3] = ((dfdx)*DxDs +(dfdy)*DyDs )*(msk);\
}


//XYshift+affine
#elif REGMODE == 4

#define NUM_PARA 6

#define TRANS_XY(XYZ,p,x,y,z,mr)\
(XYZ) = (float4)(\
(1+(float*)(p))[2])*(x)+(  (float*)(p))[3])*(y)+((float*)(t))[0]/(mr),\
(  (float*)(p))[4])*(x)+(1+(float*)(p))[5])*(y)+((float*)(t))[1]/(mr),\
(z),0)

#define JACOBIAN_P(j,p,x,y,dfdx,dfdy,msk,mr)\
{\
((float*)(j))[0] = (dfdx)*(msk)/(mr);\
((float*)(j))[1] = (dfdy)*(msk)/(mr);\
((float*)(j))[2] = (dfdx)*(x)*(msk);\
((float*)(j))[3] = (dfdx)*(y)*(msk);\
((float*)(j))[4] = (dfdy)*(x)*(msk);\
((float*)(j))[5] = (dfdy)*(y)*(msk);\
}

#endif


/*contrast factor mode*/
#ifndef CNTMODE
#define CNTMODE 0
#endif
//no contrast factor
#if CNTMODE == 0

#define NUM_CPARA 0

#define CONTRAST_F(cnt,bkg,x,y,p,mr)\
{\
(cnt)=1.0f;\
(bkg)=0.0f;\
}

#define JACOBIAN_C(j,p,x,y,f,msk,mr)\
{\
}


//contrast(exp(g)) + bkg(const)
#elif CNTMODE == 1

#define NUM_CPARA 2

#define CONTRAST_F(cnt,bkg,x,y,p,mr)\
{\
(cnt)=exp(((float*)(p))[NUM_PARA]);\
(bkg)=((float*)(p))[NUM_PARA+1];\
}

#define JACOBIAN_C(j,p,x,y,f,msk,mr)\
{\
((float*)(j))[NUM_PARA]   = (f)*(msk);\
((float*)(j))[NUM_PARA+1] = (msk);\
}


//contrast(exp(g)) + bkg(s*x+t*y+const: 1st order plane)
#elif CNTMODE == 2

#define NUM_CPARA 4

#define CONTRAST_F(cnt,bkg,x,y,p,mr)\
{\
(cnt)=exp(((float*)(p))[NUM_PARA]);\
(bkg)=((float*)(p))[NUM_PARA+1]+\
((float*)(p))[NUM_PARA+2]*(x)*(mr)+((float*)(p))[NUM_PARA+3]*(y)*(mr);\
}

#define JACOBIAN_C(j,p,x,y,f,msk,mr)\
{\
((float*)(j))[NUM_PARA]   = (f)*(msk);\
((float*)(j))[NUM_PARA+1] = (msk);\
((float*)(j))[NUM_PARA+2] = (x)*(mr)*(msk);\
((float*)(j))[NUM_PARA+3] = (y)*(mr)*(msk);\
}

#endif

#define P_NUM  NUM_PARA+NUM_CPARA
#define P_NUM_SQ (NUM_PARA+NUM_CPARA)*(NUM_PARA+NUM_CPARA)

//sampler
__constant sampler_t s_linear = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;
__constant sampler_t s_linear_cEdge = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE;


//reduction
inline float reduction(__local float *loc_mem, const size_t local_ID, const size_t localsize)
{
    for(size_t s=localsize/2;s>0;s>>=1){
        if(local_ID<s){
            loc_mem[local_ID]+=loc_mem[local_ID+s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    float res = loc_mem[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    return res;
}

//calc tJJ & tJdf
inline void calc_tJJ_tJdF(float *tJJ,float *tJdF,float *dp,float *p_err_pr,float *lamda_diag_tJJ,
                          __local float *loc_mem, __constant float *p_fix,
                           const size_t local_ID, const size_t localsize,
                           size_t dim, float lambda, float epsilon, float dev)
{
    for(size_t i=0;i<dim;i++){
        loc_mem[local_ID] = tJJ[i+i*dim];
        barrier(CLK_LOCAL_MEM_FENCE);
        tJJ[i+i*dim] = reduction(loc_mem,local_ID,localsize);
        barrier(CLK_LOCAL_MEM_FENCE);
        p_err_pr[i] = (p_fix[i]==0.0f)? 0.0f:fmax(sqrt(1.0f/tJJ[i+i*dim]),0.01f);
        tJJ[i+i*dim] *= (1+lambda)*p_fix[i];
        
        for(size_t j=i+1;j<dim;j++){
            loc_mem[local_ID] = tJJ[i+j*dim];
            barrier(CLK_LOCAL_MEM_FENCE);
            tJJ[i+j*dim] = reduction(loc_mem,local_ID,localsize);
            barrier(CLK_LOCAL_MEM_FENCE);
            tJJ[i+j*dim] *= p_fix[j]*p_fix[i];
            tJJ[j+i*dim] = tJJ[i+j*dim];
        }
        
        loc_mem[local_ID] = tJdF[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        tJdF[i] = reduction(loc_mem,local_ID,localsize);
        barrier(CLK_LOCAL_MEM_FENCE);
        tJdF[i] *= p_fix[i];
    }
    
    for(size_t i=0;i<dim;i++){
        tJJ[i+i*dim] /= dev;
        lamda_diag_tJJ[i]=tJJ[i+i*dim]*lambda;
        
        for(size_t j=i+1;j<dim;j++){
            tJJ[i+j*dim] /= dev;
            tJJ[j+i*dim] /= dev;
        }

        tJdF[i] /= dev;
        dp[i]=tJdF[i];
    }
    
}

//simultaneous linear equation
static inline void sim_linear_eq(float *A, float *x, float *inv_A,
                                 size_t dim,__constant char *p_fix){
    
    float a = 0.0f;
    
    //inversed matrix
    for(int i=0; i<dim; i++){
        if(p_fix[i]== 48) continue;
        
        //devide (i,i) to 1
        a = 1/A[i+i*dim];
        
        for(int j=0; j<dim; j++){
            if(p_fix[j]== 48) continue;
            A[i+j*dim] *= a;
            inv_A[i+j*dim] *= a;
        }
        x[i] *= a;
        
        //erase (j,i) (i!=j) to 0
        for(int j=i+1; j<dim; j++){
            if(p_fix[j]== 48) continue;
            
            a = A[j+i*dim];
            for(int k=0; k<dim; k++){
                if(p_fix[k]== 48) continue;
                A[j+k*dim] -= a*A[i+k*dim];
                inv_A[j+k*dim] -= a*inv_A[i+k*dim];
            }
            x[j] -= a*x[i];
        }
    }
    for(int i=0; i<dim; i++){
        if(p_fix[i]== 48) continue;
        
        for(int j=i+1; j<dim; j++){
            if(p_fix[j]== 48) continue;
            
            a = A[i+j*dim];
            for(int k=0; k<dim; k++){
                if(p_fix[k]== 48) continue;
                A[k+j*dim] -= a*A[k+i*dim];
                inv_A[k+j*dim] -= a*inv_A[k+i*dim];
            }
            x[i] -= a*x[j];
        }
    }
}


//kernel
//mt conversion
__kernel void mt_conversion(__global float *dark, __global float *I0,
                            __global ushort *It_buffer, __global float *mt_buffer,
                            __write_only image2d_array_t mt_img1,
                            __write_only image2d_array_t mt_img2,
                            int shapeNo,int startpntX, int startpntY,
                            uint width, uint height, float angle, int evaluatemode){
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_group_id(0);
    const size_t Y = get_global_id(1);
    int X,ID,IDxy;
    int4 XYZ;
    float4 mt_f1,mt_f2;
    float mask=1.0f;
    float radius2=0;
    float trans, mt, It, dark1;
    float absX, absY;
    
    
    for(size_t i=0;i<IMAGESIZE_X/localsize;i++){
        X=local_ID+i*localsize;
        ID=X+IMAGESIZE_X*Y+Z*IMAGESIZE_M;
        IDxy=X+IMAGESIZE_X*Y;
        XYZ=(int4)(X,Y,Z,0);
            
        absX = (X-startpntX)*cos(angle/180*(float)PI)-(Y-startpntY)*sin(angle/180*(float)PI);
        absX = fabs(absX);
        absY = (X-startpntX)*sin(angle/180*(float)PI)+(Y-startpntY)*cos(angle/180*(float)PI);
        absY = fabs(absY);
        switch(shapeNo){
            case 0: //square or rectangle
                mask=(absX<=width/2 & absY<=height/2) ? 1.0f:0.0f;
                break;
                    
            case 1: //circle or orval
                radius2=(float)(absX*absX)/width/width+(float)(absY*absY)/height/height;
                mask=(radius2<=0.25f) ? 1.0f:0.0f;
                break;
                
            default:
                mask=1.0f;
                break;
        }
        
        It = (float)It_buffer[ID+32*Z];
        dark1 = dark[IDxy];
        trans = (I0[IDxy]-dark1)/(It-dark1);
        trans = (trans < 1.0f) ? 1.0f:trans;
        trans = (trans > 1.0E7f) ? 1.0E7f:trans;
        mt = log(trans);
        
        switch(evaluatemode){
            case 0: //mt
                mt_f1 = (float4)(mt,mask,0.0f,0.0f);
                break;
                
            case 1: // trans^(-1)-1
                mt_f1 = (float4)(1/trans-1,mask,0.0f,0.0f);
                break;
                
            case 2: // It
                mt_f1 = (float4)(1/It,mask,0.0f,0.0f);
                break;
                
            default: //mt
                mt_f1 = (float4)(mt,mask,0.0f,0.0f);
                break;
        }
        mt_f1 = (float4)(mt,mask,0.0f,0.0f);
        mt_f2 = (float4)(mt,1.0f,0.0f,0.0f);
        write_imagef(mt_img1,XYZ,mt_f1);
        write_imagef(mt_img2,XYZ,mt_f2);
        mt_buffer[ID]=mt;
    }
    
}


//mt transfer
__kernel void mt_transfer(__global float *mt_buffer,
                            __write_only image2d_array_t mt_img1,
                            __write_only image2d_array_t mt_img2,
                            int shapeNo,int startpntX, int startpntY,
                            uint width, uint height, float angle){
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const int Z = get_group_id(0);
    const int Y = get_global_id(1);
    int X,ID;
    int4 XYZ;
    float4 mt_f1,mt_f2;
    float mask=1.0f;
    float radius2=0.0f;
    float mt;
    float absX, absY;
    float angle_rad = angle/180.0f*PI;
    
    
    for(size_t i=0;i<IMAGESIZE_X/localsize;i++){
        X=local_ID+i*localsize;
        ID=X+IMAGESIZE_X*Y+Z*IMAGESIZE_M;
        XYZ=(int4)(X,Y,Z,0);
        
        absX = (float)(X-startpntX)*cos(angle_rad)-(float)(Y-startpntY)*sin(angle_rad);
        absX = fabs(absX);
        absY = (float)(X-startpntX)*sin(angle_rad)+(float)(Y-startpntY)*cos(angle_rad);
        absY = fabs(absY);
        switch(shapeNo){
            case 0: //square or rectangle
                mask=(absX<=width/2 & absY<=height/2) ? 1.0f:0.0f;
                break;
                
            case 1: //circle or orval
                radius2=(float)(absX*absX)/width/width+(float)(absY*absY)/height/height;
                mask=(radius2<=0.25f) ? 1.0f:0.0f;
                break;
                
            default:
                mask=1.0f;
                break;
        }
        
        mt = (float)mt_buffer[ID];
        mt_f1 = (float4)(mt,mask,0.0f,0.0f);
        mt_f2 = (float4)(mt,1.0f,0.0f,0.0f);
        write_imagef(mt_img1,XYZ,mt_f1);
        write_imagef(mt_img2,XYZ,mt_f2);
        //mt_buffer[ID]=mt;
    }
    
}


//create merged image
__kernel void merge(__read_only image2d_array_t input_img, __write_only image2d_array_t output_img,
                    unsigned int mergeN){
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    
    float4 img;
    float4 XYZ_in;
    int4 XYZ_out;
    int X, Y;
    
    for(size_t i=0;i<IMAGESIZE_Y/mergeN;i++){
        for(size_t j=0;j<IMAGESIZE_X/mergeN/localsize;j++){
            X = local_ID+j*localsize;
            Y = i;
            img = (float4)(0.0f,0.0f,0.0f,0.0f);
            XYZ_out = (int4)(X,Y,group_ID,0);
            for(size_t k=0; k<mergeN; k++){
                for(size_t l=0; l<mergeN; l++){
                    XYZ_in = (float4)(k+X*mergeN,l+Y*mergeN,group_ID,0.0f);
                    img += read_imagef(input_img,s_linear,XYZ_in);
                }
            }
            //if(img.y>1) printf("X%dY%d: %f,%f \n",X,Y,img.x,img.y);
            img /= mergeN*mergeN;
            img.y = (img.y>0.5) ? 1.0f:0.0f;
            write_imagef(output_img,XYZ_out,img);
        }
    }
}


//image registration
__kernel void imageRegistration(__read_only image2d_array_t mt_t_img,
                                __read_only image2d_array_t mt_s_img,
                                __global float *lambda_buffer,
                                __global float *p,__global float *p_err,
                                __constant float *p_target,__constant float *p_fix,
                                __local float *loc_mem, unsigned int mergeN, float epsilon)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    
    size_t X,Y;
    float4 XYZ_s, XYZ_t;
    float4 mt_s_f4, mt_t_f4;
    float mt_s_f, mt_t_f;
    
    float4 XYZ_s_xp, XYZ_s_xm;
    float mt_s_f4_xp, mt_s_f4_xm;
    float4 XYZ_s_yp, XYZ_s_ym;
    float mt_s_f4_yp, mt_s_f4_ym;
    
    float msk;
    float cnt, bkg;
    float dF_dx, dF_dy;
    float J[P_NUM];
    float dF;
    float tJJ[P_NUM_SQ];
    float inv_tJJ[P_NUM_SQ];
    float tJdF[P_NUM],dp[P_NUM],lamda_diag_tJJ[P_NUM];
    float mergeN_f=(float)mergeN;
    float p_pr[P_NUM], p_cdt[P_NUM], p_err_pr[P_NUM];
    float p_target_pr[P_NUM];
    float dev;
    
    //initialize LM parameter
    float chi2_new, chi2;
    float lambda,rho=0.0f,d_L=0.0f,nyu=2.0f;
    float l_A, l_B;
    
    
    //copy data from global lambda_buffer to private lambda
    lambda = lambda_buffer[group_ID];
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
   
    int i,j,k,l;
    //copy data from global p to private p_pr
    for(i=0;i<P_NUM;i++){
        p_pr[i]=p[i+group_ID*P_NUM];
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
        
        p_target_pr[i]=p_target[i+group_ID*P_NUM];
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    }
    
    
    //initilize tJJ, tJdF
    loc_mem[local_ID]=0.0f;
    for(i=0;i<P_NUM;i++){
        for(j=0;j<P_NUM;j++){
            tJJ[i+j*P_NUM]=0.0f;
        }
        tJdF[i]=0.0f;
    }
    
    
    //estimate tJJ, tJdF, chi2 (Integrate by x,y)
    chi2=0.0f;
    dev =0.0f;
    int diffStep = 2;
    float diffweight = 3.0f/diffStep/(diffStep+1)/(2*diffStep+1);
    for(j=0;j<IMAGESIZE_Y/mergeN;j++){
        for(i=0;i<IMAGESIZE_X/mergeN/localsize;i++){
            X=local_ID+i*localsize;
            Y=j;
            
            //XYZ_t = (float4)(X,Y,group_ID,0);
            TRANS_XY(XYZ_t,p_target_pr,X,Y,group_ID,mergeN_f);
            TRANS_XY(XYZ_s,p_pr,X,Y,group_ID,mergeN_f);
            CONTRAST_F(cnt,bkg,X,Y,p_pr,mergeN_f);
            
            mt_t_f4 = read_imagef(mt_t_img,s_linear_cEdge,XYZ_t);
            mt_s_f4 = read_imagef(mt_s_img,s_linear_cEdge,XYZ_s);
            
            msk = mt_t_f4.y*mt_s_f4.y;
            
            mt_t_f = mt_t_f4.x;
            mt_s_f = mt_s_f4.x*cnt;
            
            
            mt_s_f4_xp = 0.0f;
            mt_s_f4_xm = 0.0f;
            mt_s_f4_yp = 0.0f;
            mt_s_f4_ym = 0.0f;
            for(k=1;k<=diffStep;k++){
                //Partial differential dF/dx
                TRANS_XY(XYZ_s_xp,p_pr,X+k,Y,group_ID,mergeN_f);
                TRANS_XY(XYZ_s_xm,p_pr,X-k,Y,group_ID,mergeN_f);
                mt_s_f4_xp += k*read_imagef(mt_s_img,s_linear_cEdge,XYZ_s_xp).x;
                mt_s_f4_xm += k*read_imagef(mt_s_img,s_linear_cEdge,XYZ_s_xm).x;
            
                //Partial differential dF/dy
                TRANS_XY(XYZ_s_yp,p_pr,X,Y+k,group_ID,mergeN_f);
                TRANS_XY(XYZ_s_ym,p_pr,X,Y-k,group_ID,mergeN_f);
                mt_s_f4_yp += k*read_imagef(mt_s_img,s_linear_cEdge,XYZ_s_yp).x;
                mt_s_f4_ym += k*read_imagef(mt_s_img,s_linear_cEdge,XYZ_s_ym).x;
            }
            dF_dx = (mt_s_f4_xp - mt_s_f4_xm)*cnt*diffweight;
            dF_dy = (mt_s_f4_yp - mt_s_f4_ym)*cnt*diffweight;
            
            //Jacobian=dF/dp_i, dF= F_T(x,y) - F(x,y;p), chi2 = (dF)^2
            JACOBIAN_P(J,p_pr,X,Y,dF_dx,dF_dy,msk,mergeN_f);
            JACOBIAN_C(J,p_pr,X,Y,mt_s_f,msk,mergeN_f);
            dF = (mt_t_f - mt_s_f - bkg)*msk;
            for(k=0;k<P_NUM;k++){
                tJJ[k+k*P_NUM] += J[k]*J[k];
                inv_tJJ[k+k*P_NUM] = 1.0f;
                for(l=k+1;l<P_NUM;l++){
                    tJJ[l+k*P_NUM] += J[k]*J[l];
                    tJJ[k+l*P_NUM] = tJJ[l+k*P_NUM];
                    inv_tJJ[l+k*P_NUM] = 0.0f;
                    inv_tJJ[k+l*P_NUM] = 0.0f;
                }
                tJdF[k] += J[k]*dF;
            }
            chi2 += dF*dF;
            dev += msk;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#if DEBUG
    if(get_global_id(0)==256){
        for(i=0;i<P_NUM;i++){
            for(j=0;j<P_NUM;j++){
                printf("tJJ[%d][%d]=%e ",i,j,tJJ[i+P_NUM*j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
    loc_mem[local_ID]=dev;
    dev = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    calc_tJJ_tJdF(tJJ,tJdF,dp,p_err_pr,lamda_diag_tJJ,
                  loc_mem, p_fix,
                  local_ID,localsize,P_NUM,lambda,epsilon,dev);
    barrier(CLK_LOCAL_MEM_FENCE);
    loc_mem[local_ID]=chi2;
    chi2 = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    chi2 /= dev;
    
    
#if DEBUG
    if(get_global_id(0)==0){
        for(i=0;i<P_NUM;i++){
            for(j=0;j<P_NUM;j++){
                printf("tJJ[%d][%d]=%e ",i,j,tJJ[i+P_NUM*j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    //solve dp of sim. linear eq. [ tJJ x dp = tJdF ]
    sim_linear_eq(tJJ, dp, P_NUM,p_fix,inv_tJJ);
#if DEBUG
    if(get_global_id(0)==0){
        for(i=0;i<P_NUM;i++){
            for(j=0;j<P_NUM;j++){
                printf("tJJ[%d][%d]=%e ",i,j,tJJ[i+P_NUM*j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_global_id(0)==0){
        for(i=0;i<P_NUM;i++){
            printf("dp[%d]=%f ",i,dp[i]);
        }
        printf("\n\n");
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    
    //estimate update candidate chi2_new (Integrate by x,y)
    for(i=0;i<P_NUM;i++){
        p_cdt[i] = p_pr[i] + dp[i];
    }
    chi2_new = 0.0f;
    dev =0.0f;
    for(j=0;j<IMAGESIZE_Y/mergeN;j++){
        for(i=0;i<IMAGESIZE_X/mergeN/localsize;i++){
            X=local_ID+i*localsize;
            Y=j;
            //XYZ_t = (float4)(X,Y,group_ID,0);
            TRANS_XY(XYZ_t,p_target_pr,X,Y,group_ID,mergeN_f);
            TRANS_XY(XYZ_s,p_cdt,X,Y,group_ID,mergeN_f);
            CONTRAST_F(cnt,bkg,X,Y,p_cdt,mergeN_f);
            
            mt_t_f4 = read_imagef(mt_t_img,s_linear_cEdge,XYZ_t);
            mt_s_f4 = read_imagef(mt_s_img,s_linear_cEdge,XYZ_s);
            
            msk = mt_t_f4.y*mt_s_f4.y;
            
            mt_t_f = mt_t_f4.x;
            mt_s_f = mt_s_f4.x*cnt;
            dF     = (mt_t_f - mt_s_f - bkg)*msk;
            chi2_new  += dF*dF;
            dev += msk;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    loc_mem[local_ID]=dev;
    dev = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    loc_mem[local_ID]=chi2_new;
    chi2_new = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    chi2_new /=dev;
    
    //update lambda & p_pr(if rho>0)
    d_L=0.0f;
    for(i=0;i<P_NUM;i++){
        d_L += dp[i]*(lamda_diag_tJJ[i]*dp[i]+tJdF[i]);
    }
    rho = (chi2-chi2_new)/d_L;  //gain ratio
    l_A = 1.0f-(2.0f*rho-1.0f)*(2.0f*rho-1.0f)*(2.0f*rho-1.0f);
    l_B = max(0.333f,l_A);
#if DEBUG
    if(get_global_id(0)==0){
        printf("chi2=%e\n",chi2);
        printf("chi2_new=%e\n",chi2_new);
        printf("d_L=%e\n",d_L);
        printf("rho=%e\n",rho);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#endif
    bool update = (rho>0.0f)? 1:0;
    lambda *= (update) ? l_B:nyu;
    nyu     = (update) ? 2.0f:2.0f*nyu;
    for(i=0;i<P_NUM;i++){
        p_pr[i] = (update) ? p_cdt[i]:p_pr[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    
    //copy lambda to global lambda_buffer
    //copy chi2 to global chi2_buffer
    //copy data from local p_pr to global p
    //copy data from local p_err_pr to global p_err
    if(local_ID==0){
        lambda_buffer[group_ID]=lambda;
        for(i=0;i<P_NUM;i++){
            p[i+group_ID*P_NUM]=p_pr[i];
            p_err[i+group_ID*P_NUM]=p_err_pr[i];
        }
#if DEBUG
        for(i=0;i<P_NUM;i++){
            printf("p[%d]=%f ",i,p[i]);
        }
        printf("\n\n");
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}






//output final image reg results
__kernel void output_imgReg_result(__read_only image2d_array_t mt_img, __global float *mt_buf,
                                   __global float *p){
    
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_group_id(0);
    const size_t Y = get_global_id(1);
    
    size_t X,ID;
    float4 XYZ, img;
    float p_pr[P_NUM];
    
    
    //copy data from global transpara to local transpara_atE
    for(size_t i=0;i<P_NUM;i++){
        p_pr[i]=p[i+Z*P_NUM];
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    }
    
    
    //convert image reg results
    for(size_t i=0;i<IMAGESIZE_X/localsize;i++){
        X = local_ID+i*localsize;
        TRANS_XY(XYZ,p_pr,X,Y,Z,1);
        ID = X+Y*IMAGESIZE_X+Z*IMAGESIZE_M;
        
        img = read_imagef(mt_img,s_linear,XYZ);
        mt_buf[ID]=img.x*img.y;
    }
}

//merge image reg images
__kernel void merge_mt(__read_only image2d_array_t mt_sample, __global float *mt_output)
{
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t globalsize_x = get_global_size(0);
    const size_t global_ID = global_x+globalsize_x*global_y;
    float4 XYZ;
    float mt=0;
    
    
    const size_t mergesize = get_image_array_size(mt_sample);
    for(size_t i=0;i<mergesize;i++){
        XYZ=(float4)(global_x,global_y,i,0);
        mt+=read_imagef(mt_sample,s_linear,XYZ).x;
    }
    mt_output[global_ID]=mt;
}


//merge raw his data
__kernel void merge_rawhisdata(__global ushort *rawhisdata, __global float *outputdata,
                               int mergeN) {
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t globalsize_x = get_global_size(0);
    const size_t global_ID = global_x+globalsize_x*global_y;
    
    float value = 0.0f;
    for(int i=0;i<mergeN;i++){
        value += (float)rawhisdata[global_ID+(IMAGESIZE_M+32)*i];
    }
    value /= mergeN;
    outputdata[global_ID]=value;
}

//QXAFS smoothing
__kernel void imQXAFS_smoothing(__global float *rawmtdata, __global float *outputdata,
                               int mergeN) {
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t globalsize_x = get_global_size(0);
    const size_t globalsize_y = get_global_size(1);
    const size_t globalsize_z = get_global_size(2);
    const size_t global_ID = global_x+globalsize_x*global_y+globalsize_x*globalsize_y*global_z;
    const size_t imagesize = globalsize_x*globalsize_y*globalsize_z;
    
    float value = 0.0f;
    for(int i=0;i<mergeN;i++){
        value += rawmtdata[global_ID+imagesize*i];
    }
    value /= mergeN;
    outputdata[global_ID]=value;
}



