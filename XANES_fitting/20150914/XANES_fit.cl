//
//  XANES_fit.cl
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/01/07.
//  Copyright (c) 2014-2015 Nozomu Ishiguro. All rights reserved.
//

#define PI 3.14159265358979323846

#ifndef IMAGESIZE_X
#define IMAGESIZE_X 2048
#endif

#ifndef IMAGESIZE_Y
#define IMAGESIZE_Y 2048
#endif

#ifndef IMAGESIZE_M
#define IMAGESIZE_M 4194304 //2048*2048
#endif

#ifndef NUM_TRIAL
#define NUM_TRIAL 20
#endif

#ifndef PARA_NUM
#define PARA_NUM 1
#endif

#ifndef ENERGY_NUM
#define ENERGY_NUM 1
#endif

#ifndef E0
#define E0 11559.0
#endif

#ifndef LAMBDA
#define LAMBDA 0.001f
#endif



#ifndef FIT_AND_JACOBIAN(x,mt,j,fp,msk)
#define FIT_AND_JACOBIAN(x,mt,j,fp,msk){\
\
float a_X = ((x)-((float*)(fp))[3])/((float*)(fp))[4];\
float l_X = ((x)-((float*)(fp))[6])/((float*)(fp))[7];\
\
(mt) = ((float*)(fp))[0] + ((float*)(fp))[1]*(x)\
+((float*)(fp))[2]*(0.5f+atanpi(a_X)\
+((float*)(fp))[5]/(1+l_X*l_X));\
(mt)*=(msk);\
\
((float*)(j))[0]=1.0;\
((float*)(j))[1]=(x);\
((float*)(j))[2]=0.5f+atanpi(a_X)+((float*)(fp))[5]/(1+l_X*l_X);\
((float*)(j))[3]=-((float*)(fp))[2]/PI/((float*)(fp))[4]/(1+a_X*a_X);\
((float*)(j))[4]=-a_X*((float*)(fp))[2]/PI/((float*)(fp))[4]/(1+a_X*a_X);\
((float*)(j))[5]=((float*)(fp))[2]/(1+l_X*l_X);\
((float*)(j))[6]=l_X*2*((float*)(fp))[2]*((float*)(fp))[5]/((float*)(fp))[7]/(1+l_X*l_X)/(1+l_X*l_X);\
((float*)(j))[7]=l_X*l_X*2*((float*)(fp))[2]*((float*)(fp))[5]/((float*)(fp))[7]/(1+l_X*l_X)/(1+l_X*l_X);\
\
}
#endif

#ifndef INITIAL_CHI2_OLD(x,fx,fp,mt)
#define INITIAL_CHI2_OLD(x,fx,fp,mt){\
\
float a_X = ((x)-((float*)(fp))[3])/((float*)(fp))[4];\
float l_X = ((x)-((float*)(fp))[6])/((float*)(fp))[7];\
\
float mt_f=((float*)(fp))[0] + ((float*)(fp))[1]*(x)\
+((float*)(fp))[2]*(0.5f+atanpi(a_X)\
+((float*)(fp))[5]/(1+l_X*l_X));\
\
(fx) += ((mt)-mt_f)*((mt)-mt_f);\
\
}
#endif



__constant sampler_t s_linear = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;



//simultaneous linear equation
inline void sim_linear_eq(float *A, float *x, size_t dim){
    
    float a = 0.0;
    
    //inversed matrix
    for(int i=0; i<dim; i++){
        //devide (i,i) to 1
        a = 1/A[i+i*dim];
        
        for(int j=0; j<dim; j++){
            A[i+j*dim] *= a;
        }
        x[i] *= a;
        
        //erase (j,i) (i!=j) to 0
        for(int j=i+1; j<dim; j++){
            a = A[j+i*dim];
            for(int k=0; k<dim; k++){
                A[j+k*dim] -= a*A[i+k*dim];
            }
            x[j] -= a*x[i];
        }
    }
    for(int i=0; i<dim; i++){
        for(int j=i+1; j<dim; j++){
            a = A[i+j*dim];
            for(int k=0; k<dim; k++){
                A[k+j*dim] -= a*A[k+i*dim];
            }
            x[i] -= a*x[j];
        }
    }
}



__kernel void XANES_fitting(__read_only image2d_array_t mt_img,
                             __read_only image2d_array_t fp_img,
                             __write_only image2d_array_t fp_dest_img,
                             __global float *energy,
                             __global float *lambda_buffer,
                             __global float *nyu_buffer)
{
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t globalsizex = get_global_size(0);
    const size_t global_ID = global_x + global_y*globalsizex;
    const size_t local_ID = get_local_id(0);
    float4 XYZf;
    int4 XYZi;
    float4 fit_r;
    
    //mt_data, mit_fit
    float mt_data[ENERGY_NUM];
    float mt_fit, dF;
    char mask=1;
    
    //LM parameters
    float X;
    float J[PARA_NUM];
    float tJJ[PARA_NUM*PARA_NUM];
    float tJdF[PARA_NUM];
    float fp[PARA_NUM], dp[PARA_NUM];
    float energy_pr[ENERGY_NUM];
    
    //dumping parameters
    float chi2_new, chi2_old=0.0;
    float lambda=lambda_buffer[global_ID];
    float nyu   =nyu_buffer[global_ID];
    float rho, d_rho;
    float l_A, l_B, l_C;
    float n_A, n_B;
    float updatecheck;
    
    
    //copy initial fitting parameter from constant memory to private memory
    for(int i=0; i<PARA_NUM;i++){
        XYZf=(float4)(global_x, global_y, i, 0);
        fp[i] = read_imagef(fp_img,s_linear,XYZf).x;
    }
    
    
    //transfer global mt_data to private memory
    for(int i=0; i<ENERGY_NUM; i++){
        XYZf=(float4)(global_x, global_y, i, 0);
        mt_data[i]=read_imagef(mt_img,s_linear,XYZf).x;
    }
    
    
    //initialize chi2_old & copy energy from global to private memory
    for(int i=0; i<ENERGY_NUM; i++){
        energy_pr[i]=energy[i];
        barrier(CLK_GLOBAL_MEM_FENCE);
        X=energy_pr[i];
        
        if (mt_data[i]==0.0) mask*=0;
        else mask *=1;
        
        INITIAL_CHI2_OLD(X,chi2_old,fp,mt_data[i]);
    }
    

    //Initialize tJJ, tJdF, fx_new
    for(int i=0; i<PARA_NUM; i++){
        tJJ[i+i*PARA_NUM]=0.0;
        tJdF[i]=0.0;
        chi2_new = 0.0;
            
        for(int j=i+1; j<PARA_NUM; j++){
            tJJ[i+j*PARA_NUM]=0.0;
            tJJ[j+i*PARA_NUM]=0.0;
        }
    }
        
    
    //Estimate Jacobian
    for(int i=0; i<ENERGY_NUM; i++){
            
        //mask data
        if (mt_data[i]==0.0) mask*=0;
        else mask *=1;
            
            
        //mt_fit (trial)
        X=energy_pr[i];
        FIT_AND_JACOBIAN(X,mt_fit,J,fp,mask);
            
            
        //delta F = mt_data - mt_fit
        dF = (mt_data[i]-mt_fit);
            
        
        //tJJ & tJdF & chi2_new
        for(int i=0; i<PARA_NUM; i++){
            tJJ[i+i*PARA_NUM] += J[i]*J[i]*(1+lambda);  //L-M method
            tJdF[i] += J[i]*dF;
            chi2_new += dF*dF;
            dp[i] = tJdF[i];
                
            for(int j=i+1; j<PARA_NUM; j++){
                tJJ[i+j*PARA_NUM] += J[i]*J[j];
                tJJ[j+i*PARA_NUM]  = tJJ[i+j*PARA_NUM];
            }
        }
    }
        
        
    //solve dp of sim. linear eq. [ tJJ x dp = tJdF ]
    sim_linear_eq(tJJ,dp,PARA_NUM);
    
    
    //update lambda & dp(if rho>0)
    d_rho = 0;
    for(int i=0; i<PARA_NUM; i++){
        d_rho += dp[i]*(dp[i]*lambda + tJdF[i]);
    }
    rho = (chi2_old-chi2_new)/d_rho;
    l_A = 1.0f-(2.0f*rho-1.0f)*(2.0f*rho-1.0f)*(2.0f*rho-1.0f);
    l_B = lambda*max(0.333f,l_A);
    l_C = lambda*nyu;
    n_A = 2.0f;
    n_B = nyu*2.0f;
    if(rho>0.0f){  //step accetable
        lambda  = l_B;
        nyu     = l_A;
        updatecheck =1.0f;
    }else{
        lambda = l_C;
        nyu    = n_B;
        updatecheck =0.0f;
    }
    barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
    lambda_buffer[global_ID]=lambda;
    nyu_buffer[global_ID]=nyu;
    
    //output dp_img
    for(int i=0; i<PARA_NUM; i++){
        XYZi=(int4)(global_x, global_y, i, 0);
        fit_r=(float4)(fp[i]+dp[i]*updatecheck,0,0,0);
        write_imagef(fp_dest_img,XYZi,fit_r);
    }
}


__kernel void outputFitParaImage(__read_only image2d_array_t fp_img,
                                 __write_only image2d_array_t fp_dest_img,
                                 __constant float *para_lower_limit,
                                 __constant float *para_upper_limit){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t globalsizex = get_global_size(0);
    const size_t global_ID = global_x + global_y*globalsizex;
    const size_t local_ID = get_local_id(0);
    float4 XYZf;
    int4 XYZi;
    float4 fit_r;
    float fp[PARA_NUM];
    
    //copy initial fitting parameter from constant memory to private memory
    for(int i=0; i<PARA_NUM;i++){
        XYZf=(float4)(global_x, global_y, i, 0);
        fp[i] = read_imagef(fp_img,s_linear,XYZf).x;
    }
    
    //output fit result img
    for(int i=0; i<PARA_NUM; i++){
        if(isnan(fp[i])) fp[i]=0;
        else if(fp[i]<para_lower_limit[i]) fp[i]=0;
        else if(fp[i]>para_upper_limit[i]) fp[i]=para_upper_limit[i];
    
        XYZi=(int4)(global_x, global_y, i, 0);
        fit_r=(float4)(fp[i],0,0,0);
        write_imagef(fp_dest_img,XYZi,fit_r);
    }
}