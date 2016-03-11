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

#ifndef PARA_NUM
#define PARA_NUM 1
#endif

#ifndef PARA_NUM_SQ
#define PARA_NUM_SQ 1
#endif

#ifndef ENERGY_NUM
#define ENERGY_NUM 1
#endif

#ifndef E0
#define E0 11559.0f
#endif


#ifndef FIT
#define FIT(x,mt,fp){\
\
float a_X = ((x)-((float*)(fp))[3])/((float*)(fp))[4];\
float l_X = ((x)-((float*)(fp))[6])/((float*)(fp))[7];\
\
(mt) = ((float*)(fp))[0] + ((float*)(fp))[1]*(x)\
+((float*)(fp))[2]*(0.5f+atanpi(a_X)\
+((float*)(fp))[5]/(1+l_X*l_X));\
\
}
#endif

#ifndef JACOBIAN
#define JACOBIAN(x,j,fp){\
\
float a_X = ((x)-((float*)(fp))[3])/((float*)(fp))[4];\
float l_X = ((x)-((float*)(fp))[6])/((float*)(fp))[7];\
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



static __constant sampler_t s_linear = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;



//simultaneous linear equation
static inline void sim_linear_eq(float *A, float *x, size_t dim){
    
    float a = 0.0f;
    
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
                            __global float *fit_results_img,__global float *fit_results_cnd_img,
                            __constant float *energy,__local float *energy_loc,
                            __global float *lambda,__global float *rho)
{
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_ID = global_x+global_y*IMAGESIZE_X;
    const size_t local_ID = get_local_id(0);
    
    //mt_data
    float mt_data[ENERGY_NUM];
    
    //LM parameters
    float tJJ[PARA_NUM_SQ];
    float fp[PARA_NUM];
    

    for(int en=0; en<ENERGY_NUM; en++){
        //copy mt data from grobal memory to private memory
        mt_data[en]=read_imagef(mt_img,s_linear,(float4)(global_x,global_y,en,0)).x;
        
        //copy energy data from global memory (energy) to local memory (energy_loc)
        if(local_ID==0) energy_loc[en]=energy[en];
    }
    
    float lambda_pr=lambda[global_ID];
    
    
    //LM minimization trial
    float tJdF[PARA_NUM];
    float lambda_diag_tJJ[PARA_NUM];
    float dp[PARA_NUM];
    for(int i=0; i<PARA_NUM; i++){
        //copy initial fitting parameter from constant memory to local memory
        fp[i] = fit_results_img[global_ID+i*IMAGESIZE_M];
        
        //tJJ, tJdF initialize
        tJdF[i]=0.0f;
        for(int j=i; j<PARA_NUM; j++){
            tJJ[i+j*PARA_NUM]=0.0f;
        }
    }
        
    float rho_pr = 0.0f;
    for(int en=0; en<ENERGY_NUM; en++){
        //mt_fit (trial)
        float J[PARA_NUM];
        float dF;
        FIT(energy_loc[en],dF,fp); //treat dF = mt_fit
        JACOBIAN(energy_loc[en],J,fp);
            
            
        //dF = mt_data - mt_fit
        dF = mt_data[en]-dF;
        //dF = read_imagef(mt_img,s_linear,(float4)(global_x,global_y,en,0)).x-dF;
        //chi2_old = dF x dF
        //rho = chi2_old
        rho_pr += dF*dF;
            
            
        //tJJ & tJdF
        for(int i=0; i<PARA_NUM; i++){
            tJJ[i+i*PARA_NUM] += J[i]*J[i];
            tJdF[i] += J[i]*dF;
            for(int j=i+1; j<PARA_NUM; j++){
                tJJ[i+j*PARA_NUM] += J[i]*J[j];
            }
        }
    }
    //copy tJdF to dp & tJJ transpose
    for(int i=0; i<PARA_NUM; i++){
        lambda_diag_tJJ[i] = tJJ[i+i*PARA_NUM]*lambda_pr;
        tJJ[i+i*PARA_NUM] *= (1+lambda_pr);
        dp[i] = tJdF[i];
            
        for(int j=i+1; j<PARA_NUM; j++){
            tJJ[j+i*PARA_NUM]  = tJJ[i+j*PARA_NUM];
        }
    }
        
        
    //solve dp of sim. linear eq. [ tJJ x dp = tJdF ]
    sim_linear_eq(tJJ,dp,PARA_NUM);
    float d_L = 0.0f;
    for(int i=0; i<PARA_NUM; i++){
        d_L   += dp[i]*(dp[i]*lambda_diag_tJJ[i] + tJdF[i]);
        fp[i] += dp[i];
        fit_results_cnd_img[global_ID+i*IMAGESIZE_M] = fp[i];
    }
        
        
    //evaluate new fp = (fp+dp) candidate
    for(int en=0; en<ENERGY_NUM; en++){
        //mt_fit (trial)
        float dF;
        FIT(energy_loc[en],dF,fp);
            
        //dF = mt_data - mt_fit
        dF = mt_data[en]-dF;
        //dF = read_imagef(mt_img,s_linear,(float4)(global_x,global_y,en,0)).x-dF;
        //chi2_new = dF x dF
        //rho = chi2_old - chi2_new
        rho_pr -= dF*dF;
    }
        
        
    //update lambda & dp(if rho>0)
    //rho = (chi2_old - chi2_new)/d_L
    rho[global_ID] = rho_pr/d_L;
    
}

__kernel void updateResults(__global float *fit_results_img,__global float *fit_results_cnd_img,
                            __global float *lambda,__global float *rho,__global float *nyu)
{
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_ID = global_x+global_y*IMAGESIZE_X;
    
    float rho_pr = rho[global_ID];
    float lambda_pr = lambda[global_ID];
    float nyu_pr = nyu[global_ID];
    
    float l_A = (2.0f*rho_pr-1.0f);
    l_A = 1.0f-l_A*l_A*l_A;
    l_A = max(0.333f,l_A)*lambda_pr;
    float l_B = nyu_pr*lambda_pr;
    
    if(rho_pr>=0.0f){
        lambda[global_ID]=l_A;
        nyu[global_ID]=2.0f;
        
        for(int i=0; i<PARA_NUM; i++){
            fit_results_img[global_ID+i*IMAGESIZE_M] = fit_results_cnd_img[global_ID+i*IMAGESIZE_M];
        }
    }else{
        lambda[global_ID]=l_B;
        nyu[global_ID]=2.0f*nyu_pr;
    }
    
    
}

__kernel void applyThreshold(__global float *fit_results_img,
                           __global char *mask,
                           __constant float *para_lower_limit,
                           __constant float *para_upper_limit)
{
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_ID = global_x+global_y*IMAGESIZE_X;
    
    float fp[PARA_NUM];
    //copy initial fitting parameter from constant memory to local memory
    for(int i=0; i<PARA_NUM;i++){
        fp[i] = fit_results_img[global_ID+i*IMAGESIZE_M];
    }
    
    //output fit result img
    for(int i=0; i<PARA_NUM; i++){
        if(isnan(fp[i])) fp[i]=0.0f;
        else if(fp[i]<para_lower_limit[i]) fp[i]=0.0f;
        else if(fp[i]>para_upper_limit[i]) fp[i]=para_upper_limit[i];
        
        fit_results_img[global_ID+i*IMAGESIZE_M] = fp[i]*mask[global_ID];
    }
}


__kernel void setMask(__read_only image2d_array_t mt_img,__global char *mask)
{
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_ID = global_x+global_y*IMAGESIZE_X;
    
    char mask_pr=1;
    float mt_data;
    //copy grobal memory to local memory, create mask
    for(int en=0; en<ENERGY_NUM; en++){
        mt_data=read_imagef(mt_img,s_linear,(float4)(global_x,global_y,en,0)).x;
    
        //mask data
        mask_pr *= (mt_data==0.0f) ? 0:1;
    }
    mask[global_ID]=mask_pr;
}
