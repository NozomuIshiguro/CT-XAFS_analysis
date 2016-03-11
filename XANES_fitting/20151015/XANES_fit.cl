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

#ifndef PARA_NUM_SQ
#define PARA_NUM_SQ 1
#endif

#ifndef ENERGY_NUM
#define ENERGY_NUM 1
#endif

#ifndef E0
#define E0 11559.0f
#endif

#ifndef LAMBDA
#define LAMBDA 0.001f
#endif



#ifndef FIT
#define FIT(x,mt,fp,msk){\
\
float a_X = ((x)-((float*)(fp))[3])/((float*)(fp))[4];\
float l_X = ((x)-((float*)(fp))[6])/((float*)(fp))[7];\
\
(mt) = ((float*)(fp))[0] + ((float*)(fp))[1]*(x)\
+((float*)(fp))[2]*(0.5f+atanpi(a_X)\
+((float*)(fp))[5]/(1+l_X*l_X));\
(mt)*=(msk);\
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



__constant sampler_t s_linear = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;



//simultaneous linear equation
inline void sim_linear_eq(float *A, float *x, size_t dim){
    
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
                            __write_only image2d_array_t fit_results_img,
                            __global float *energy,
                            __constant float *initial_fit_para,
                            __constant float *para_lower_limit,
                            __constant float *para_upper_limit)
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
    float tJJ[PARA_NUM_SQ];
    float tJdF[PARA_NUM];
    float fp[PARA_NUM], dp[PARA_NUM], fp_cnd[PARA_NUM];
    float energy_pr[ENERGY_NUM];
    
    
    //dumping parameters
    float chi2_new=0.0f, chi2_old=0.0f;
    float lambda=LAMBDA;
    float rho, d_L, nyu;
    float l_A, l_B, l_C;
    float n_B, n_C;
    
    
    //copy initial fitting parameter from constant memory to local memory
    for(int i=0; i<PARA_NUM;i++){
        fp[i] = initial_fit_para[i];
    }
    barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
    
        
    //copy grobal memory to private memory
    for(int en=0; en<ENERGY_NUM; en++){
        //copy global mt_img to private mt_data
        XYZf=(float4)(global_x, global_y, en, 0);
        mt_data[en]=read_imagef(mt_img,s_linear,XYZf).x;
        
        //copy energy data from global memory (energy) to private memory (energy_pr)
        energy_pr[en]=energy[en];
        barrier(CLK_GLOBAL_MEM_FENCE);
        X=energy_pr[en];
        
        //mask data
        if (mt_data[en]==0.0f) mask*=0;
        else mask *=1;
    }
    
    
    //LM minimization trial
    for(int trial=0;trial<NUM_TRIAL;trial++){
        //chi2, tJJ, tJdF reset
        chi2_old = 0.0f;
        chi2_new = 0.0f;
        for(int i=0; i<PARA_NUM; i++){
            tJJ[i+i*PARA_NUM]=0.0f;
            tJdF[i]=0.0f;
            
            for(int j=i+1; j<PARA_NUM; j++){
                tJJ[i+j*PARA_NUM]=0.0f;
                tJJ[j+i*PARA_NUM]=0.0f;
            }
        }
        
        
        for(int en=0; en<ENERGY_NUM; en++){
            //mt_fit (trial)
            X=energy_pr[en];
            FIT(X,mt_fit,fp,mask);
            JACOBIAN(X,J,fp);
            
            
            //dF = mt_data*mask - mt_fit
            //chi2_old = dF x dF
            dF = (mt_data[en]*mask-mt_fit);
            chi2_old += dF*dF;
            
            
            //tJJ & tJdF
            for(int i=0; i<PARA_NUM; i++){
                tJJ[i+i*PARA_NUM] += J[i]*J[i]*(1+lambda);  //L-M method
                tJdF[i] += J[i]*dF;
                for(int j=i+1; j<PARA_NUM; j++){
                    tJJ[i+j*PARA_NUM] += J[i]*J[j];
                }
            }
        }
        //copy tJdF to dp & tJJ transpose
        for(int i=0; i<PARA_NUM; i++){
            dp[i] = tJdF[i];
            
            for(int j=i+1; j<PARA_NUM; j++){
                tJJ[j+i*PARA_NUM]  = tJJ[i+j*PARA_NUM];
            }
        }
        
        
        //solve dp of sim. linear eq. [ tJJ x dp = tJdF ]
        sim_linear_eq(tJJ,dp,PARA_NUM);
        
        
        //estimate fp_cnd and chi2_new as candidate
        for(int i=0; i<PARA_NUM; i++){
            fp_cnd[i] = fp[i] + dp[i];
        }
        for(int en=0; en<ENERGY_NUM; en++){
            //mt_fit (trial)
            X=energy_pr[en];
            FIT(X,mt_fit,fp_cnd,mask);
            
            //dF = mt_data*mask - mt_fit
            //chi2_new = dF x dF
            dF = (mt_data[en]*mask-mt_fit);
            chi2_new += dF*dF;
        }
        
        
        //update lambda & dp(if rho>0)
        d_L = 0.0f;
        for(int i=0; i<PARA_NUM; i++){
            d_L += dp[i]*(dp[i]*lambda*tJJ[i+i*PARA_NUM] + tJdF[i]);
        }
		rho = (chi2_old-chi2_new)/d_L;
        l_A = 1.0f-(2.0f*rho-1.0f)*(2.0f*rho-1.0f)*(2.0f*rho-1.0f);
        l_B = lambda*max(0.333f,l_A);
        l_C = lambda*nyu;
        n_B = 2.0f;
        n_C = nyu*2.0f;
        if(rho>=0.0f){  //step accetable
            lambda  = l_B;
            nyu     = n_B;
            chi2_old = chi2_new;
            for(int i=0; i<PARA_NUM; i++){
                fp[i] = fp_cnd[i];
            }
        }else{
            lambda = l_C;
            nyu    = n_B;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    
    //output fit result img
    for(int i=0; i<PARA_NUM; i++){
        if(isnan(fp[i])) fp[i]=0.0f;
        else if(fp[i]<para_lower_limit[i]) fp[i]=0.0f;
        else if(fp[i]>para_upper_limit[i]) fp[i]=para_upper_limit[i];
        
        XYZi=(int4)(global_x, global_y, i, 0);
        fit_r=(float4)(fp[i]*mask,0.0f,0.0f,0.0f);
        write_imagef(fit_results_img,XYZi,fit_r);
    }
}
