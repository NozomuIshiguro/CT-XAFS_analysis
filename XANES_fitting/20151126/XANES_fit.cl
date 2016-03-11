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
#define FIT(x,mt,fp,dp){\
\
float a_X = ((x)-(((float*)(fp))[3]+((float*)(dp))[3]))/(((float*)(fp))[4]+((float*)(dp))[4]);\
float l_X = ((x)-((float*)(fp))[6])/((float*)(fp))[7];\
\
(mt) = (((float*)(fp))[0]+((float*)(dp))[0]) + (((float*)(fp))[1]+((float*)(dp))[1])*(x)\
+(((float*)(fp))[2]+((float*)(dp))[2])*(0.5f+atanpi(a_X)\
+(((float*)(fp))[5]+((float*)(dp))[5])/(1+l_X*l_X));\
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
                            __write_only image2d_array_t fit_results_img,
                            __constant float *energy,
                            __constant float *initial_fit_para,
                            __constant float *para_lower_limit,
                            __constant float *para_upper_limit)
{
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t local_ID = get_local_id(0);
    
    //mt_data
    float mt_data[ENERGY_NUM];
	__local float energy_loc[ENERGY_NUM];
    char mask=1;
    
    
    //LM parameters
    float tJJ[PARA_NUM_SQ];
    float fp[PARA_NUM];
    
    
    //dumping parameters
    float lambda=LAMBDA, nyu=2.0f;
    
    
    //copy initial fitting parameter from constant memory to local memory
    for(int i=0; i<PARA_NUM;i++){
        fp[i] = initial_fit_para[i];
    }
    
        
    //copy grobal memory to local memory, create mask
    for(int en=0; en<ENERGY_NUM; en++){
        mt_data[en]=read_imagef(mt_img,s_linear,(float4)(global_x,global_y,en,0)).x;
        
        //copy energy data from global memory (energy) to local memory (energy_loc)
		if(local_ID==0) energy_loc[en]=energy[en];
        
        //mask data
        mask *= (mt_data[en]==0.0f) ? 0:1;
    }
    
    
    //LM minimization trial
    //for(int trial=0;trial<NUM_TRIAL;trial++){
        //tJJ, tJdF reset
		float tJdF[PARA_NUM];
        float diag_tJJ[PARA_NUM];
        float dp[PARA_NUM];
        for(int i=0; i<PARA_NUM; i++){
			tJdF[i]=0.0f;
            dp[i]=0.0;
            diag_tJJ[i]=0.0f;
            for(int j=i; j<PARA_NUM; j++){
                tJJ[i+j*PARA_NUM]=0.0f;
            }
        }
        
        float rho = 0.0f;
        for(int en=0; en<ENERGY_NUM; en++){
            //mt_fit (trial)
			float J[PARA_NUM];
            float dF;
            FIT(energy_loc[en],dF,fp,dp); //treat dF = mt_fit
            JACOBIAN(energy_loc[en],J,fp);
            
            
            //dF = mt_data - mt_fit
            dF = mt_data[en]-dF;
            //chi2_old = dF x dF
            //rho = chi2_old
            rho += dF*dF;
            
            
            //tJJ & tJdF
            for(int i=0; i<PARA_NUM; i++){
                tJJ[i+i*PARA_NUM] += J[i]*J[i]*(1+lambda);  //L-M method
                diag_tJJ[i] += J[i]*J[i];
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
		float d_L = 0.0f;
        for(int i=0; i<PARA_NUM; i++){
            d_L += dp[i]*(dp[i]*lambda*diag_tJJ[i] + tJdF[i]);
        }
        
        
        //evaluate new fp = (fp+dp) candidate
        for(int en=0; en<ENERGY_NUM; en++){
            //mt_fit (trial)
            float dF;
            FIT(energy_loc[en],dF,fp,dp);
            
            //dF = mt_data - mt_fit
            dF = mt_data[en]-dF;
            //chi2_new = dF x dF
            //rho = chi2_old - chi2_new
            rho -= dF*dF;
        }
        
        
        //update lambda & dp(if rho>0)
        //rho = (chi2_old - chi2_new)/d_L
		rho /= d_L;
		float l_A = (2.0f*rho-1.0f);
        l_A = 1.0f-l_A*l_A*l_A;
        l_A = max(0.333f,l_A)*lambda;
        float l_B = nyu*lambda;
        char update_flg = (rho>=0.0f) ? 1:0;
        lambda = (rho>=0.0f) ? l_A:l_B;
        nyu = (rho>=0.0f) ? 2.0f:2.0f*nyu;
		for(int i=0; i<PARA_NUM; i++){
            fp[i] += dp[i]*update_flg;
        }
    //}
    
    
    //output fit result img
    for(int i=0; i<PARA_NUM; i++){
        if(isnan(fp[i])) fp[i]=0.0f;
        else if(fp[i]<para_lower_limit[i]) fp[i]=0.0f;
        else if(fp[i]>para_upper_limit[i]) fp[i]=para_upper_limit[i];
        
        write_imagef(fit_results_img,
					(int4)(global_x, global_y, i, 0),
					(float4)(fp[i]*mask,0.0f,0.0f,0.0f));
    }
}
