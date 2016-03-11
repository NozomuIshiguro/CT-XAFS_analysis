//
//  XANES_fit.cl
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/01/07.
//  Copyright (c) 2014-2015 Nozomu Ishiguro. All rights reserved.
//
//pragma enable

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

__kernel void XANES_fitting(MT_DEF, FIT_RESULTS_DEF,
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
    
    //image pointers
    /*if(local_ID==0){
     MT_P;
     FIT_RESULTS_P;
     }
     barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);*/
    
    //mt_data, mit_fit
    float mt_data[ENERGY_NUM];
    float mt_fit, delta_mt_fit;
    char mask=1;
    
    //LM parameters
    float X, atan_X, lor_X;
    float Jacob[PARA_NUM];
    float tJJ[PARA_NUM][PARA_NUM];
    float tJ_delta_mt[PARA_NUM];
    float fit_para[PARA_NUM], delta_fit_para[PARA_NUM];
    float energy_pr[ENERGY_NUM];
    
    
    //dumping parameters
    float Fx_new, Fx_old=0.0;
    float lambda, rho, delta_rho, nyu;
    float lambdaA, lambdaB, lambdaC;
    float nyuA, nyuB;
    
    
    //parameter for inversed matrix
    float a;
    
    //copy initial fitting parameter from constant memory to local memory
    for(int i=0; i<PARA_NUM;i++){
        fit_para[i] = initial_fit_para[i];
    }
    barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
    
    //L-M damping parameter
    lambda=0.2f;
    nyu=2.0f;
    rho=0.0f;
    delta_rho=1.0f;
    
    //transfer global mt_data to private memory
    MT_DATA_READ(mt_data,global_ID);
    
    //initialize Fx_old & copy energy from global to private memory
    for(int i=0; i<ENERGY_NUM; i++){
        energy_pr[i]=energy[i];
        barrier(CLK_GLOBAL_MEM_FENCE);
        X=energy_pr[i];
        
        //mt_data = mt_p[i][global_ID];
        if (mt_data[i]==0.0) mask*=0;
        else mask *=1;
        
        INITIAL_FX_OLD(X,Fx_old,fit_para,mt_data[i]);
    }
    
    
    for(int trial=0;trial<20;trial++){
        //tJJ, delta_fit, fx_new reset
        for(int i=0; i<PARA_NUM; i++){
            tJJ[i][i]=lambda;
            tJ_delta_mt[i]=0.0;
            Fx_new = 0.0;
            
            for(int j=i+1; j<PARA_NUM; j++){
                tJJ[i][j]=0.0;
                tJJ[j][i]=0.0;
            }
            
        }
        
        
        for(int i=0; i<ENERGY_NUM; i++){
            //copy mt_data
            //mt_data = mt_p[i][global_ID];
            if (mt_data[i]==0.0) mask*=0;
            else mask *=1;
            
            //mt_fit (trial)
            X=energy_pr[i];
            FIT_AND_JACOBIAN(X,mt_fit,Jacob,fit_para,mask);
            
            //delta mt fitting
            delta_mt_fit = (mt_data[i]-mt_fit);
            
            
            //tJJ & tJ_delta_mt & Fx_new
            for(int i=0; i<PARA_NUM; i++){
                tJJ[i][i] += Jacob[i]*Jacob[i];
                tJ_delta_mt[i] += Jacob[i]*delta_mt_fit;
                Fx_new += delta_mt_fit*delta_mt_fit;
                delta_fit_para[i] = tJ_delta_mt[i];
                
                for(int j=i+1; j<PARA_NUM; j++){
                    tJJ[i][j] += Jacob[i]*Jacob[j];
                    tJJ[j][i] = tJJ[i][j];
                }
            }
        }
        
        
        //inversed matrix
        for(int i=0; i<PARA_NUM; i++){
            //devide (i,i) to 1
            a = 1/tJJ[i][i];
            
            for(int j=0; j<PARA_NUM; j++){
                tJJ[i][j] *= a;
            }
            delta_fit_para[i] *= a;
            
            //erase (j,i) (i!=j) to 0
            for(int j=i+1; j<PARA_NUM; j++){
                a = tJJ[j][i];
                for(int k=0; k<PARA_NUM; k++){
                    tJJ[j][k] -= a*tJJ[i][k];
                }
                delta_fit_para[j] -= a*delta_fit_para[i];
            }
        }
        for(int i=0; i<PARA_NUM; i++){
            for(int j=i+1; j<PARA_NUM; j++){
                a = tJJ[i][j];
                for(int k=0; k<PARA_NUM; k++){
                    tJJ[k][j] -= a*tJJ[k][i];
                }
                delta_fit_para[i] -= a*delta_fit_para[j];
            }
        }
        
        //update lambda
        delta_rho = 0;
        for(int i=0; i<PARA_NUM; i++){
            delta_rho += delta_fit_para[i]*(delta_fit_para[i]*lambda + tJ_delta_mt[i]);
        }
        rho = (Fx_old-Fx_new)/delta_rho;
        lambdaA = 1.0f-(2.0f*rho-1.0f)*(2.0f*rho-1.0f)*(2.0f*rho-1.0f);
        lambdaB = lambda*max(0.333f,lambdaA);
        lambdaC = lambda*nyu;
        nyuA=2.0f;
        nyuB=nyu*2.0f;
        if(rho>0.0f){
            lambda = lambdaB;
            nyu=nyuA;
        }else{
            lambda = lambdaC;
            nyu=nyuB;
        }
        Fx_old = Fx_new;
        
        
        for(int i=0; i<PARA_NUM; i++){
            fit_para[i] += delta_fit_para[i];
        }
    }
    
    for(int i=0; i<PARA_NUM; i++){
        if(isnan(fit_para[i])) fit_para[i]=0;
        else if(fit_para[i]<para_lower_limit[i]) fit_para[i]=0;
        else if(fit_para[i]>para_upper_limit[i]) fit_para[i]=0;
    }
    
    /*for(int i=0; i<PARA_NUM; i++){
     fit_results_p[i][global_ID]=fit_para[i];
     }*/
    FIT_RESULTS_WRITE(fit_para,global_ID);
}
