//
//  XANES_fit.cl
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2017/04/28.
//  Copyright (c) 2017 Nozomu Ishiguro. All rights reserved.
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

#ifndef IMAGESIZE_E
#define IMAGESIZE_E 1024
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

#ifndef EZERO
#define EZERO 11559.0f
#endif

#ifndef CONTRAIN_NUM
#define CONTRAIN_NUM 0
#endif

#ifndef START_E
#define START_E -100.0f
#endif

#ifndef END_E
#define END_E 100.0f
#endif

#ifndef FUNC_NUM
#define FUNC_NUM 1
#endif

#ifndef EPSILON
#define EPSILON 1.0f
#endif



static __constant sampler_t s_linearXANES = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;


inline float line(float energy, float* p){
    
    return p[0]+p[1]*energy;
}


inline void Jacobian_line(float* J, float energy, float*p){
    
    J[0] = 1.0f;
    J[1] = energy;
}

inline float Victoreen(float energy, float* p){
    
    float e1 = 1.0e3f/(energy+p[3]);
    float e3 = e1*e1*e1;
    float e4 = e3*e1;
    
    return p[0] + p[1]*e3 - p[2]*e4;
}

inline void Jacobian_Victoreen(float* J, float energy, float*p){
    
    float e1 = 1.0e3f/(energy+p[3]);
    float e3 = e1*e1*e1;
    float e4 = e3*e1;
    float e5 = e4*e1;
    
    J[0] = 1.0f;
    J[1] = e3;
    J[2] = -e4;
    J[3] = -3.0e-3f*p[1]*e4 + 4.0e-3f*p[2]*e5;
}

inline float thirdPolynominal(float energy, float* p){
    
    return p[0] + p[1]*energy + p[2]*energy*energy  + p[3]*energy*energy*energy;
}

inline void Jacobian_thirdPolynominal(float* J, float energy, float*p){
    J[0] = 1.0f;
    J[1] = energy;
    J[2] = energy*energy;
    J[3] = energy*energy*energy;
}

inline float McMaster(float energy, float* p){
    
    float e1 = (energy+p[2])*1.0e-3f;
    e1 = log(e1);
    float pow_e = exp(-2.75f*e1);
    
    return p[0]+p[1]*pow_e;
}

inline void Jacobian_McMaster(float* J, float energy, float*p){
    
    float e1 = (energy+p[2])*1.0e-3f;
    e1 = log(e1);
    float pow_e = exp(-2.75f*e1);
    
    J[0] = 1.0f;
    J[1] = pow_e;
    J[2] = p[1]*pow_e/e1*1.0e-3f;
}

inline float Gaussian(float energy, float* p){
    
    float val = (energy-p[1])/p[2];
    return p[0]*exp(-val*val);
}

inline void Jacobian_Gaussian(float*J, float energy, float*p){
    
    float val = (energy-p[1])/p[2];
    
    J[0] = exp(-val*val);
    J[1] = 2.0f*p[0]*val/p[2]*exp(-val*val);
    J[2] = 2.0f*p[0]*val*val/p[2]*exp(-val*val);
}


inline float Lorentzian(float energy, float* p){
    
    float val = (energy-p[1])/p[2];
    
    return p[0]/(1.0f+val*val);
}


inline void Jacobian_Lorentzian(float* J, float energy, float*p){
    
    float val = (energy-p[1])/p[2];
    
    J[0] = 1.0f/(1.0f+val*val);
    J[1] = 2.0f*p[0]*val/p[2]/(1.0f+val*val)/(1.0f+val*val);
    J[2] = 2.0f*p[0]*val*val/p[2]/(1.0f+val*val)/(1.0f+val*val);
}


inline float arctan_edge(float energy, float* p){
    
    float val = (energy-p[1])/p[2];
    
    return p[0]*(0.5f+atanpi(val));
}


inline void Jacobian_arctan_edge(float* J, float energy, float*p){
    
    float val = (energy-p[1])/p[2];
    
    J[0] = 0.5f+atanpi(val);
    J[1] = -p[0]/(1.0f+val*val)/p[2]/PI;
    J[2] = -p[0]/(1.0f+val*val)*val/p[2]/PI;
}

inline float erf_edge(float energy, float* p){
    
    float val = (energy-p[1])/p[2];
    
    return p[0]*(1.0f+erf(val))*0.5f;
}

inline void Jacobian_erf_edge(float* J, float energy, float*p){
    
    float val = (energy-p[1])/p[2];
    
    J[0] = (1.0f+erf(val))*0.5f;
    J[1] = -p[0]*exp(-val*val)/p[2]*0.5f;
    J[2] = -p[0]*exp(-val*val)*val/p[2]*0.5f;
}

inline float LCF(float energy, float* p, __read_only image1d_array_t refSpectra,
                 int offset){
    
    float E_pitch = (float)(END_E-START_E)/((float)IMAGESIZE_E);
    float2 XY = (float2)((energy+p[1]-START_E)/E_pitch+0.5f,offset);
    
    float val = read_imagef(refSpectra,s_linearXANES,XY).x;
    return p[0]*val;
}

inline void Jacobian_LCF(float* J, float energy, float* p, __read_only image1d_array_t refSpectra,
                          int offset){
    
    float E_pitch = (float)(END_E-START_E)/((float)IMAGESIZE_E);
    float X = (energy+p[1]-START_E)/E_pitch;
    
    float val = read_imagef(refSpectra,s_linearXANES,(float2)(X,offset)).x;
    float valp = read_imagef(refSpectra,s_linearXANES,(float2)(X+1.0f,offset)).x;
    float valm = read_imagef(refSpectra,s_linearXANES,(float2)(X-1.0f,offset)).x;
    
    J[0] = val;
    J[1] = p[0]*(valp-valm)/E_pitch/2.0f;
}


inline float mtFit(float* fp, float energy, __constant int* funcModeList,
                   __read_only image1d_array_t refSpectra){
    
    float mt=0.0f;
    int fpOffset=0;
    int LCFoffset=0;
    for(int i=0;i<FUNC_NUM;i++){
        switch(funcModeList[i]){
            case 0: //line
                mt += line(energy,fp+fpOffset);
                fpOffset +=2;
                break;
                
            case 1: //Gaussian
                mt += Gaussian(energy,fp+fpOffset);
                fpOffset +=3;
                break;
                
            case 2: //Lorentzian
                mt += Lorentzian(energy,fp+fpOffset);
                fpOffset +=3;
                break;
                
            case 3: //arctan edge
                mt += arctan_edge(energy,fp+fpOffset);
                fpOffset +=3;
                break;
                
            case 4: //erf edge
                mt += erf_edge(energy,fp+fpOffset);
                fpOffset +=3;
                break;
                
            case 5: //LCF
                mt += LCF(energy,fp+fpOffset,refSpectra,LCFoffset);
                fpOffset +=2;
                LCFoffset++;
                break;
                
            case 6: //Victoreen
                mt += Victoreen(energy,fp+fpOffset);
                fpOffset +=4;
                break;
                
            case 7: //McMaster
                mt += McMaster(energy,fp+fpOffset);
                fpOffset +=3;
                break;
                
            case 8: //3rd polynominal
                mt += thirdPolynominal(energy,fp+fpOffset);
                fpOffset +=4;
                break;
                
        }
    }
    return mt;
}


inline void Jacobian(float* J, float* fp, float energy,
                     __constant int* funcModeList, __read_only image1d_array_t refSpectra){
    
    int fpOffset=0;
    int LCFoffset=0;
    for(int i=0;i<FUNC_NUM;i++){
        switch(funcModeList[i]){
            case 0: //line
                Jacobian_line(J+fpOffset, energy,fp+fpOffset);
                fpOffset +=2;
                break;
                
            case 1: //Gaussian
                Jacobian_Gaussian(J+fpOffset, energy,fp+fpOffset);
                fpOffset +=3;
                break;
                
            case 2: //Lorentzian
                Jacobian_Lorentzian(J+fpOffset, energy,fp+fpOffset);
                fpOffset +=3;
                break;
                
            case 3: //arctan edge
                Jacobian_arctan_edge(J+fpOffset, energy,fp+fpOffset);
                fpOffset +=3;
                break;
                
            case 4: //erf edge
                Jacobian_erf_edge(J+fpOffset, energy,fp+fpOffset);
                fpOffset +=3;
                break;
                
            case 5: //LCF
                Jacobian_LCF(J+fpOffset,energy,fp+fpOffset,refSpectra,LCFoffset);
                fpOffset +=2;
                LCFoffset++;
                break;
                
            case 6: //Victoreen
                Jacobian_Victoreen(J+fpOffset, energy,fp+fpOffset);
                fpOffset +=4;
                break;
                
            case 7: //McMaster
                Jacobian_McMaster(J+fpOffset, energy,fp+fpOffset);
                fpOffset +=3;
                break;
            case 8: //3rd polynominal
                Jacobian_thirdPolynominal(J+fpOffset, energy,fp+fpOffset);
                fpOffset +=4;
                break;
        }
    }
}



__kernel void chi2Stack(__global float* mt_img, __global float* fp_img,
                        __read_only image1d_array_t refSpectra, __global float* chi2_img,
                        __constant float* energy, __constant int* funcModeList,
                        int startEnum, int endEnum,
                        __global float* weight_img, __global float* weight_thd_img, float CI){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_ID = global_x+global_y*IMAGESIZE_X;
    
    float fp[PARA_NUM];
    for(int i=0; i<PARA_NUM; i++){
        fp[i] = fp_img[global_ID+i*IMAGESIZE_M];
    }
    
    float chi2=0.0f;
    float chi=0.0f;
    float mt_data,mt_fit,weight;
    
    for(int en=startEnum; en<=endEnum; en++){
        mt_data = mt_img[global_ID+en*IMAGESIZE_M];
        mt_fit = mtFit(fp,energy[en],funcModeList,refSpectra);
        weight = weight_img[global_ID+en*IMAGESIZE_M];
        
        chi  += (mt_data-mt_fit)*weight;
        chi2 += (mt_data-mt_fit)*(mt_data-mt_fit)*weight;
    }
    
    float weight_thd = fabs(chi)+sqrt(max(0.0f,chi2 - chi*chi))*CI;
    weight_thd_img[global_ID] = weight_thd;
    
    chi2_img[global_ID] = chi2;
}


__kernel void chi2_tJdF_tJJ_Stack(__global float* mt_img, __global float* fp_img,
                                  __read_only image1d_array_t refSpectra,
                                  __global float* chi2_img, __global float* tJdF_img,
                                  __global float* tJJ_img,
                                  __constant float* energy, __constant int* funcModeList,
                                  __constant char *p_fix, int startEnum, int endEnum,
                                  __global float* weight_img, __global float* weight_thd_img){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_ID = global_x+global_y*IMAGESIZE_X;
    
    
    //initialization
    float chi2=0.0f;
    float weight_thd = weight_thd_img[global_ID];
    float val,weight;
    float J[PARA_NUM],tJdF[PARA_NUM],tJJ[PARA_NUM*(PARA_NUM+1)/2];
    float fp[PARA_NUM];
    for(int i=0; i<PARA_NUM; i++){
        fp[i] = fp_img[global_ID+i*IMAGESIZE_M];
        tJdF[i] = 0.0f;
        tJJ[PARA_NUM*i-(i-1)*i/2] = 0.0f;
        for(int j=i+1; j<PARA_NUM; j++){
            tJJ[PARA_NUM*i-(i+1)*i/2+j] = 0.0f;
        }
    }
    
    //chi2, tJdF, tJJ calculation
    float mt_data,mt_fit;
    for(int en=startEnum; en<=endEnum; en++){
        mt_data = mt_img[global_ID+en*IMAGESIZE_M];
        mt_fit = mtFit(fp,energy[en],funcModeList,refSpectra);
        val = (mt_data-mt_fit);
        weight = val/weight_thd;
        weight = (fabs(weight)<=1.0f) ? (1.0f - weight*weight):0.0f;
        weight = 1.0f;
        
        Jacobian(J,fp,energy[en],funcModeList,refSpectra);
        
        chi2 += val*val*weight;
        for(int i=0; i<PARA_NUM; i++){
            if(p_fix[i] == 48) continue;
            
            tJdF[i] += J[i]*val*weight;
            for(int j=i; j<PARA_NUM; j++){
                if(p_fix[j] == 48) continue;
                tJJ[PARA_NUM*i-(i+1)*i/2+j] += J[i]*J[j]*weight;
            }
        }
        weight_img[global_ID+en*IMAGESIZE_M]=weight;
    }
    
    
    //write to global memory
    chi2_img[global_ID] = chi2;
    
    for(int i=0; i<PARA_NUM; i++){
        tJdF_img[global_ID+i*IMAGESIZE_M] = tJdF[i];
        tJJ_img[global_ID+(PARA_NUM*i-(i-1)*i/2)*IMAGESIZE_M] = tJJ[PARA_NUM*i-(i-1)*i/2];
        for(int j=i; j<PARA_NUM; j++){
            tJJ_img[global_ID+(PARA_NUM*i-(i+1)*i/2+j)*IMAGESIZE_M] = tJJ[PARA_NUM*i-(i+1)*i/2+j];
        }
    }
}


__kernel void setMask(__global float* mt_img,__global char *mask_img)
{
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_ID = global_x+global_y*IMAGESIZE_X;
    
    char mask=1;
    float mt_data;
    //copy grobal memory to local memory, create mask
    for(int en=0; en<ENERGY_NUM; en++){
        mt_data=mt_img[global_ID+en*IMAGESIZE_M];
        
        //mask data
        mask *= (mt_data==0.0f) ? 0:1;
    }
    mask_img[global_ID]=mask;
}


__kernel void applyMask(__global float *fp_img, __global char *mask)
{
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t imageSizeX = get_global_size(0);
    const size_t imageSizeY = get_global_size(1);
    const size_t IDxy = global_x+global_y*imageSizeX;
    const size_t IDxyz = IDxy + global_z*imageSizeX*imageSizeY;
    
    float fp = fp_img[IDxyz];
    fp = isnan(fp) ? 0.0f:fp;
    fp_img[IDxyz] = fp*(float)mask[IDxy];
}



__kernel void redimension_refSpecta(__write_only image1d_array_t refSpectra,
                                    __read_only image1d_t refSpectrum_raw,
                                    __constant float* energy, int numE, int offset){
    
    float e1, e2, e3;
    float E_pitch = (float)(END_E-START_E)/((float)IMAGESIZE_E);
    int n = 0;
    float X;
    float4 img;
    
    for(int en=0; en<numE; en++){
        e1 = energy[en] - EZERO;
        e2 = energy[en+1] - EZERO;
        if(e2>START_E && e1<END_E){
            e3 = START_E+n*E_pitch;
            while(e3<e2){
                X=en+(e3-e1)/(e2-e1)+0.5f;
                img = read_imagef(refSpectrum_raw,s_linearXANES,X);
                write_imagef(refSpectra,(int2)(n,offset),img);
                n++;
                e3 = START_E+n*E_pitch;
                if(n==IMAGESIZE_E) break;
            }
        }
        if(n==IMAGESIZE_E) break;
    }
}

