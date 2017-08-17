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

#ifndef CONSTRAIN_NUM
#define CONSTRAIN_NUM 0
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
                        int startEnum, int endEnum){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_ID = global_x+global_y*IMAGESIZE_X;
    
    float fp[PARA_NUM];
    for(int i=0; i<PARA_NUM; i++){
        fp[i] = fp_img[global_ID+i*IMAGESIZE_M];
    }
    
    float chi2=0.0f;
    float mt_data,mt_fit;
    for(int en=startEnum; en<=endEnum; en++){
        mt_data = mt_img[global_ID+en*IMAGESIZE_M];
        mt_fit = mtFit(fp,energy[en],funcModeList,refSpectra);
        chi2 += (mt_data-mt_fit)*(mt_data-mt_fit);
    }
    
    chi2_img[global_ID] = chi2;
}


__kernel void chi2_tJdF_tJJ_Stack(__global float* mt_img, __global float* fp_img,
                                  __read_only image1d_array_t refSpectra,
                                  __global float* chi2_img, __global float* tJdF_img,
                                  __global float* tJJ_img,
                                  __constant float* energy, __constant int* funcModeList,
                                  __constant char *p_fix, int startEnum, int endEnum){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_ID = global_x+global_y*IMAGESIZE_X;
    
    
    //initialization
    float chi2=0.0f;
    float val;
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
        
        Jacobian(J,fp,energy[en],funcModeList,refSpectra);
        
        chi2 += val*val;
        for(int i=0; i<PARA_NUM; i++){
            if(p_fix[i] == 48) continue;
            
            tJdF[i] += J[i]*val;
            tJJ[PARA_NUM*i-(i-1)*i/2] += J[i]*J[i];
            for(int j=i+1; j<PARA_NUM; j++){
                if(p_fix[j] == 48) continue;
                tJJ[PARA_NUM*i-(i+1)*i/2+j] += J[i]*J[j];
            }
        }
    }
    
    
    //write to global memory
    chi2_img[global_ID] = chi2;
    for(int i=0; i<PARA_NUM; i++){
        tJdF_img[global_ID+i*IMAGESIZE_M] = tJdF[i];
        tJJ_img[global_ID+(PARA_NUM*i-(i-1)*i/2)*IMAGESIZE_M] = tJJ[PARA_NUM*i-(i-1)*i/2];
        for(int j=i+1; j<PARA_NUM; j++){
            tJJ_img[global_ID+(PARA_NUM*i-(i+1)*i/2+j)*IMAGESIZE_M] = tJJ[PARA_NUM*i-(i+1)*i/2+j];
        }
    }
}



__kernel void constrain(__global float* fp_cnd_img,
                        __constant float *C_mat, __constant float *D_vec){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_ID = global_x+global_y*IMAGESIZE_X;
    
    float fp[PARA_NUM];
    for(int i=0; i<PARA_NUM; i++){
        fp[i] = fp_cnd_img[global_ID+i*IMAGESIZE_M];
    }
    for(int j=0;j<CONSTRAIN_NUM;j++){
        float eval=0.0f;
        float C2 =0.0f;
        for(int i=0; i<PARA_NUM; i++){
            eval += C_mat[j*PARA_NUM+i]*fp[i];
            C2 += C_mat[j*PARA_NUM+i]*C_mat[j*PARA_NUM+i];
        }
        bool eval_b = (eval>D_vec[j]);
        float h = (eval-D_vec[j])/sqrt(C2);
        for(int i=0; i<PARA_NUM; i++){
            fp[i] = (eval_b) ? fp[i]-h*C_mat[j*PARA_NUM+i]:fp[i];
        }
    }
    for(int i=0; i<PARA_NUM; i++){
        fp_cnd_img[global_ID+i*IMAGESIZE_M] = fp[i];
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


__kernel void partialDerivativeOfGradiant(__global float* original_img_src,
                                          __global float* img_src,
                                          __global float* img_dest,
                                          __constant float* epsilon_g, __constant float*alpha_g){
    
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    size_t global_ID;
    
    float epsilon =epsilon_g[Z];
    float alpha =alpha_g[Z];
    
    global_ID = X + Y*IMAGESIZE_X + Z*IMAGESIZE_M;
    float f_o = original_img_src[global_ID];
    float f_i_j = img_src[global_ID];
    global_ID = (X+1) + Y*IMAGESIZE_X + Z*IMAGESIZE_M;
    float f_ip_j = (X+1<IMAGESIZE_X) ? img_src[global_ID]:0.0f;
    global_ID = (X-1) + Y*IMAGESIZE_X + Z*IMAGESIZE_M;
    float f_im_j = (X-1>=0) ? img_src[global_ID]:0.0f;
    global_ID = (X-1) + (Y+1)*IMAGESIZE_X + Z*IMAGESIZE_M;
    float f_im_jp = (X-1>=0&&Y+1<IMAGESIZE_Y) ? img_src[global_ID]:0.0f;
    global_ID = X + (Y+1)*IMAGESIZE_X + Z*IMAGESIZE_M;
    float f_i_jp = (Y+1<IMAGESIZE_Y) ? img_src[global_ID]:0.0f;
    global_ID = X + (Y-1)*IMAGESIZE_X + Z*IMAGESIZE_M;
    float f_i_jm = (Y-1>=0) ? img_src[global_ID]:0.0f;
    global_ID = (X+1) + (Y-1)*IMAGESIZE_X + Z*IMAGESIZE_M;
    float f_ip_jm = (X+1<IMAGESIZE_X&&Y-1>=0) ? img_src[global_ID]:0.0f;
    
    float v=0.0f;
    float grad = epsilon + (f_ip_j-f_i_j)*(f_ip_j-f_i_j) + (f_i_jp-f_i_j)*(f_i_jp-f_i_j);
    v += (2.0f*f_i_j - f_ip_j - f_i_jp)/sqrt(grad);
    grad = epsilon + (f_i_j-f_im_j)*(f_i_j-f_im_j) + (f_im_jp-f_im_j)*(f_im_jp-f_im_j);
    v += (f_i_j - f_im_j)/sqrt(grad);
    grad = epsilon + (f_ip_jm-f_i_jm)*(f_ip_jm-f_i_jm) + (f_i_j-f_i_jm)*(f_i_j-f_i_jm);
    v += (f_i_j - f_i_jm)/sqrt(grad);
    
    f_i_j -= (v>=0.0f) ? alpha*fabs(f_o):-alpha*fabs(f_o);
    //f_i_j -= alpha*v;
    
    global_ID = X + Y*IMAGESIZE_X + Z*IMAGESIZE_M;
    img_dest[global_ID]= f_i_j;
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


__kernel void SoftThresholdingFunc(__global float* fp_img,
                                   __global float* dp_img,__global float* dp_cnd_img,
                                   __global float* inv_tJJ_img,
                                   __constant char *p_fix,__constant float* lambda_fista){
    
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int z = get_global_id(2);
    const size_t global_xy = x+y*IMAGESIZE_X;
    const size_t global_ID = x+y*IMAGESIZE_X+z*IMAGESIZE_M;
    
    float lambda1 = inv_tJJ_img[global_xy+(PARA_NUM*z-(z-1)*z/2)*IMAGESIZE_M]*lambda_fista[z];//
    float fpdp_neighbor[4];
    int px = x+1;
    int py = y+1;
    int mx = x-1;
    int my = y-1;
    
    float fp = fp_img[global_ID];
    float dp = dp_img[global_ID];
    fpdp_neighbor[0]=(px>=IMAGESIZE_X)	? fp+dp:fp_img[px+y*IMAGESIZE_X+z*IMAGESIZE_M]+dp_img[px+y*IMAGESIZE_X+z*IMAGESIZE_M];
    fpdp_neighbor[1]=(py>=IMAGESIZE_Y)	? fp+dp:fp_img[x+py*IMAGESIZE_X+z*IMAGESIZE_M]+dp_img[x+py*IMAGESIZE_X+z*IMAGESIZE_M];
    fpdp_neighbor[2]=(mx < 0)			? fp+dp:fp_img[mx+y*IMAGESIZE_X+z*IMAGESIZE_M]+dp_img[mx+y*IMAGESIZE_X+z*IMAGESIZE_M];
    fpdp_neighbor[3]=(my < 0)			? fp+dp:fp_img[x+my*IMAGESIZE_X+z*IMAGESIZE_M]+dp_img[x+my*IMAGESIZE_X+z*IMAGESIZE_M];
    
    //change order of img_neibor[4]
    float fp1, fp2;
    int biggerN;
    for(int i=0;i<4;i++){
        fp1 = fpdp_neighbor[i];
        fp2 =fp1;
        biggerN = i;
        for(int j=i+1;j<4;j++){
            biggerN = (fpdp_neighbor[j]>fp2) ? j:biggerN;
            fp2 = (fpdp_neighbor[j]>fp2) ? fpdp_neighbor[j]:fp2;
        }
        fpdp_neighbor[i] = fp2;
        fpdp_neighbor[biggerN] = fp1;
    }
    
    //not going well if non-diag factors are considered
    float dp_cnd[9];
    dp_cnd[0] = dp - 4.0f*lambda1;// - lambda2;
    dp_cnd[1] = fpdp_neighbor[0] - fp;
    dp_cnd[2] = dp - 2.0f*lambda1;// - lambda2;
    dp_cnd[3] = fpdp_neighbor[1] - fp;
    dp_cnd[4] = dp;// - lambda2;
    dp_cnd[5] = fpdp_neighbor[2] - fp;
    dp_cnd[6] = dp + 2.0f*lambda1;// - lambda2;
    dp_cnd[7] = fpdp_neighbor[3] - fp;
    dp_cnd[8] = dp + 4.0f*lambda1;// - lambda2;
    
    float fp_range[8];
    fp_range[0] = fpdp_neighbor[0] + 4.0f*lambda1;// + lambda2;
    fp_range[1] = fpdp_neighbor[0] + 2.0f*lambda1;// + lambda2;
    fp_range[2] = fpdp_neighbor[1] + 2.0f*lambda1;// + lambda2;
    fp_range[3] = fpdp_neighbor[1];// + lambda2;
    fp_range[4] = fpdp_neighbor[2];// + lambda2;
    fp_range[5] = fpdp_neighbor[2] - 2.0f*lambda1;// + lambda2;
    fp_range[6] = fpdp_neighbor[3] - 2.0f*lambda1;// + lambda2;
    fp_range[7] = fpdp_neighbor[3] - 4.0f*lambda1;// + lambda2;
    
    float dp_new=dp_cnd[0];
    for(int i=0;i<8;i++){
        dp_new = (fp+dp<fp_range[i]) ? dp_cnd[i+1]:dp_new;
    }
    
    dp_cnd_img[global_ID] = (p_fix[z]==48) ? 0.0f:dp_new;
}

__kernel void FISTAupdate(__global float* fp_new_img,__global float* fp_old_img,
                          __global float* beta_img,__global float* w_img){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_ID = global_x+global_y*IMAGESIZE_X;
    
    //update of beta image
    float beta = beta_img[global_ID];
    float beta_new = beta*beta*4.0f + 1.0f;
    beta_new = (sqrt(beta_new) + 1.0f)*0.5f;
    beta_img[global_ID] = beta_new;
    
    //update of w img
    float w_new;
    for(int i=0;i<PARA_NUM;i++){
        w_new= fp_new_img[global_ID+i*IMAGESIZE_M] + (beta - 1.0f)*(fp_new_img[global_ID+i*IMAGESIZE_M]-fp_old_img[global_ID+i*IMAGESIZE_M])/beta_new;
        w_img[global_ID+i*IMAGESIZE_M]=w_new;
    }
    
}

__kernel void powerIteration(__global float* A_img, __global float* maxEval_img,
                             int iter){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_ID = global_x+global_y*IMAGESIZE_X;
    
    float A[PARA_NUM_SQ];
    float x1[PARA_NUM], x2[PARA_NUM];;
    for(int i=0;i<PARA_NUM;i++){
        A[i+i*PARA_NUM] = A_img[global_ID+(PARA_NUM*i-(i-1)*i/2)*IMAGESIZE_M];
        for(int j=i+1;j<PARA_NUM;j++){
            A[i+j*PARA_NUM] = A_img[global_ID+(PARA_NUM*i-(i+1)*i/2+j)*IMAGESIZE_M];
            A[j+i*PARA_NUM] = A[i+j*PARA_NUM];
        }
        x1[i]=1.0f;
    }
    
    float absX;
    for(int k=0;k<iter;k++){
        absX = 0.0f;
        for(int i=0;i<PARA_NUM;i++){
            x2[i]=0.0f;
            for(int j=0;j<PARA_NUM;j++){
                x2[i] += A[i+j*PARA_NUM]*x1[j];
            }
            absX += x2[i]*x2[i];
        }
        
        for(int i=0;i<PARA_NUM;i++){
            x1[i] = x2[i]/absX;
        }
    }
    
    maxEval_img[global_ID] = absX;
}
