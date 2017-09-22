#define PI      3.14159265358979323846
#define PI_2    1.57079632679489661923
#define EFF     0.262468426103175
#define K_PITCH 0.05

#define DELTA_S02 1.0E-6
#define DELTA_CN 1.0E-4
#define DELTA_DR 1.0E-5
#define DELTA_DE0 1.0E-4
#define DELTA_SS 1.0E-4
#define DELTA_E0IMAG 1.0E-4
#define DELTA_C3 1.0E-4
#define DELTA_C4 1.0E-4


#ifndef PARA_NUM
#define PARA_NUM 1
#endif

#ifndef PARA_NUM_SQ
#define PARA_NUM_SQ 1
#endif


#ifndef FFT_SIZE
#define FFT_SIZE 2048
#endif

extern float2 cmplxMult(float2 A, float2 B);
extern float2 cmplxMultConj(float2 Aconj, float2 B);
extern float cmplxAbs(float2 A);
extern float cmplxAbs2(float2 A);
extern float2 cmplxSqrt(float2 A);
extern float2 cmplxExp(float2 A);
extern float2 cmplxConj(float2 A);
extern float2 cmplxDiv(float2 A, float2 B);

//uncertainty: mean of magnutude of FT @ R=15-25

static __constant sampler_t s_linear = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE;


__kernel void hanningWindowFuncIMGarray(__global float2* wave, float zmin, float zmax,
                                        float windz, float zgrid){
    
    const size_t x_id = get_global_id(0);
    const size_t y_id = get_global_id(1);
    const size_t z_id = get_global_id(2);
    const size_t x_size = get_global_size(0);
    const size_t y_size = get_global_size(1);
    const size_t z_offset = get_global_offset(2);
    const size_t xy_size = x_size*y_size;
    const size_t xyz_id = x_id + x_size*y_id + xy_size*(z_id-z_offset);
    
    
    if(z_id*zgrid<zmin-windz){
        wave[xyz_id] = 0.0f;
    }else if(z_id*zgrid<zmin+windz){
        wave[xyz_id] *= 0.5f*(1.0f+cos((float)PI*(zgrid*z_id-zmin-windz)/windz/2));
    }else if(z_id*zgrid<zmax-windz){
        wave[xyz_id] *= 1.0f;
    }else if(z_id*zgrid<zmax+windz){
        wave[xyz_id] *= 0.5f*(1.0f+cos((float)PI*(zgrid*z_id-zmax+windz)/windz/2));
    }else{
        wave[xyz_id] = 0.0f;
    }
}


__kernel void redimension_feffShellPara(__global float* paraW,__read_only image1d_t paraW_raw,
                                        __constant float* kw, int numPnts){
    
    
    float k1, k2, k3;
    int n = 0;
    float X;
    float4 img;
    
    for(int kn=0; kn<numPnts; kn++){
        k1 = kw[kn];
        k2 = kw[kn+1];
        
        if(k2>=0.0f && k1<20.0f+K_PITCH){
            k3 = n*K_PITCH;
            while(k3<=k2){
                X=kn+(k3-k1)/(k2-k1)+0.5f;
                img = read_imagef(paraW_raw,s_linear,X);
                paraW[n] = img.x;
                n++;
                k3 = n*K_PITCH;
                if(k3==20.0f+K_PITCH) break;
            }
        }
        if(k3==20.0f+K_PITCH) break;
    }
}


//real2phc:     total central atom phase shift Re(2phc)
//mag:          Back scattering amplitude for each shell
//phase:        Phase shift for each shell
//redFactor:    Red factor of central atom
//lambda:       The mean free path in angstrom, lambda = 1/|Im(p)|
//real_p:       real part of local momentum Re(p)
inline float2 EXAFSshell(float kval,float Reff,
                         float S02,float CN,float dR,float dE0,float ss,
                         float E0img,float C3,float C4,
                         float real2phcW, float mag, float phase, float redFactor,
                         float lambdaW,   float real_p){
    
    //total central atom phase shift Re(2phc)
    float2 real2phc = (float2)(real2phcW,0.0f);
    //restrict lambda
    float lambda   = fmax(lambdaW, 0.0001f);
    
    float r2m2   = (Reff+dR)*(Reff+dR);
    r2m2         = 1.0f/r2m2;
    float s02r2n = S02*CN*r2m2;
    
    //imaginary energy
    float2 cE0i  = (float2)(0.0f,E0img*EFF);
    
    //corrected energy
    float E0eff  = dE0*EFF;
    float energy	= kval*kval-E0eff;
    //corrected wavenumber q
    float qf        = (energy>=0.0f) ? sqrt(energy):sqrt(-energy);
    
    //complexed local momentum p = Re(p)+1/lambda
    float2 cp      = (float2)(real_p,1.0f/lambda);
    float2 cp2     = cmplxMult(cp,cp)+cE0i;
    cp 		= cmplxSqrt(cp2);
    
    //mean free path
    float2 cxlam 	= (float2)(-2.0f*Reff*cp.y,0.0f);
    
    //complexed Debye-Waller factor with 4th cumulant
    float2 cdwf 	= -2.0f*(ss*cp2 - cmplxMult(cp2,cp2)*C4/3.0f);
    
    
    float C1     = dR-2.0f*ss/Reff;
    //complexed phase shift with 3rd cumulant
    float2 cphshf 	= 2.0f*(C1*cp - cmplxMult(cp,cp2)*C3*2.0f/3.0f) + real2phc;
    
    //complex chi
    float2 cargu 	= (float2)(2.0f*qf*Reff+phase,0) + cphshf ;
    cargu           = cxlam + cdwf + (float2)(-cargu.y,cargu.x);
    cargu.x 		= fmax(-30.0f, fmin(30.0f, cargu.x));
    float2 chi      = (energy>=0.0f) ? mag*s02r2n/qf*cmplxExp(cargu)*redFactor:0.0f;
    
    return chi;
}


__kernel void outputchi(__global float2* chi, float Reff, __global float* S02,
                        __global float* CN, __global float* dR,
                        __global float* dE0, __global float* ss,
                        __global float* E0imag, __global float* C3,__global float* C4,
                        __global float* real2phcW, __global float* magW,
                        __global float* phaseW, __global float* redFactorW,
                        __global float* lambdaW, __global float* real_pW,
                        int kw){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t offset_z = get_global_offset(2);
    const size_t IDxy = global_x + global_y*size_x;
    const size_t IDxyz = IDxy + (global_z-offset_z)*size_x*size_y;
    
    float real2phc = real2phcW[global_z];
    float mag = magW[global_z];
    float phase = phaseW[global_z];
    float redFactor = redFactorW[global_z];
    float lambda = lambdaW[global_z];
    float real_p = real_pW[global_z];
    
    float kval = K_PITCH*global_z;
    
    float2 chi_c=EXAFSshell(kval,Reff,S02[IDxy],CN[IDxy],dR[IDxy],dE0[IDxy],ss[IDxy],
                            E0imag[IDxy],C3[IDxy],C4[IDxy],
                            real2phc,mag,phase,redFactor,lambda,real_p);
    
    float wgt;
    switch(kw){
        case 0:
            wgt=1.0f;
            break;
        case 1:
            wgt=kval;
            break;
        case 2:
            wgt=kval*kval;
            break;
        case 3:
            wgt=kval*kval*kval;
            break;
    }
    chi_c.x=0.0f;
    
    chi[IDxyz] += chi_c*wgt;
}


__kernel void outputchi_r(__global float* chi, float Reff, __global float* S02,
                           __global float* CN, __global float* dR,
                          __global float* dE0, __global float* ss,
                          __global float* E0imag, __global float* C3,__global float* C4,
                          __global float* real2phcW, __global float* magW,
                          __global float* phaseW, __global float* redFactorW,
                          __global float* lambdaW, __global float* real_pW,
                          int kw){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t offset_z = get_global_offset(2);
    const size_t IDxy = global_x + global_y*size_x;
    const size_t IDxyz = IDxy + (global_z-offset_z)*size_x*size_y;
    
    float real2phc = real2phcW[global_z];
    float mag = magW[global_z];
    float phase = phaseW[global_z];
    float redFactor = redFactorW[global_z];
    float lambda = lambdaW[global_z];
    float real_p = real_pW[global_z];
    
    float kval = K_PITCH*global_z;
    
    float2 chi_c=EXAFSshell(kval,Reff,S02[IDxy],CN[IDxy],dR[IDxy],dE0[IDxy],ss[IDxy],
                            E0imag[IDxy],C3[IDxy],C4[IDxy],
                            real2phc,mag,phase,redFactor,lambda,real_p);
    
    float wgt;
    switch(kw){
        case 0:
            wgt=1.0f;
            break;
        case 1:
            wgt=kval;
            break;
        case 2:
            wgt=kval*kval;
            break;
        case 3:
            wgt=kval*kval*kval;
            break;
    }
    chi_c.x=0.0f;
    
    chi[IDxyz] += chi_c.y*wgt;
}



inline float2 Jacobian_k_S02(float kval,float Reff,
                             float S02,float CN,float dR,float dE0,float ss,
                             float E0img,float C3,float C4,
                             float real2phcW, float mag, float phase, float redFactor,
                             float lambdaW,   float real_p){
    
    
    float2 J=EXAFSshell(kval,Reff,S02,CN,dR,dE0,ss,E0img,C3,C4,
                        real2phcW,mag,phase,redFactor,lambdaW,real_p);
    J/=S02;
    return J;
}

inline float2 Jacobian_k_CN(float kval,float Reff,
                            float S02,float CN,float dR,float dE0,float ss,
                            float E0img,float C3,float C4,
                            float real2phcW, float mag, float phase, float redFactor,
                            float lambdaW,   float real_p){
    
    
    float2 J=EXAFSshell(kval,Reff,S02,CN,dR,dE0,ss,E0img,C3,C4,
                        real2phcW,mag,phase,redFactor,lambdaW,real_p);
    J/=CN;
    return J;
}


inline float2 Jacobian_k_dR(float kval,float Reff,
                            float S02,float CN,float dR,float dE0,float ss,
                            float E0img,float C3,float C4,
                            float real2phcW, float mag, float phase, float redFactor,
                            float lambdaW,   float real_p){
    
    
    float2 J=EXAFSshell(kval,Reff,S02,CN,dR,dE0,ss,E0img,C3,C4,
                        real2phcW,mag,phase,redFactor,lambdaW,real_p);
    
    //imaginary energy
    float2 ciEi  = (float2)(0.0f,E0img*EFF);
    
    //complexed local momentum p = Re(p)+1/lambda
    float2 cp      = (float2)(real_p,1.0f/lambdaW);
    float2 cp2     = cmplxMult(cp,cp)+ciEi;
    cp 		= cmplxSqrt(cp2);
    
    float2 diff = (float2)(-2/(Reff+dR),0.0f) + (float2)(-cp.y,cp.x);
    J = cmplxMult(J,diff);
    
    return J;
}


inline float2 Jacobian_k_dE0(float kval,float Reff,
                             float S02,float CN,float dR,float dE0,float ss,
                             float E0img,float C3,float C4,
                             float real2phcW, float mag, float phase, float redFactor,
                             float lambdaW,   float real_p){
    
    
    float2 J=EXAFSshell(kval,Reff,S02,CN,dR,dE0,ss,E0img,C3,C4,
                        real2phcW,mag,phase,redFactor,lambdaW,real_p);
    
    //corrected energy
    float E0eff  = dE0*EFF;
    float energy	= kval*kval-E0eff;
    //corrected wavenumber q
    float qf        = (energy>=0.0f) ? sqrt(energy):sqrt(-energy);
    
    float2 diff = (float2)(1.0f/qf,2.0f*Reff)*(float)EFF/qf/2.0f;
    J = cmplxMult(J,diff);
    
    return J;
}


inline float2 Jacobian_k_ss(float kval,float Reff,
                            float S02,float CN,float dR,float dE0,float ss,
                            float E0img,float C3,float C4,
                            float real2phcW, float mag, float phase, float redFactor,
                            float lambdaW,   float real_p){
    
    
    float2 J=EXAFSshell(kval,Reff,S02,CN,dR,dE0,ss,E0img,C3,C4,
                        real2phcW,mag,phase,redFactor,lambdaW,real_p);
    
    //imaginary energy
    float2 ciEi  = (float2)(0.0f,E0img*EFF);
    
    //complexed local momentum p = Re(p)+1/lambda
    float2 cp      = (float2)(real_p,1.0f/lambdaW);
    float2 cp2     = cmplxMult(cp,cp)+ciEi;
    cp 		= cmplxSqrt(cp2);
    
    float2 diff = -2.0f*cp2-4.0f*(float2)(-cp.y,cp.x)/Reff;
    J = cmplxMult(J,diff);
    
    return J;
}


inline float2 Jacobian_k_E0img(float kval,float Reff,
                               float S02,float CN,float dR,float dE0,float ss,
                               float E0img,float C3,float C4,
                               float real2phcW, float mag, float phase, float redFactor,
                               float lambdaW,   float real_p){
    
    
    float2 J=EXAFSshell(kval,Reff,S02,CN,dR,dE0,ss,E0img,C3,C4,
                        real2phcW,mag,phase,redFactor,lambdaW,real_p);
    
    //imaginary energy
    float2 ciEi  = (float2)(0.0f,E0img*EFF);
    
    //complexed local momentum p = Re(p)+1/lambda
    float2 cp      = (float2)(real_p,1.0f/lambdaW);
    float2 cp2     = cmplxMult(cp,cp)+ciEi;
    cp 		= cmplxSqrt(cp2);
    
    float2 diff1 = (-4.0f*cp*ss -8.0f/3.0f*cmplxMult(cp,cp2)*C4)*(float)EFF/2.0f;
    float2 diff2 = ((float2)(dR,0.0f)-2.0f*C3*cp2)*(float)EFF;
    diff2 = (float2)(-diff2.y,diff2.x);
    
    J = cmplxMult(J,diff1+diff2);
    J = cmplxDiv(J, cp);
    
    return J;
}


inline float2 Jacobian_k_C3(float kval,float Reff,
                            float S02,float CN,float dR,float dE0,float ss,
                            float E0img,float C3,float C4,
                            float real2phcW, float mag, float phase, float redFactor,
                            float lambdaW,   float real_p){
    
    
    float2 J=EXAFSshell(kval,Reff,S02,CN,dR,dE0,ss,E0img,C3,C4,
                        real2phcW,mag,phase,redFactor,lambdaW,real_p);
    
    //imaginary energy
    float2 ciEi  = (float2)(0.0f,E0img*EFF);
    
    //complexed local momentum p = Re(p)+1/lambda
    float2 cp      = (float2)(real_p,1.0f/lambdaW);
    float2 cp2     = cmplxMult(cp,cp)+ciEi;
    cp 		= cmplxSqrt(cp2);
    
    float2 diff = -4.0f/3.0f*cmplxMult(cp,cp2);
    diff = (float2)(-diff.y,diff.x);
    J = cmplxMult(J,diff);
    
    return J;
}


inline float2 Jacobian_k_C4(float kval,float Reff,
                            float S02,float CN,float dR,float dE0,float ss,
                            float E0img,float C3,float C4,
                            float real2phcW, float mag, float phase, float redFactor,
                            float lambdaW,   float real_p){
    
    
    float2 J=EXAFSshell(kval,Reff,S02,CN,dR,dE0,ss,E0img,C3,C4,
                        real2phcW,mag,phase,redFactor,lambdaW,real_p);
    
    //imaginary energy
    float2 ciEi  = (float2)(0.0f,E0img*EFF);
    
    //complexed local momentum p = Re(p)+1/lambda
    float2 cp      = (float2)(real_p,1.0f/lambdaW);
    float2 cp2     = cmplxMult(cp,cp)+ciEi;
    cp 		= cmplxSqrt(cp2);
    
    float2 diff = -2.0f/3.0f*cmplxMult(cp2,cp2);
    J = cmplxMult(J,diff);
    
    return J;
}

__kernel void Jacobian_k_new(__global float2* J, float Reff, __global float* S02,
                             __global float* CN, __global float* dR,
                             __global float* dE0, __global float* ss,
                             __global float* E0imag, __global float* C3,__global float* C4,
                             __global float* real2phcW, __global float* magW,
                             __global float* phaseW, __global float* redFactorW,
                             __global float* lambdaW, __global float* real_pW,
                             int kw, int paramode, int realpart){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t offset_z = get_global_offset(2);
    const size_t IDxy = global_x + global_y*size_x;
    const size_t IDxyz = global_x + global_y*size_x + size_x*size_y*(global_z-offset_z);
    
    float real2phc = real2phcW[global_z];
    float mag = magW[global_z];
    float phase = phaseW[global_z];
    float redFactor = redFactorW[global_z];
    float lambda = lambdaW[global_z];
    float real_p = real_pW[global_z];
    
    float S02_p = S02[IDxy];
    float CN_p = CN[IDxy];
    float dR_p = dR[IDxy];
    float dE0_p = dE0[IDxy];
    float ss_p = ss[IDxy];
    float E0imag_p = E0imag[IDxy];
    float C3_p = C3[IDxy];
    float C4_p = C4[IDxy];
    
    float2 J_p;
    float kval = K_PITCH*global_z;
    switch(paramode){
        case 0: //S02
            J_p = Jacobian_k_S02(kval,Reff,S02_p,CN_p,dR_p,dE0_p,ss_p,E0imag_p,C3_p,C4_p,
                                 real2phc,mag,phase,redFactor,lambda,real_p);
            break;
        case 1: //CN
            J_p = Jacobian_k_CN(kval,Reff,S02_p,CN_p,dR_p,dE0_p,ss_p,E0imag_p,C3_p,C4_p,
                                real2phc,mag,phase,redFactor,lambda,real_p);
            break;
        case 2: //dR
            J_p = Jacobian_k_dR(kval,Reff,S02_p,CN_p,dR_p,dE0_p,ss_p,E0imag_p,C3_p,C4_p,
                                real2phc,mag,phase,redFactor,lambda,real_p);
            break;
        case 3: //dE0
            J_p = Jacobian_k_dE0(kval,Reff,S02_p,CN_p,dR_p,dE0_p,ss_p,E0imag_p,C3_p,C4_p,
                                 real2phc,mag,phase,redFactor,lambda,real_p);
            break;
        case 4: //ss
            J_p = Jacobian_k_ss(kval,Reff,S02_p,CN_p,dR_p,dE0_p,ss_p,E0imag_p,C3_p,C4_p,
                                real2phc,mag,phase,redFactor,lambda,real_p);
            break;
        case 5: //E0imag
            J_p = Jacobian_k_E0img(kval,Reff,S02_p,CN_p,dR_p,dE0_p,ss_p,E0imag_p,C3_p,C4_p,
                                   real2phc,mag,phase,redFactor,lambda,real_p);
            break;
        case 6: //C3
            J_p = Jacobian_k_C3(kval,Reff,S02_p,CN_p,dR_p,dE0_p,ss_p,E0imag_p,C3_p,C4_p,
                                real2phc,mag,phase,redFactor,lambda,real_p);
            break;
        case 7: //C4
            J_p = Jacobian_k_C4(kval,Reff,S02_p,CN_p,dR_p,dE0_p,ss_p,E0imag_p,C3_p,C4_p,
                                real2phc,mag,phase,redFactor,lambda,real_p);
            break;
    }
     if(realpart==0) J_p.x=0.0f;
    
    
    float wgt;
    switch(kw){
        case 0:
            wgt=1.0f;
            break;
        case 1:
            wgt=kval;
            break;
        case 2:
            wgt=kval*kval;
            break;
        case 3:
            wgt=kval*kval*kval;
            break;
    }
    J_p *= wgt;
    
    J[IDxyz] += J_p;
}

__kernel void Jacobian_k(__global float2* J, float Reff, __global float* S02,
                         __global float* CN, __global float* dR,
                         __global float* dE0, __global float* ss,
                         __global float* E0imag, __global float* C3,__global float* C4,
                         __global float* real2phcW, __global float* magW,
                         __global float* phaseW, __global float* redFactorW,
                         __global float* lambdaW, __global float* real_pW,
                         int kw, int paramode, int realpart){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t offset_z = get_global_offset(2);
    const size_t IDxy = global_x + global_y*size_x;
    const size_t IDxyz = global_x + global_y*size_x + size_x*size_y*(global_z-offset_z);
    
    float real2phc = real2phcW[global_z];
    float mag = magW[global_z];
    float phase = phaseW[global_z];
    float redFactor = redFactorW[global_z];
    float lambda = lambdaW[global_z];
    float real_p = real_pW[global_z];
    
    float S02_nu = S02[IDxy];
    float CN_nu = CN[IDxy];
    float dR_nu = dR[IDxy];
    float dE0_nu = dE0[IDxy];
    float ss_nu = ss[IDxy];
    float E0imag_nu = E0imag[IDxy];
    float C3_nu = C3[IDxy];
    float C4_nu = C4[IDxy];
    
    float S02_p = S02_nu;
    float CN_p = CN_nu;
    float dR_p = dR_nu;
    float dE0_p = dE0_nu;
    float ss_p = ss_nu;
    float E0imag_p = E0imag_nu;
    float C3_p = C3_nu;
    float C4_p = C4_nu;
    
    float S02_m = S02_nu;
    float CN_m = CN_nu;
    float dR_m = dR_nu;
    float dE0_m = dE0_nu;
    float ss_m = ss_nu;
    float E0imag_m = E0imag_nu;
    float C3_m = C3_nu;
    float C4_m = C4_nu;
    float pitch;
    
    switch(paramode){
        case 0: //S02
            S02_p += DELTA_S02;
            S02_m -= DELTA_S02;
            pitch = 2.0f*DELTA_S02;
            break;
        case 1: //CN
            CN_p += DELTA_CN;
            CN_m -= DELTA_CN;
            pitch = 2.0f*DELTA_CN;
            break;
        case 2: //dR
            dR_p += DELTA_DR;
            dR_m -= DELTA_DR;
            pitch = 2.0f*DELTA_DR;
            break;
        case 3: //dE0
            dE0_p += DELTA_DE0;
            dE0_m -= DELTA_DE0;
            pitch = 2.0f*DELTA_DE0;
            break;
        case 4: //ss
            ss_p += DELTA_SS;
            ss_m -= DELTA_SS;
            pitch = 2.0f*DELTA_SS;
            break;
        case 5: //E0imag
            E0imag_p += DELTA_E0IMAG;
            E0imag_m -= DELTA_E0IMAG;
            pitch = 2.0f*DELTA_E0IMAG;
            break;
        case 6: //C3
            C3_p += DELTA_C3;
            C3_m -= DELTA_C3;
            pitch = 2.0f*DELTA_C3;
            break;
        case 7: //C4
            C4_p += DELTA_C4;
            C4_m -= DELTA_C4;
            pitch = 2.0f*DELTA_C4;
            break;
    }
    
    float kval = K_PITCH*global_z;
    
    float2 chifit_p=EXAFSshell(kval,Reff,S02_p,CN_p,dR_p,dE0_p,ss_p,
                               E0imag_p,C3_p,C4_p,
                               real2phc,mag,phase,redFactor,lambda,real_p);
    
    float2 chifit_m=EXAFSshell(kval,Reff,S02_m,CN_m,dR_m,dE0_m,ss_m,
                               E0imag_m,C3_m,C4_m,
                               real2phc,mag,phase,redFactor,lambda,real_p);
    
    float2 delta_chifit = (chifit_p-chifit_m)/pitch;
    if(realpart==0) delta_chifit.x=0.0f;
    
    float wgt;
    switch(kw){
        case 0:
            wgt=1.0f;
            break;
        case 1:
            wgt=kval;
            break;
        case 2:
            wgt=kval*kval;
            break;
        case 3:
            wgt=kval*kval*kval;
            break;
    }
    delta_chifit *= wgt;
    
    J[IDxyz] += delta_chifit;
}

__kernel void estimate_tJJ(__global float* tJJ, __global float2* J1, __global float2* J2,
                           int z_id, int kRqsize){
    //caution: z_id is for tJdF (parameter number) and kRqsize for chi_data/chi_fit (k/R/q) size
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t IDxy = global_x + global_y*size_x;
    size_t IDxyz1;
    const size_t IDxyz2 = IDxy + z_id*size_x*size_y;
    
    float2 J1_p, J2_p;
    float tJJ_p=0.0f;
    for(int i=0; i<kRqsize; i++){
        IDxyz1 = IDxy + i*size_x*size_y;
        
        J1_p = J1[IDxyz1];
        J2_p = J2[IDxyz1];
        
        tJJ_p += J1_p.x*J2_p.x+J1_p.y*J2_p.y;
    }
    
    tJJ[IDxyz2] = tJJ_p;
}

__kernel void estimate_tJdF(__global float* tJdF, __global float2* J,
                            __global float2* chi_data, __global float2* chi_fit,
                            int z_id, int kRqsize){
    //caution: z_id is for tJdF (parameter number) and kRqsize for chi_data/chi_fit (k/R/q) size
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t IDxy = global_x + global_y*size_x;
    size_t IDxyz1;
    const size_t IDxyz2 = IDxy + z_id*size_x*size_y;
    
    float2 J_p, dF_p;
    float tJdF_p = 0.0f;
    for(int i=0; i<kRqsize; i++){
        IDxyz1 = IDxy + i*size_x*size_y;
        
        J_p = J[IDxyz1];
        dF_p = chi_data[IDxyz1] - chi_fit[IDxyz1];
        
        tJdF_p += J_p.x*dF_p.x+J_p.y*dF_p.y;
    }
    
    tJdF[IDxyz2] = tJdF_p;
}

__kernel void estimate_dF2(__global float* dF2,
                           __global float2* chi_data, __global float2* chi_fit,
                           int kRqsize){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t IDxy = global_x + global_y*size_x;
    size_t IDxyz;
    
    float2 dF_p;
    float dF2_p = 0.0f;
    for(int i=0; i<kRqsize; i++){
        IDxyz = IDxy + i*size_x*size_y;
        
        dF_p = chi_data[IDxyz] - chi_fit[IDxyz];
        dF2_p += cmplxAbs2(dF_p);
    }
    
    dF2[IDxy] = dF2_p;
}

__kernel void estimate_Rfactor(__global float* Rfactor,
                               __global float2* chi_data, __global float2* chi_fit,
                               int kRqsize){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t IDxy = global_x + global_y*size_x;
    size_t IDxyz;
    
    float2 dF_p, chiData_p;
    float dF2_p = 0.0f;
    float chi2_p = 0.0f;
    for(int i=0; i<kRqsize; i++){
        IDxyz = IDxy + i*size_x*size_y;
        
        chiData_p = chi_data[IDxyz];
        dF_p = chiData_p - chi_fit[IDxyz];
        dF2_p  += cmplxAbs2(dF_p);
        chi2_p += cmplxAbs2(chiData_p);
    }
    
    Rfactor[IDxy] = dF2_p/chi2_p;
}


//variable error when delta(chi2)=1 is |2*tJdF|
__kernel void estimate_error(__global float* tJdF, __global float* p_error){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t IDxyz = global_x + global_y*size_x + global_z*size_x*size_y;
    
    float error = tJdF[IDxyz];
    error = fabs(2.0f*error);
    
    p_error[IDxyz] = error;
}



__kernel void chi2cmplxChi_imgStck(__global float* chi, __global float2* chi_c,
                                   int kn, int koffset, int kw){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t x_size = get_global_size(0);
    const size_t y_size = get_global_size(1);
    const size_t IDxy = global_x+global_y*x_size;
    const size_t IDxyz1 = IDxy + global_z*x_size*y_size;
    const size_t IDxyz2 = IDxy + kn*x_size*y_size;
    
    
    float2 chi_cp;
    chi_cp.x = 0.0f;
    chi_cp.y = chi[IDxyz1];
    
    float weight;
    float kval = (kn+koffset)*K_PITCH;
    switch(kw){
        case 0:
            weight = 1.0f;
            break;
        case 1:
            weight = kval;
            break;
        case 2:
            weight = kval*kval;
            break;
        case 3:
            weight = kval*kval*kval;
            break;
    }
    
    chi_c[IDxyz2] = chi_cp*weight;
}

__kernel void chi2cmplxChi_chiStck(__global float* chi, __global float2* chi_c,
                                   int XY, int kw, int XY_size){
    
    const size_t global_x = get_global_id(0);
    const size_t offset_x = get_global_offset(0);
    const size_t IDxyz = XY + XY_size*(global_x-offset_x);
    
    
    float2 chi_cp;
    chi_cp.x = 0.0f;
    chi_cp.y = chi[global_x];
    
    float weight;
    float kval = global_x*K_PITCH;
    switch(kw){
        case 0:
            weight = 1.0f;
            break;
        case 1:
            weight = kval;
            break;
        case 2:
            weight = kval*kval;
            break;
        case 3:
            weight = kval*kval*kval;
            break;
    }
    
    chi_c[IDxyz] = chi_cp*weight;
}


__kernel void CNweighten(__global float* CN, __global float* edgeJ, float iniCN){
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t size_x = get_global_size(0);
    const size_t IDxy = global_x + global_y*size_x;
    
    float CN_p = iniCN*edgeJ[IDxy];
    
    CN[IDxy] = CN_p;
}


__kernel void outputBondDistance(__global float* dR, __global float* R, float Reff){
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t imageSizeX = get_global_size(0);
    const size_t global_ID = global_x+global_y*imageSizeX;
    
    R[global_ID] = dR[global_ID]+Reff;
}
