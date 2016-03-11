#define PI      3.14159265358979323846
#define PI_2    1.57079632679489661923
#define EFF     0.262468426103175
#define FFT_SIZE 1024

static __constant sampler_t s_linear = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;


float2 cmplxMult(float2 A, float2 B){
    return (A.x*B.x - A.y*B.y, A.x*B.y + A.y*B.x);
}


float cmplxAbs(float2 A){
    return sqrt(A.x*A.x+A.y*A.y);
}


float2 cmplxSqrt(float2 A){
    float2 B;
    
    if(A.x>=0){
        B = (float2)(sqrt(A.x+cmplxAbs(A)),sqrt(-A.x+cmplxAbs(A)));
    }else{
        B = (float2)(sqrt(A.x+cmplxAbs(A)),-sqrt(-A.x+cmplxAbs(A)));
    }
    B *= rsqrt(2.0f);
    
    return B;
}


float2 cmplxExp(float2 A){
    float2 B = (float2)(A.x,A.x-PI_2);
    B = cos(B)*exp(A.x);
    
    return B;
}


__kernel void spinFact(__global float2* w, int N){
    
    unsigned int i = get_global_id(0);
    
    float2 angle = (float2)(2*i*PI/(float)N,(2*i*PI/(float)N)+PI_2);
    
    w[i] = cos(angle);
    
}


__kernel void bitReverseIMGarray(__global float2* x1, __global float2* x2, int M){
    
    const size_t x_id = get_global_id(0);
    const size_t y_id = get_global_id(1);
    const size_t x_size = get_global_size(0);
    const size_t y_size = get_global_size(1);
    const size_t xy_id = x_id + x_size*y_id;
    const size_t xy_size =x_size*y_size;
    
    for(unsigned int i=0; i < 1 << (M-1); i++){
        unsigned int j=i;
        
        j = (j & 0x55555555) << 1  | (j & 0xAAAAAAAA) >> 1;
        j = (j & 0x33333333) << 2  | (j & 0xCCCCCCCC) >> 2;
        j = (j & 0x0F0F0F0F) << 4  | (j & 0xF0F0F0F0) >> 4;
        j = (j & 0x00FF00FF) << 8  | (j & 0xFF00FF00) >> 8;
        j = (j & 0x0000FFFF) << 16 | (j & 0xFFFF0000) >> 16;
        
        j >>= (32-M);
        
        x2[xy_id + i*xy_size] = x1[xy_id + j*xy_size];
        
    }
}


__kernel void butterflyIMGarray(__global float2* x_fft, __constant float2* w, int N, int iter){
    
    const size_t x_id = get_global_id(0);
    const size_t y_id = get_global_id(1);
    const size_t x_size = get_global_size(0);
    const size_t y_size = get_global_size(1);
    const size_t xy_id = x_id + x_size*y_id;
    const size_t xy_size =x_size*y_size;
    

    
    unsigned int m = 1 << (iter-1);
    
    for(unsigned int i=0; i < N; i <<= iter){
        for(unsigned int j=0;j<m;j++){
            
            float2 val1 = x_fft[xy_id+(i+j)*xy_size];
            float2 val2 = x_fft[xy_id+(i+j+m)*xy_size];
            
            float2 vala = val1 + cmplxMult(val1,w[j]);
            float2 valb = val2 - cmplxMult(val2,w[j]);
            
            x_fft[xy_id+(i+j)*xy_size]      = vala;
            x_fft[xy_id+(i+j+m)*xy_size]    = valb;
        }
    }
    
}


__kernel void FFTnormIMGarray(__global float2* x_fft,float xgrid, int N){
    const size_t x_id = get_global_id(0);
    const size_t y_id = get_global_id(1);
    const size_t x_size = get_global_size(0);
    const size_t y_size = get_global_size(1);
    const size_t xy_id = x_id + x_size*y_id;
    const size_t xy_size =x_size*y_size;

    
    for(unsigned int i=0; i < N; i++){
            x_fft[xy_id+i*xy_size] *= xgrid/sqrt((float)PI);
    }
}


__kernel void IFFTnormIMGarray(__global float2* x_ifft,float xgrid, int N){
    const size_t x_id = get_global_id(0);
    const size_t y_id = get_global_id(1);
    const size_t x_size = get_global_size(0);
    const size_t y_size = get_global_size(1);
    const size_t xy_id = x_id + x_size*y_id;
    const size_t xy_size =x_size*y_size;
    
    
    for(unsigned int i=0; i < N; i++){
        x_ifft[xy_id+i*xy_size] *= sqrt((float)PI)/xgrid;
    }
}


__kernel void EXAFSoscillationIMGarray(__global float2* chi,__global float* fit_p,
                                       __global float2* Jacob,
                                       __constant float* mag, __constant float* phase,
                                       __constant float* redFactor, __constant float* lambda,
                                       __constant float* real_2phc, __constant float* real_p,
                                       float Reff, int n_shell, float kgrid){
    
    
    const size_t x_id = get_global_id(0);
    const size_t y_id = get_global_id(1);
    const size_t z_id = get_global_id(2);
    const size_t x_size = get_global_size(0);
    const size_t y_size = get_global_size(1);
    const size_t z_size = get_global_size(2);
    const size_t xy_id = x_id + x_size*y_id;
    const size_t xy_size = x_size*y_size;
    const size_t xyz_id = x_id + x_size*y_id + xy_size*z_id;
    const size_t xyz_size = xy_size*FFT_SIZE;
    
    float third=0.0f, fourth=0.0f;
    float imag_E0=0.0f;
    
    float rm = 1/(Reff+fit_p[xy_id+xy_size*(n_shell*4+2)]);
    float r2m2 = 1/((Reff+fit_p[xy_id+xy_size*(n_shell*4+2)])*(Reff+fit_p[xy_id+xy_size*(n_shell*4+2)]));
    float S02r2n = fit_p[xy_id]*fit_p[xy_id+xy_size*(n_shell*4+1)]*r2m2;
    float first = fit_p[xy_id+xy_size*(n_shell*4+2)]-fit_p[xy_id+xy_size*(n_shell*4+4)]/Reff;
    float2 ciEi = (float2)(0.0f,imag_E0*EFF);
        

    float energy = (kgrid*z_id)*(kgrid*z_id)-fit_p[xy_id+xy_size*(n_shell*4+3)]*EFF;
    float q = (energy>=0) ? sqrt(energy):-sqrt(-energy);
    float2 cp2 = (float2)(real_p[z_id],1/fmax(lambda[z_id],0.0001f));
    cp2 = cmplxMult(cp2,cp2)+ciEi;
    float2 cp = cmplxSqrt(cp2);
    float2 cxlam = (float2)(-2*Reff*cp.y,0.0f);
    float2 cdwf = (float2)(fit_p[xy_id+xy_size*(n_shell*4+4)],0)-cp2*fourth/3;
    cdwf = -2*cmplxMult(cp2,cdwf);
    float2 cphshf = (float2)(first,0)-2*cp2*third/3;
    cphshf = (float2)(2*q*Reff+phase[z_id]+real_2phc[z_id],0) + 2*cmplxMult(cp,cphshf);
    float2 cargu =cmplxMult((float2)(0,1),cphshf) + cxlam + cdwf;
    cargu.x = minmag(cargu.x,30.0f);
        
    //Imag側がEXAFSの公式
    float2 chi_k = (energy>=0) ? mag[z_id]*S02r2n/q*redFactor[z_id]*cmplxExp(cargu):0.0f;
    //chi
    chi[xyz_id]                            += chi_k;
    //jacob(S02)
    Jacob[xyz_id]                          += chi_k/fit_p[xy_id];
    //jacob(CN)
    Jacob[xyz_id+xyz_size*(n_shell*4+1)]   = chi_k/fit_p[xy_id];
    //jacob(dR)
    Jacob[xyz_id+xyz_size*(n_shell*4+2)]   = -2*chi_k*(rm+cp);
    //jacob(dE0)
    Jacob[xyz_id+xyz_size*(n_shell*4+3)]   = (float)EFF*chi_k*(1/(2*energy)+Reff/q);
    //jacob(ss)
    Jacob[xyz_id+xyz_size*(n_shell*4+4)]   = 2*chi_k*(cp*rm-cp2);
    
}

__kernel void hanningWindowFuncIMGarray(__global float2* wave, float zmin, float zmax,
                         float windz, float zgrid){
    
    const size_t x_id = get_global_id(0);
    const size_t y_id = get_global_id(1);
    const size_t z_id = get_global_id(2);
    const size_t x_size = get_global_size(0);
    const size_t y_size = get_global_size(1);
    const size_t z_size = get_global_size(2);
    const size_t xy_id = x_id + x_size*y_id;
    const size_t xy_size = x_size*y_size;
    const size_t xyz_id = x_id + x_size*y_id + xy_size*z_id;
    const size_t xyz_size = xy_size*FFT_SIZE;
    

    if(z_id*zgrid<zmin-windz){
        wave[xyz_id] = 0.0f;
    }else if(z_id*zgrid<zmin){
        wave[xyz_id] *= 0.5f*(1.0f+cos((float)PI*(zgrid*z_id-zmin)/windz));
    }else if(z_id*zgrid>zmax){
        wave[xyz_id] = 0.0f;
    }else if(z_id*zgrid>zmax+windz){
        wave[xyz_id] *= 0.5f*(1.0f+cos((float)PI*(zgrid*z_id-zmax)/windz));
    }
}

__kernel void kWeightIMGarray(__global float2* chi, int kw, float kgrid){
    const size_t x_id = get_global_id(0);
    const size_t y_id = get_global_id(1);
    const size_t z_id = get_global_id(2);
    const size_t x_size = get_global_size(0);
    const size_t y_size = get_global_size(1);
    const size_t z_size = get_global_size(2);
    const size_t xy_id = x_id + x_size*y_id;
    const size_t xy_size = x_size*y_size;
    const size_t xyz_id = x_id + x_size*y_id + xy_size*z_id;
    const size_t xyz_size = xy_size*FFT_SIZE;
    
    switch(kw){
        case 1:
            chi[xyz_id] *= kgrid*z_id;
            break;
        case 2:
            chi[xyz_id] *= kgrid*z_id*kgrid*z_id;
            break;
        case 3:
            chi[xyz_id] *= kgrid*z_id*kgrid*z_id*kgrid*z_id;
            break;
        default:
            break;
    }
}

