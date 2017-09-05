#define PI      3.14159265358979323846
#define PI_2    1.57079632679489661923
#define EFF     0.262468426103175

#ifndef FFT_SIZE
#define FFT_SIZE 2048
#endif

float2 cmplxMult(float2 A, float2 B){
    
    return (float2)(A.x*B.x-A.y*B.y, A.x*B.y+A.y*B.x);
}


float2 cmplxMultConj(float2 Aconj, float2 B){
    
    return (float2)(Aconj.x*B.x + Aconj.y*B.y, Aconj.x*B.y - Aconj.y*B.x);
}

float cmplxAbs(float2 A){
    return sqrt(A.x*A.x+A.y*A.y);
}

float cmplxAbs2(float2 A){
    return A.x*A.x+A.y*A.y;
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
    float2 B = (float2)(A.y,A.y-PI_2);
    B = cos(B)*exp(A.x);
    
    return B;
}

float2 cmplxConj(float2 A){
    return (float2)(A.x,-A.y);
}


float2 cmplxDiv(float2 A, float2 B){
    
    float2 val = cmplxMultConj(B,A)/cmplxAbs2(B);
    
    return val;
}

__kernel void XY_transpose(__global float2* src, __global float2* dest){
    const size_t x_id = get_global_id(0);
    const size_t y_id = get_global_id(1);
    const size_t z_id = get_global_id(2);
    const size_t x_size = get_global_size(0);
    const size_t y_size = get_global_size(1);
    const size_t xy_id = x_id + x_size*y_id;
    const size_t xy_size =x_size*y_size;
    
    const size_t yx_id = y_id + y_size*x_id;
    
    dest[yx_id+xy_size*z_id] = src[xy_id+xy_size*z_id];
}

__kernel void XZ_transpose(__global float2* src, __global float2* dest,
                           int offsetX_src, int x_size_src, int offsetX_dst, int x_size_dst,
                           int offsetY_src, int y_size_src, int offsetY_dst, int y_size_dst){
    
    const size_t x_id = get_global_id(0);
    const size_t y_id = get_global_id(1);
    const size_t z_id = get_global_id(2);
    
    //size_t X = x_id+offsetX_src;
    //X = (X<FFT_SIZE/2) ? X+FFT_SIZE/2:X-FFT_SIZE/2;
    
    size_t ID_src = (x_id+offsetX_src) + x_size_src*(y_id+offsetY_src) + x_size_src*y_size_src*z_id;
    size_t ID_dst = (z_id+offsetX_dst) + x_size_dst*(y_id+offsetY_dst) + x_size_dst*y_size_dst*x_id;

    dest[ID_dst] = src[ID_src];
}


__kernel void spinFact(__global float2* w){
    
    unsigned int i = get_global_id(0);
    
    float theta_i = 2.0f*(float)i*(float)PI/(float)FFT_SIZE;
    float2 angle = (float2)(theta_i,theta_i+PI_2);
    
    w[i] = cos(angle);
    
}


void bitReverse_local(__local float2* src, __local float2* dest, int M){
    
    const size_t x_id = get_global_id(0);
    
    unsigned int j = x_id;
    j = (j & 0x55555555) << 1  | (j & 0xAAAAAAAA) >> 1;
    j = (j & 0x33333333) << 2  | (j & 0xCCCCCCCC) >> 2;
    j = (j & 0x0F0F0F0F) << 4  | (j & 0xF0F0F0F0) >> 4;
    j = (j & 0x00FF00FF) << 8  | (j & 0xFF00FF00) >> 8;
    j = (j & 0x0000FFFF) << 16 | (j & 0xFFFF0000) >> 16;
    
    j >>= (32-M);
    
    dest[j] = src[x_id];
}


__kernel void bitReverse(__global float2* src, __global float2* dest, int M){
    
    const size_t x_id = get_global_id(0);
    const size_t y_id = get_global_id(1);
    const size_t z_id = get_global_id(2);
    const size_t x_size = get_global_size(0);
    const size_t y_size = get_global_size(1);
    
    unsigned int j = x_id;
    j = (j & 0x55555555) << 1  | (j & 0xAAAAAAAA) >> 1;
    j = (j & 0x33333333) << 2  | (j & 0xCCCCCCCC) >> 2;
    j = (j & 0x0F0F0F0F) << 4  | (j & 0xF0F0F0F0) >> 4;
    j = (j & 0x00FF00FF) << 8  | (j & 0xFF00FF00) >> 8;
    j = (j & 0x0000FFFF) << 16 | (j & 0xFFFF0000) >> 16;
        
    j >>= (32-M);
    
    dest[j + x_size*y_id + x_size*y_size*z_id] = src[x_id + x_size*y_id + x_size*y_size*z_id];
    
}


__kernel void bitReverseAndXZ_transpose(__global float2* src, __global float2* dest, int M,
                                        int offsetX_src, int x_size_src,
                                        int offsetX_dst, int x_size_dst,
                                        int offsetY_src, int y_size_src,
                                        int offsetY_dst, int y_size_dst){
    
    const size_t x_id = get_global_id(0);
    const size_t y_id = get_global_id(1);
    const size_t z_id = get_global_id(2);
    
    unsigned int j = z_id + offsetX_dst;
    j = (j & 0x55555555) << 1  | (j & 0xAAAAAAAA) >> 1;
    j = (j & 0x33333333) << 2  | (j & 0xCCCCCCCC) >> 2;
    j = (j & 0x0F0F0F0F) << 4  | (j & 0xF0F0F0F0) >> 4;
    j = (j & 0x00FF00FF) << 8  | (j & 0xFF00FF00) >> 8;
    j = (j & 0x0000FFFF) << 16 | (j & 0xFFFF0000) >> 16;
    
    j >>= (32-M);
    
    size_t ID_src = (x_id+offsetX_src) + x_size_src*(y_id+offsetY_src) + x_size_src*y_size_src*z_id;
    size_t ID_dst = j                  + x_size_dst*(y_id+offsetY_dst) + x_size_dst*y_size_dst*x_id;
    dest[ID_dst] = src[ID_src];
}


void butterfly_local(__local float2* x_fft, __constant float2* w, int iter){
    
    const size_t x_id = get_global_id(0);
    const size_t x_size = get_global_size(0);
    
    int butterflySize      = 1 << (iter-1);
    int butterflyGrpDist   = 1 << iter;
    int butterflyGrpNum    = x_size >> iter;
    int butterflyGrpBase   = (x_id >> (iter-1))*butterflyGrpDist;
    int butterflyGrpOffset = x_id & (butterflySize-1);
    
    size_t a = butterflyGrpBase + butterflyGrpOffset;
    size_t b = a + butterflySize;
    
    size_t l = butterflyGrpNum*butterflyGrpOffset;
    
    float2 val1 = x_fft[a];
    float2 val2 = x_fft[b];
    
    float2 vala = val1 + cmplxMult(val2,w[l]);
    float2 valb = val1 - cmplxMult(val2,w[l]);
    
    x_fft[a] = vala;
    x_fft[b] = valb;
    
}

__kernel void butterfly(__global float2* x_fft, __constant float2* w, uint iter, uint flag){
    
    const size_t x_id = get_global_id(0);
    const size_t y_id = get_global_id(1);
    const size_t z_id = get_global_id(2);
    //const size_t x_size = get_global_size(0);  //caution! x_size is usually FFT_size/2
    const size_t y_size = get_global_size(1);
    
    
    uint butterflySize      = 1 << (iter-1);
    uint butterflyGrpDist   = 1 << iter;
    uint butterflyGrpNum    = FFT_SIZE >> iter;
    uint butterflyGrpBase   = (x_id >> (iter-1))*butterflyGrpDist;
    uint butterflyGrpOffset = x_id & (butterflySize-1);
    
    uint a = butterflyGrpBase + butterflyGrpOffset + FFT_SIZE*y_id + FFT_SIZE*y_size*z_id;
    uint b = a + butterflySize;
    
    size_t l = butterflyGrpNum*butterflyGrpOffset;
    
    float2 val1 = x_fft[a];
    float2 val2 = x_fft[b];
    
    //FFT(flag=0x00000000)
    //IFFT(flag=0x80000000)
    float2 wval = (flag==0x0) ? cmplxMult(w[l],val2):cmplxMultConj(w[l],val2);
    float2 vala = val1 + wval;
    float2 valb = val1 - wval;
    
    x_fft[a] = vala;
    x_fft[b] = valb;
}


void FFTnorm_local(__local float2* x_fft,float xgrid){
    const size_t x_id = get_global_id(0);
    
    x_fft[x_id] *= xgrid/sqrt((float)PI);
    
}


__kernel void FFTnorm(__global float2* x_fft,float xgrid){
    const size_t x_id = get_global_id(0);
    const size_t y_id = get_global_id(1);
    const size_t z_id = get_global_id(2);
    const size_t x_size = get_global_size(0);
    const size_t y_size = get_global_size(1);
    const size_t xyz_id = x_id + x_size*y_id +  x_size*y_size*z_id;
    
    float2 val = x_fft[xyz_id]*xgrid/sqrt((float)PI);
    barrier(CLK_GLOBAL_MEM_FENCE);
    x_fft[xyz_id] = val;

}

void IFFTnorm_local(__local float2* x_ifft,float xgrid){
    const size_t x_id = get_global_id(0);
    
    x_ifft[x_id] *= sqrt((float)PI)/xgrid/FFT_SIZE;
}

__kernel void IFFTnorm(__global float2* x_ifft,float xgrid){
    const size_t x_id = get_global_id(0);
    const size_t y_id = get_global_id(1);
    const size_t z_id = get_global_id(2);
    const size_t x_size = get_global_size(0);
    const size_t y_size = get_global_size(1);
    const size_t xyz_id = x_id + x_size*y_id +  x_size*y_size*z_id;
    
    x_ifft[xyz_id] *= sqrt((float)PI)/xgrid/FFT_SIZE;
}
