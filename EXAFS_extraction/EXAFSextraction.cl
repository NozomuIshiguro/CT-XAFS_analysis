#define PI      3.14159265358979323846
#define PI_2    1.57079632679489661923
#define EFF     0.262468426103175

#define FFT_SIZE 2048
#define LOG_FFT_SIZE 11

#define K_PITCH   0.05
#define R_PITCH   0.03067961575771282340

#ifndef N_KNOT
#define N_KNOT 13
#endif

extern float2 cmplxMult(float2 A, float2 B);
extern float2 cmplxMultConj(float2 Aconj, float2 B);
extern float cmplxAbs(float2 A);
extern float cmplxAbs2(float2 A);
extern float2 cmplxSqrt(float2 A);
extern float2 cmplxExp(float2 A);
extern float2 cmplxConj(float2 A);
extern void bitReverse_local(__local float2* src, __local float2* dest, int M);
extern void butterfly_local(__local float2* x_fft, __constant float2* w, int iter);
extern void FFTnorm_local(__local float2* x_fft,float xgrid);


//estimate bkg image (value of pre-edge line @E=E0)
__kernel void estimateBkg(__global float* bkg_img, __global float* fp_img,
                          int funcmode, float E0){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t size_M = size_x*size_y;
    const size_t ID = global_x + global_y*size_x;
    
    float bkg;
    float e1 = 1.0e3f/E0;
    float e3 = e1*e1*e1;
    float e4 = e3*e1;
    float lne = E0*1.0e-3f;
    lne = log(e1);
    switch(funcmode){
        case 0: //line
            bkg = fp_img[ID];
            break;
        
        case 1: //Victoreen
            bkg = fp_img[ID] + fp_img[ID+size_M]*e3 + fp_img[ID+size_M*2]*e4;
            break;
        
        case 2: //Macmaster
            bkg = fp_img[ID] + fp_img[ID+size_M]*exp(-2.75f*e1);
            break;
    }
    
    bkg_img[ID] = bkg;
}


//estimate Ej image (delta between value of post-edge line @E=E0 and bkg)
__kernel void estimateEJ(__global float* ej_img, __global float* bkg_img, __global float* fp_img){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t size_M = size_x*size_y;
    const size_t ID = global_x + global_y*size_x;
    
    float ej = fp_img[ID] - bkg_img[ID];
    ej_img[ID] = ej;
}


__kernel void redimension_mt2chi(__global float* mt_img, __global float* chi_img,
                                 __constant float* energy, int numPnts,
                                 int koffset, int ksize){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t size_M = size_x*size_y;
    const size_t ID = global_x + global_y*size_x;
    
    
    float k1, k2, k3;
    float mt1, mt2;
    float X,chi;
    int kn = koffset;
    for(int en=0; en<numPnts-1; en++){
        k1 = energy[en]*EFF;
        k1 = sqrt(k1);
        k1 = (energy[en]<0) ? -k1:k1;
        k2 = energy[en+1]*EFF;
        k2 = sqrt(k2);
        k2 = (energy[en+1]<0) ? -k2:k2;
        
        mt1 = mt_img[ID+en*size_M];
        mt2 = mt_img[ID+(en+1)*size_M];
        
        if(k2>=0.0f && k1<20.0f+K_PITCH){
            k3 = kn*K_PITCH;
            while(k3<=k2){
                X=(k3-k1)/(k2-k1);
                chi_img[ID + (kn-koffset)*size_M] = (1.0f-X)*mt1 + X*mt2;
                kn++;
                k3 = kn*K_PITCH;
                if(kn==koffset+ksize) break;
            }
        }
        if(kn==koffset+ksize) break;
    }
}


__kernel void convert2realChi(__global float2* chi_cmplx_img, __global float* chi_img){
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t size_M = size_x*size_y;
    const size_t offset_z = get_global_offset(2);
    const size_t ID = global_x + global_y*size_x;
    const size_t IDxyz1 = ID + global_z*size_M;
    const size_t IDxyz2 = ID + (global_z-offset_z)*size_M;
    
    float chi = chi_cmplx_img[IDxyz2].y;
    chi_img[IDxyz1] = chi;
}


//x-dimesion is knot zone number
//y-dimension is k-space
//size_x is set to be Nknot
__kernel void Bspline_basis_zero(__global float* basis, float h){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t size_x = get_global_size(0);
    const size_t ID = global_x + global_y*size_x;
    
    
    float zoneStart = (float)global_x*h;
    float zoneEnd = (float)(global_x+1)*h;
    float kval = (float)global_y*K_PITCH;
    
    basis[ID] = (kval>=zoneStart && kval<zoneEnd) ? 1.0f:0.0f;
}


//x-dimesion is knot zone number
//y-dimension is k-space
//size_x is set to be Nknot-1
//OUM must initialize to 0 before processing
__kernel void Bspline_orderUpdatingMatrix(__global float* OUM, float h, int order){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t size_x = get_global_size(0);
    const size_t ID1 = global_x + global_x*(N_KNOT-1) + global_y*(N_KNOT-1)*(N_KNOT-1);
    const size_t ID2 = (global_x+1) + global_x*(N_KNOT-1) + global_y*(N_KNOT-1)*(N_KNOT-1);
    
    float kval = (float)global_x*K_PITCH;
    float zoneStart = (float)global_y*h;
    float zoneEnd = (float)(global_y+1)*h;
    
    OUM[ID1] = (kval-zoneStart)/order/h;
    OUM[ID2] = (zoneEnd-kval+h)/order/h;
}


//x-dimesion is knot zone number
//y-dimension is k-space
//size_x is set to be Nknot
__kernel void Bspline_basis_updateOrder(__global float* basis_src, __global float* basis_dest,
                                        __global float* OUM){
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t size_x = get_global_size(0);
    
    float basis = 0.0f;
    for(int i=0;i<size_x;i++){
        size_t j = global_x + i;
        j = (j<size_x) ? j:j-size_x;
        size_t ID1 = j + global_y*size_x;
        size_t ID2 = j + global_x*size_x + global_y*size_x*size_x;
        
        basis += OUM[ID2]*basis_src[ID1];
    }
    
    size_t ID3 = global_x + global_y*size_x;
    basis_dest[ID3] = basis;
}

//xy-dimensions are image dimension
//z-dimension is k-space
__kernel void Bspline(__global float* spline, __global float* basis){
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t ID = global_x + global_y*size_x + global_z*size_x*size_y;
}

