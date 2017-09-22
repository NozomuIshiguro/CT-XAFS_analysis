#define PI      3.14159265358979323846
#define PI_2    1.57079632679489661923
#define EFF     0.262468426103175

#ifndef FFT_SIZE
#define FFT_SIZE 2048
#endif

#define K_PITCH   0.05
#define R_PITCH   0.03067961575771282340


#ifndef PARA_NUM
#define PARA_NUM 1
#endif

#ifndef PARA_NUM_SQ
#define PARA_NUM_SQ 1
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
extern inline float line(float energy, float* p);
extern inline float Victoreen(float energy, float* p);
extern inline float McMaster(float energy, float* p);


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
    float fp[PARA_NUM];
    for(int i=0;i<PARA_NUM;i++){
        fp[i]=fp_img[ID+i*size_M];
    }
    
    switch(funcmode){
        case 0: //line
            //bkg = fp_img[ID];
            bkg = line(0.0f, fp);
            break;
        
        case 1: //Victoreen
            //bkg = fp_img[ID] + fp_img[ID+size_M]*e3 - fp_img[ID+size_M*2]*e4;
            bkg = Victoreen(0.0f, fp);
            break;
        
        case 2: //McMaster
            bkg = McMaster(0.0f, fp);
            //bkg = fp_img[ID] + fp_img[ID+size_M]*exp(-2.75f*lne);
            break;
    }
    
    bkg_img[ID] = bkg;
}


//estimate Ej image (delta between value of post-edge line @E=E0 and bkg)
__kernel void estimateEJ(__global float* ej_img, __global float* bkg_img, __global float* fp_img){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t size_x = get_global_size(0);
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
    float X;
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


__kernel void redimension_chi2mt(__global float* mt_img, __global float* chi_img,
                                 __constant float* energy, int numPnts,
                                 int koffset, int ksize){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t size_M = size_x*size_y;
    const size_t ID = global_x + global_y*size_x;
    
    
    float e1, e2, e3;
    float chi1, chi2;
    float X;
    
    int en=0;
    e3 = energy[en];
    float e_offset = koffset*K_PITCH;
    e_offset *= e_offset;
    e_offset /= EFF;
    while(e3<=e_offset){
        en++;
        e3 = energy[en];
        if(en==numPnts) break;
    }
    
    //if(ID==0) printf("%f\n",energy[numPnts-1]);
    for(int kn=koffset; kn<koffset+ksize-1; kn++){
        e1 = kn*K_PITCH;
        e1 *= e1;
        e1 /=EFF;
        e2 = (kn+1.0f)*K_PITCH;
        e2 *= e2;
        e2 /=EFF;
        
        chi1 = chi_img[ID+kn*size_M];
        chi2 = chi_img[ID+(kn+1)*size_M];
        
        if(e1<energy[numPnts-1]){
            e3 = energy[en];
            while(e3<=e2){
                X=(e3-e1)/(e2-e1);
                mt_img[ID + en*size_M] = (1.0f-X)*chi1 + X*chi2;
                //if(ID==0) printf("  %d,%f\n", en, e3);
                en++;
                e3 = energy[en];
                if(en==numPnts) break;
            }
        }
        if(en==numPnts) break;
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



//x-dimesion is knot zone space (0<=x<basisOffset are dummy zone)
//y-dimension is k-space
__kernel void Bspline_basis_zero(__global float* basis, int N_ctrlP,
                                 float zoneStart, float zonePitch, int basisOffset){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t offset_y = get_global_offset(1);
    const size_t ID = global_x + (global_y-offset_y)*(basisOffset+N_ctrlP);
    float kval = (float)global_y*K_PITCH;
    
    float basis_p;
    int i = global_x - basisOffset;
    if(i<0){
        basis_p = 0.0f;
    }else if(i<N_ctrlP-1){
        basis_p = (kval>=zoneStart+i*zonePitch && kval<zoneStart+(i+1.0f)*zonePitch) ? 1.0f:0.0f;
    }else if(i==N_ctrlP-1){
        basis_p = (kval==zoneStart+(N_ctrlP-1.0f)*zonePitch) ? 1.0f:0.0f;
    }
    basis[ID] = basis_p;
}

//x-dimesion is knot zone space (0<=x<basisOffset are dummy zone)
//y-dimension is k-space
__kernel void Bspline_basis_updateOrder(__global float* basis_src, __global float* basis_dest,
                                        int N_ctrlP,float zoneStart, float zonePitch,
                                        int basisOffset, int order){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t offset_y = get_global_offset(1);
    const size_t ID1 = global_x + (global_y-offset_y)*(basisOffset+N_ctrlP);
    const size_t ID2 = global_x+1 + (global_y-offset_y)*(basisOffset+N_ctrlP);
    float kval = (float)global_y*K_PITCH;
    
    float basis0;
    float basis1 = basis_src[ID1];
    float basis2 = basis_src[ID2];
    int i = global_x - basisOffset;
    float t1 = zoneStart + fmax(0.0f,i)*zonePitch;
    float t2 = zoneStart + fmin(N_ctrlP-1.0f,fmax(0.0f,i+1.0f+order))*zonePitch;
    
    int di1 = min(N_ctrlP-1,max(0,i+order))   - max(0,i);
    int di2 = min(N_ctrlP-1,max(0,i+order+1)) - min(N_ctrlP-1,max(0,i+1));
    
    /*i < -order
    di1 = 0;
    di2 = 0;
    -order <= i < 0
    di1 = i+order;
    di2 = i+order+1;
    0 <= i < N_ctrlP-order-1
    di1 = order;
    di2 = order;
    N_ctrlP-order-1 <= i <N_ctrlP-1
    di1 = N_ctrlP-1-i;
    di2 = N_ctrlP-2-i;
    i = N_ctrlP-1
    di1 = 0;
    di2 = 0;*/
    
    if(di1==0&&di2==0){
        basis0 = 0.0f;
    }else if(di1==0){
        basis0 = (t2-kval)/di2/zonePitch*basis2;
    }else if(di2==0){
        basis0 = (kval-t1)/di1/zonePitch*basis1 + basis2;
    }else{
        basis0 = (kval-t1)/di1/zonePitch*basis1 + (t2-kval)/di2/zonePitch*basis2;
    }
    basis_dest[ID1] = basis0;
}


//xy-dimensions are image dimension
//z-dimension is k-space
__kernel void Bspline(__global float2* spline, __global float* basis, __global float* ctrlP,
                            int N_ctrlP, int basisOffset, int order){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t ID1 = global_x + global_y*size_x + global_z*size_x*size_y;
    
    float spine_p=0.0f;
    int offset = basisOffset-(int)floor(((float)order+1.0f)/2.0f);
    for(int i=0;i<offset;i++){
        size_t ID2 = global_x + global_y*size_x;
        size_t ID3 = i + global_z*(basisOffset+N_ctrlP);
        spine_p += ctrlP[ID2]*basis[ID3];
    }
    for(int i=offset;i<N_ctrlP+offset;i++){
        size_t ID2 = global_x + global_y*size_x + (i-offset)*size_x*size_y;
        size_t ID3 = i + global_z*(basisOffset+N_ctrlP);
        spine_p += ctrlP[ID2]*basis[ID3];
    }
    for(int i=N_ctrlP+offset;i<N_ctrlP+basisOffset;i++){
        size_t ID2 = global_x + global_y*size_x + (N_ctrlP-1)*size_x*size_y;
        size_t ID3 = i + global_z*(basisOffset+N_ctrlP);
        spine_p += ctrlP[ID2]*basis[ID3];
    }
    
    spline[ID1] = (float2)(0.0f,spine_p);
}


//xy-dimensions are image dimension
//z-dimension is k-space
__kernel void BsplineRemoval(__global float2* chiData_img, __global float2* chiFit_img,
                             __global float* basis, __global float* ctrlP,
                             int N_ctrlP, int basisOffset, int order){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t ID1 = global_x + global_y*size_x + global_z*size_x*size_y;
    
    float spine_p=0.0f;
    int offset = basisOffset-(int)floor(((float)order+1.0f)/2.0f);
    for(int i=0;i<offset;i++){
        size_t ID2 = global_x + global_y*size_x;
        size_t ID3 = i + global_z*(basisOffset+N_ctrlP);
        spine_p += ctrlP[ID2]*basis[ID3];
    }
    for(int i=offset;i<N_ctrlP+offset;i++){
        size_t ID2 = global_x + global_y*size_x + (i-offset)*size_x*size_y;
        size_t ID3 = i + global_z*(basisOffset+N_ctrlP);
        spine_p += ctrlP[ID2]*basis[ID3];
    }
    for(int i=N_ctrlP+offset;i<N_ctrlP+basisOffset;i++){
        size_t ID2 = global_x + global_y*size_x + (N_ctrlP-1)*size_x*size_y;
        size_t ID3 = i + global_z*(basisOffset+N_ctrlP);
        spine_p += ctrlP[ID2]*basis[ID3];
    }
    
    chiFit_img[ID1] = chiData_img[ID1] - (float2)(0.0f,spine_p);
}


__kernel void Jacobian_BsplineRemoval(__global float2* J, int paraID,__global float* basis,
                                      int N_ctrlP, int basisOffset, int order){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t ID1 = global_x + global_y*size_x + global_z*size_x*size_y;
    
    float spine_p=0.0f;
    int offset = basisOffset-(int)floor(((float)order+1.0f)/2.0f);
    if(paraID==0){
        for(int i=0;i<offset;i++){
            size_t ID3 = i + global_z*(basisOffset+N_ctrlP);
            spine_p += basis[ID3];
        }
            
    }else if(paraID==N_ctrlP-1){
        for(int i=N_ctrlP+offset;i<N_ctrlP+basisOffset;i++){
            size_t ID3 = i + global_z*(basisOffset+N_ctrlP);
            spine_p += basis[ID3];
        }
    }
    size_t ID3 = (paraID+offset) + global_z*(basisOffset+N_ctrlP);
    spine_p += basis[ID3];
    
    J[ID1] = -(float2)(0.0f,spine_p);
}


__kernel void kweight(__global float2* chi_img, int kw){
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t offset_z = get_global_offset(2);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t ID1 = global_x + global_y*size_x + (global_z-offset_z)*size_x*size_y;
    
    float weight=1.0f;
    float kval = global_z*K_PITCH;
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
    chi_img[ID1] *= weight;
}


__kernel void initialCtrlP(__global float2* chiData_img, __global float* ctrlP_img,
                           float knotPitch, int ksize){
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t size_M = size_x*size_y;
    const size_t IDxy = global_x + global_y*size_x;
    const size_t IDxyz = IDxy + global_z*size_M;
    
    float k_ctrlP = global_z*knotPitch;
    float k1,k2,X;
    float chi1, chi2;
    for(int kn=0; kn<ksize-1; kn++){
        k1 = kn*K_PITCH;
        k2 = (kn+1.0f)*K_PITCH;
        
        chi1 = chiData_img[IDxy+kn*size_M].y;
        chi2 = chiData_img[IDxy+(kn+1)*size_M].y;
        
        if(k_ctrlP>=k1 && k_ctrlP<=k2){
            X=(k_ctrlP-k1)/(k2-k1);
            ctrlP_img[IDxyz] = (1.0f-X)*chi1 + X*chi2;
        }
    }
    
}
