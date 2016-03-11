#define PI 3.14159265358979323846
#define PI_2 1.57079632679489661923

#ifndef IMAGESIZE_X
#define IMAGESIZE_X 2048
#endif

#ifndef IMAGESIZE_Y
#define IMAGESIZE_Y 2048
#endif

#ifndef IMAGESIZE_M
#define IMAGESIZE_M 4194304 //2048*2048
#endif

#ifndef PRJ_IMAGESIZE
#define PRJ_IMAGESIZE 2048
#endif

#ifndef PRJ_ANGLESIZE
#define PRJ_ANGLESIZE 160
#endif

#ifndef ZP_SIZE
#define ZP_SIZE 4096
#endif


__constant sampler_t s_linear = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;

inline void zeroPadding(__read_only image2d_array_t prj_img, __local float *prz, int th,
                        const size_t local_ID, const size_t localsize, const size_t group_ID){
    
    int X;
    float4 X_th;
    X_th.y=th;
    X_th.z=group_ID;
    
    //prz initialization
    for(int i=0;i<ZP_SIZE/localsize;i++){
        X=local_ID+i*localsize;
        prz[X]=0;
    }
    
    for(int i=0;i<PRJ_IMAGESIZE/localsize;i++){
        X=local_ID+i*localsize;
        X_th.x=X;
        
        prz[X-PRJ_IMAGESIZE/2+ZP_SIZE/2]=read_imagef(prj_img,s_linear,X_th).x;
    }
    
}

inline void spinFact(__local float2 *W,const size_t local_ID, const size_t localsize){
    int X;
    float d = 2*PI/ZP_SIZE;
    float2 ang;
    
    //prz initialization
    for(int i=0;i<ZP_SIZE/2/localsize;i++){
        X=local_ID+i*localsize;
        ang =  (float2)(d*i,d*i+PI_2);
        
        W[X]=cos(ang);  //(cos(ang),-sin(ang))
    }
}

inline void bitReverse(__local float2 *xc,const size_t local_ID, const size_t localsize,
                       const uint iter){
    
    uint X1, X2;
    float2 val;
    
    for(int i=0;i<ZP_SIZE/2/localsize;i++){
        X1=local_ID+i*localsize;
        X2 = X1;
        
        X2 = (X2 & 0x55555555) << 1  | (X2 & 0xAAAAAAAA) >> 1;
        X2 = (X2 & 0x33333333) << 2  | (X2 & 0xCCCCCCCC) >> 2;
        X2 = (X2 & 0x0F0F0F0F) << 4  | (X2 & 0xF0F0F0F0) >> 4;
        X2 = (X2 & 0x00FF00FF) << 8  | (X2 & 0xFF00FF00) >> 8;
        X2 = (X2 & 0x0000FFFF) << 16 | (X2 & 0xFFFF0000) >> 16;
        
        X2 >>= (32-iter);
        
        val = xc[X1];
        //barrier(CLK_LOCAL_MEM_FENCE);
        xc[X1] = xc[X2];
        xc[X2] = val;
    }
}

inline void Butterfly(__local float2 *xc,__local float2 *W,
                      const size_t local_ID, const size_t localsize,uint iter,int2 flag){
    
    int X;
    uint bf_size, bf_GpDist, bf_GpNum, bf_GpBase, bf_GpOffset, a, b, l;
    float2 xc_a, xc_b, xc_bxx, xc_byy, Wab_xy, Wab_yx, res_a, res_b;
    
    for(int j=0;j<iter;j++){
        for (int i=0;i<ZP_SIZE/2/localsize;i++) {
            X=local_ID+i*localsize;
            
            bf_size     = 1 << (iter-1);
            bf_GpDist   = 1 << iter;
            bf_GpNum    = ZP_SIZE >> iter;
            bf_GpBase   = (X >> (iter-1))*bf_GpDist;
            bf_GpOffset = X & (bf_size-1);
            
            a = bf_GpBase + bf_GpOffset;
            b = a + bf_size;
            l = bf_GpNum*bf_GpOffset;
            
            xc_a = xc[a];
            xc_b = xc[b];
            
            xc_bxx = xc_b.xx;
            xc_byy = xc_b.yy;
            
            //FFT(flag=(1,1)):(cos(2*pi/N*l),-sin(2*pi/N*l))
            //IFFT(flag=(1,-1)):(cos(2*pi/N*l),sin(2*pi/N*l))
            Wab_xy = W[l]*flag;
            Wab_yx = Wab_xy.yx;
            
            res_a = xc_a + xc_bxx*Wab_xy + xc_byy*Wab_yx;
            res_b = xc_a - xc_bxx*Wab_xy - xc_byy*Wab_yx;
            
            xc[a] = res_a;
            xc[b] = res_b;
            
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}

inline void Normalization(__local float2 *xc,const size_t local_ID, const size_t localsize){
    int X;
    
    for (int i=0;i<ZP_SIZE/localsize;i++) {
        X=local_ID+i*localsize;
        
        xc[X] /= ZP_SIZE;
    }
}


inline void FFT(__local float2 *xc,__local float2 *W,
                const size_t local_ID, const size_t localsize,uint iter){
    
    //bit reverse
    bitReverse(xc,local_ID,localsize,iter);
    
    //butterfly calculation
    Butterfly(xc,W,local_ID,localsize,iter,(int2)(1,1));
    
}

inline void IFFT(__local float2 *xc,__local float2 *W,
                 const size_t local_ID, const size_t localsize,uint iter){
    
    //bit reverse
    bitReverse(xc,local_ID,localsize,iter);
    
    //butterfly calculation
    Butterfly(xc,W,local_ID,localsize,iter,(int2)(1,-1));
    
    //normalization
    Normalization(xc,local_ID,localsize);
}

inline void Filter(__local float2 *xc,const size_t local_ID, const size_t localsize){
    
    float	h = PI/ZP_SIZE;
    int X;
    
    for (int i=0;i<ZP_SIZE/2/localsize;i++){
        X=local_ID+i*localsize;
        
        xc[X] *= (X * h);
        xc[X+ZP_SIZE/2] *= ((ZP_SIZE/2-X) * h);
    }
}


inline void write_bprj(__write_only image2d_array_t bprj_img, __local float *prz, int th,
                        const size_t local_ID, const size_t localsize, const size_t group_ID){
    
    int X;
    int4 X_th;
    X_th.y=th;
    X_th.z=group_ID;
    float bprj;
    
    //prz initialization
    for(int i=0;i<ZP_SIZE/localsize;i++){
        X=local_ID+i*localsize;
        prz[X]=0;
    }
    
    for(int i=0;i<IMAGESIZE_X/localsize;i++){
        X=local_ID+i*localsize;
        X_th.x=X;
        
        bprj=prz[X-PRJ_IMAGESIZE/2+ZP_SIZE/2];
        write_imagef(bprj_img, X_th, (float4)(bprj,0,0,1.0));
    }
    
}


__kernel void FBP1(__read_only image2d_array_t prj_img,__write_only image2d_array_t bprj_img,
                   __local float *prz,__local float2 *xc,__local float2 *W){
    
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    
    int X;
    uint iter = 32 - clz(ZP_SIZE);
    
    //initialize FFT parameters
    spinFact(W,local_ID,localsize);
    
    for(int th=0;th<PRJ_ANGLESIZE;th++){
        zeroPadding(prj_img,prz,th,local_ID,localsize,group_ID);
        
        for (int i=0;i<ZP_SIZE/2/localsize;i++) {
            X=local_ID+i*localsize;
            
            xc[X] = (float2)(prz[X+ZP_SIZE/2],0);
            xc[X+ZP_SIZE/2] = (float2)(prz[X],0);
        }
        
        FFT(xc,W,local_ID,localsize,iter);
        Filter(xc,local_ID,localsize);
        IFFT(xc,W,local_ID,localsize,iter);
        
        for (int i=0;i<ZP_SIZE/2/localsize;i++) {
            X=local_ID+i*localsize;
            
            prz[X-ZP_SIZE/2+PRJ_IMAGESIZE/2] = xc[X+ZP_SIZE/2].x;
            prz[X+PRJ_IMAGESIZE/2] = xc[X].x;
        }
        
        write_bprj(bprj_img,prz,th,local_ID,localsize,group_ID);
    }
}

__kernel void FBP2(__read_only image2d_array_t bprj_img,
                   __write_only image2d_array_t reconst_img,
                   __constant float *angle){
    
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    
    float bprj;
    int X,Y;
    int4 xy_i;
    float4 X_th;
    X_th.z = group_ID;
    
    for(int i=0;i<IMAGESIZE_X/localsize;i++){
        X=local_ID+i*localsize;
        for(int j=0; j<IMAGESIZE_Y; j++){
            Y=j;
            xy_i = (int4)(X,Y,group_ID,0);
            bprj=0.0;
            for(int th=0;th<PRJ_ANGLESIZE;th++){
                X_th.x =  (X-IMAGESIZE_X/2)*cos(angle[th]*PI/180)-(Y-IMAGESIZE_Y/2)*sin(angle[th]*PI/180) + PRJ_IMAGESIZE/2;
                X_th.y = th;
                
                bprj += read_imagef(bprj_img,s_linear,X_th).x;
            }
            
            write_imagef(reconst_img, xy_i, (float4)(bprj,0,0,1.0));
        }
    }
}