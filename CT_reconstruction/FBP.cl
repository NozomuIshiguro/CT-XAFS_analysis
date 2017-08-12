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

extern __constant sampler_t s_linear;
extern __constant sampler_t s_nearest;



//1. calc spinfactor
__kernel void spinFactor(__global float2 *W){
    int ID = get_global_id(0);
    float d = 2.0*PI/ZP_SIZE;
    
    float2 ang =  (float2)(d*ID,d*ID+PI_2);
    W[ID]=cos(ang);  //(cos(ang),-sin(ang))
}


//2. zero padding
__kernel void zeroPadding(__read_only image2d_array_t prj_img, global float2 *xc){
    
    const int X = get_global_id(0);
    const int th = get_global_id(1);
    const int Z = get_global_id(2);
    const size_t imgOffset = th*ZP_SIZE+Z*ZP_SIZE*PRJ_ANGLESIZE;
    
    //zero padding and swap data
    float4 XthZ_f =(float4)(X,th,Z,0.0f);
    int Xcnt = X - PRJ_IMAGESIZE / 2 + ZP_SIZE / 2;
    int Xswap = (Xcnt < ZP_SIZE / 2) ? (Xcnt + ZP_SIZE / 2) : (Xcnt - ZP_SIZE / 2);
    float prj = read_imagef(prj_img, s_linear, XthZ_f).x;
    xc[Xswap + imgOffset] = (float2)(prj, 0.0f);
}

//3. bit reverse
__kernel void bitReverse(__global float2 *xc_src, __global float2 *xc_dest, const uint iter){
    
    const uint X1 = get_global_id(0);;
    const int th = get_global_id(1);
    const int Z = get_global_id(2);
    const size_t imgOffset = th*ZP_SIZE+Z*ZP_SIZE*PRJ_ANGLESIZE;
    uint X2;
    
    X2 = X1;
        
    X2 = (X2 & 0x55555555) << 1  | (X2 & 0xAAAAAAAA) >> 1;
    X2 = (X2 & 0x33333333) << 2  | (X2 & 0xCCCCCCCC) >> 2;
    X2 = (X2 & 0x0F0F0F0F) << 4  | (X2 & 0xF0F0F0F0) >> 4;
    X2 = (X2 & 0x00FF00FF) << 8  | (X2 & 0xFF00FF00) >> 8;
    X2 = (X2 & 0x0000FFFF) << 16 | (X2 & 0xFFFF0000) >> 16;
        
    X2 >>= (32-iter);
        
    xc_dest[X2+imgOffset] = xc_src[X1+imgOffset];
}

// 4. butterfly
__kernel void butterfly(__global float2 *xc, __constant float2 *W, uint flag, int iter){
    const int X = get_global_id(0);
    const int th = get_global_id(1);
    const int Z = get_global_id(2);
    const size_t imgOffset = th*ZP_SIZE+Z*ZP_SIZE*PRJ_ANGLESIZE;
    
    uint bf_size, bf_GpDist, bf_GpNum, bf_GpBase, bf_GpOffset, a, b, l;
	float2 xa, xb, xbxx, xbyy, Wab, Wayx, Wbyx, res_a, res_b;


    bf_size = 1 << (iter - 1);
    bf_GpDist = 1 << iter;
    bf_GpNum = ZP_SIZE >> iter;
    bf_GpBase = (X >> (iter - 1))*bf_GpDist;
    bf_GpOffset = X & (bf_size - 1);

    a = bf_GpBase + bf_GpOffset;
    b = a + bf_size;
    l = bf_GpNum*bf_GpOffset;

    xa = xc[a + imgOffset];
    xb = xc[b + imgOffset];

    xbxx = xb.xx;
    xbyy = xb.yy;

    //FFT(flag=0x00000000)
    //IFFT(flag=0x80000000)
    Wab = as_float2(as_uint2(W[l]) ^ (uint2)(0x0, flag));
    Wayx = as_float2(as_uint2(Wab.yx) ^ (uint2)(0x80000000, 0x0));
    Wbyx = as_float2(as_uint2(Wab.yx) ^ (uint2)(0x0, 0x80000000));

    res_a = xa + xbxx*Wab + xbyy*Wayx;
    res_b = xa - xbxx*Wab + xbyy*Wbyx;
            
    xc[a+imgOffset] = res_a;
    xc[b+imgOffset] = res_b;
}



//5. filtering process
__kernel void filtering(__global float2 *xc){
    const int X = get_global_id(0);
    const int th = get_global_id(1);
    const int Z = get_global_id(2);
    const size_t imgOffset = th*ZP_SIZE+Z*ZP_SIZE*PRJ_ANGLESIZE;
    
    //filter
    const float	h = PI/ZP_SIZE;
    xc[X+imgOffset] *= (X<ZP_SIZE/2) ? (X*h):((ZP_SIZE-X)*h);
}


//6.normalization for IFFT
__kernel void normalization(__global float2 *xc){
    const int X = get_global_id(0);
    const int th = get_global_id(1);
    const int Z = get_global_id(2);
    const size_t imgOffset = th*ZP_SIZE+Z*ZP_SIZE*PRJ_ANGLESIZE;
        
    xc[X+imgOffset] /= ZP_SIZE;
}


//7. swap data and output to image object
__kernel void outputImage(__global float2 *xc, __write_only image2d_array_t fprj_img){
    const int X = get_global_id(0);
    const int th = get_global_id(1);
    const int Z = get_global_id(2);
	const size_t imgOffset = th*ZP_SIZE + Z*ZP_SIZE*PRJ_ANGLESIZE;
    
    int4 XthZ_i = (int4)(X,th,Z,0);
    int Xcnt = X - PRJ_IMAGESIZE / 2 + ZP_SIZE / 2;
    int Xswap = (Xcnt < ZP_SIZE / 2) ? (Xcnt + ZP_SIZE / 2) : (Xcnt - ZP_SIZE / 2);
    float fprj = xc[Xswap+imgOffset].x;
    write_imagef(fprj_img, XthZ_i, (float4)(fprj,0.0f,0.0f,1.0f));
}


//8. back projection of filtered image
__kernel void backProjectionFBP(__read_only image2d_array_t fprj_img,
                                __write_only image2d_array_t reconst_img,
                                __constant float *angle){
    
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    
    float bprj=0.0f;
    int4 xyz_i = (int4)(X,Y,Z,0);
    float4 XthZ;
    XthZ.z = Z;
	int i, th;
	float angle_pr;
    float radius = sqrt((X-IMAGESIZE_X*0.5f)*(X-IMAGESIZE_X*0.5f)+(Y-IMAGESIZE_Y*0.5f)*(Y-IMAGESIZE_Y*0.5f));
    for(th=0;th<PRJ_ANGLESIZE;th++){
        angle_pr = angle[th]*PI/180.0f;
        XthZ.x =  (X-IMAGESIZE_X*0.5f)*cos(angle_pr)-(Y-IMAGESIZE_Y*0.5f)*sin(angle_pr) + PRJ_IMAGESIZE*0.5f;
            
        XthZ.y = th;
        bprj += (radius<IMAGESIZE_X*0.5f) ? read_imagef(fprj_img,s_linear,XthZ).x:0.0f;
    }
    bprj /= PRJ_ANGLESIZE*2.0f;
    write_imagef(reconst_img, xyz_i, (float4)(bprj,0.0f,0.0f,1.0f));
}
