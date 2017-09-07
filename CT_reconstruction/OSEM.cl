#ifndef PI
#define PI 3.14159265358979323846
#endif

#ifndef IMAGESIZE_X
#define IMAGESIZE_X 2048
#endif

#ifndef IMAGESIZE_Y
#define IMAGESIZE_Y 2048
#endif

#ifndef DEPTHSIZE
#define DEPTHSIZE 2048
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

#ifndef SS
#define SS 16
#endif

extern __constant sampler_t s_linear;
extern __constant sampler_t s_nearest;


//projection of trial image and calculate ratio to projected image data
__kernel void OSEM1(__read_only image2d_array_t reconst_img,
                    __read_only image2d_array_t prj_img, __write_only image2d_array_t rprj_img,
                    __constant float *angle, int sub){
    
    const int X = get_global_id(0);
    const int Z = get_global_id(2);
    const int th = sub + get_global_id(1)*SS;
    
    float4 xyz;
    xyz.z=Z;
    int4 XthZ= (int4)(X,th,Z,0);
    int Y;
    float prj=0.0f, rprj;
    float angle_pr;
    //projection from assumed image & calculate ratio
    // reconst_img: lambda(k)
    // prj_img:     y_i
    // rprj_img:    y(k) = C x lambda(k) => y'(k) = y_i / y(k)
    for(Y=0; Y < DEPTHSIZE; Y++){
        angle_pr = angle[th]*PI/180;
        xyz.x =  (X-PRJ_IMAGESIZE/2)*cos(angle_pr)+(Y-DEPTHSIZE/2)*sin(angle_pr) + IMAGESIZE_X/2+0.5f;
        xyz.y = -(X-PRJ_IMAGESIZE/2)*sin(angle_pr)+(Y-DEPTHSIZE/2)*cos(angle_pr) + IMAGESIZE_Y/2+0.5f;
        
        prj += read_imagef(reconst_img,s_linear,xyz).x;
    }
    
    rprj = max(1.0e-6f,read_imagef(prj_img,s_linear,XthZ).x);
    rprj = (isnan(rprj)) ? 1.0e-6f:rprj;
    rprj = (prj<0.0001f) ? rprj:rprj/prj;
    write_imagef(rprj_img, XthZ, (float4)(rprj,0.0f,0.0f,1.0f));
}

//back-projection of projection ratio
__kernel void OSEM2(__read_only image2d_array_t reconst_img,
                    __write_only image2d_array_t reconst_dest_img,
                    __read_only image2d_array_t rprj_img,
                    __constant float *angle, int sub){
    
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    
    float4 XthZ;
    XthZ.z = Z;
    float bprj=0.0f;
    float img;
    
    //back projection from ratio
    // reconst_img:         lambda(k)
    // rprj_img:            y'(k)
    // bprj:                lambda' = y'(k) x C_pinv
    // reconst_dest_img:    lambda(k+1) = lambda' x lambda(k)
    float4 xyz_f = (float4)(X+0.5f,Y+0.5f,Z,0.0f);
    int4 xyz_i = (int4)(X,Y,Z,0);
    float angle_pr;
    for(int th=sub;th<PRJ_ANGLESIZE;th+=SS){
        angle_pr = angle[th]*PI/180.0f;
        XthZ.x =  (X-IMAGESIZE_X/2)*cos(angle_pr)-(Y-IMAGESIZE_Y/2)*sin(angle_pr) + PRJ_IMAGESIZE/2+0.5f;
        XthZ.y = th+0.5f;
        
        bprj += read_imagef(rprj_img,s_nearest,XthZ).x;
    }
    
    //update assumed img
    img = read_imagef(reconst_img,s_nearest,xyz_f).x*bprj*SS/PRJ_ANGLESIZE;
    img = isnan(img) ? 1.0e-6f:img;
    write_imagef(reconst_dest_img, xyz_i, (float4)(img,0.0f,0.0f,1.0f));
}
