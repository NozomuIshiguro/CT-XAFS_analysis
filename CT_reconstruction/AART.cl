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


//projection of trial image and calculate delta between projected image data
__kernel void AART1(__read_only image2d_array_t reconst_img,
                    __read_only image2d_array_t prj_img, __write_only image2d_array_t dprj_img,
                    __constant float *angle, int sub){
    
    const int X = get_global_id(0);
    const int th = sub + get_global_id(1)*SS;
    const int Z = get_global_id(2);
    
    float4 xyz;
    xyz.z=Z;
    float prj=0.0f, dprj;
    float angle_pr;
    int Y;
    
    //projection from assumed image & calculate delta
    // reconst_img: lambda(k)
    // prj_img:     y_i
    // dprj_img:    y(k) = C x lambda(k) => y'(k) = y_i - y(k)
    int4 XthZ = (int4)(X,th,Z,0);
    for(Y=0; Y < DEPTHSIZE; Y++){
        angle_pr = angle[th]*PI/180.0f;
        xyz.x =  (X-PRJ_IMAGESIZE*0.5f)*cos(angle_pr)+(Y-DEPTHSIZE*0.5f)*sin(angle_pr) + IMAGESIZE_X*0.5f;
        xyz.y = -(X-PRJ_IMAGESIZE*0.5f)*sin(angle_pr)+(Y-DEPTHSIZE*0.5f)*cos(angle_pr) + IMAGESIZE_Y*0.5f;
        
        prj += read_imagef(reconst_img,s_linear,xyz).x;
    }
    
    dprj = read_imagef(prj_img,s_linear,XthZ).x;
    dprj = dprj - prj;
    write_imagef(dprj_img, XthZ, (float4)(dprj,0.0f,0.0f,1.0f));
}


//back-projection of projection delta
__kernel void AART2(__read_only image2d_array_t reconst_img,
                    __write_only image2d_array_t reconst_dest_img,
                    __read_only image2d_array_t dprj_img,
                    __constant float *angle, int sub, float alpha){
    
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    
    float4 XthZ;
    XthZ.z = Z;
    float bprj=0.0f;
    float img;
    int th;
    float angle_pr;
    
    //back projection from delta
    // reconst_img:         lambda(k)
    // dprj_img:            y'(k)
    // bprj:                lambda' = y'(k) x C_pinv
    // reconst_dest_img:    lambda(k+1) = lambda' x lambda(k)
    float4 xyz_f = (float4)(X,Y,Z,0.0f);
    int4 xyz_i = (int4)(X,Y,Z,0);
    for(th=sub;th<PRJ_ANGLESIZE;th+=SS){
        angle_pr = angle[th]*PI/180.0f;
        XthZ.x =  (X-IMAGESIZE_X*0.5f)*cos(angle_pr)-(Y-IMAGESIZE_Y*0.5f)*sin(angle_pr) + PRJ_IMAGESIZE*0.5f;
        XthZ.y = th;
        
        bprj += read_imagef(dprj_img,s_nearest,XthZ).x;
    }
    
    //update assumed img
    img = read_imagef(reconst_img,s_nearest,xyz_f).x + bprj*alpha*SS/PRJ_ANGLESIZE/DEPTHSIZE;
    write_imagef(reconst_dest_img, xyz_i, (float4)(img,0.0f,0.0f,1.0f));
}
