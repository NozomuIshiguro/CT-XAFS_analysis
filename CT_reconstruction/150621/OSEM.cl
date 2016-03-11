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

#ifndef PRJ_IMAGESIZE
#define PRJ_IMAGESIZE 2048
#endif

#ifndef PRJ_ANGLESIZE
#define PRJ_ANGLESIZE 160
#endif

__constant sampler_t s_linear = CLK_FILTER_NEAREST|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;

__kernel void OSEM1(__read_only image2d_array_t reconst_img,
                    __read_only image2d_array_t prj_img, __write_only image2d_array_t rprj_img,
                    __constant float *angle, int sub, int ss){
    
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    
    float4 xy;
    int4 X_th;
    int X,Y;
    float aprj, rprj, bprj, img;
    
    xy.z=group_ID;
    
    //projection from assumed image & calculate ratio
    for(int i=0;i<PRJ_IMAGESIZE/localsize;i++){
        X=local_ID+i*localsize;
        for(int th=sub;th<PRJ_ANGLESIZE;th+=ss){
            aprj=0.0;
            X_th = (int4)(X,th,group_ID,0);
            for(int j=0; j<PRJ_IMAGESIZE; j++){
                Y=j;
                
                xy.x =  (X-PRJ_IMAGESIZE/2)*cos(angle[th]*PI/180)+(Y-PRJ_IMAGESIZE/2)*sin(angle[th]*PI/180) + IMAGESIZE_X/2;
                xy.y = -(X-PRJ_IMAGESIZE/2)*sin(angle[th]*PI/180)+(Y-PRJ_IMAGESIZE/2)*cos(angle[th]*PI/180) + IMAGESIZE_Y/2;
                
                aprj += read_imagef(reconst_img,s_linear,xy).x;
            }
            
            rprj = read_imagef(prj_img,s_linear,X_th).x;
            rprj = (aprj<0.0001) ? rprj:rprj/aprj;
            write_imagef(rprj_img, X_th, (float4)(rprj,0,0,1.0));
        }
    }
}

__kernel void OSEM2(__read_only image2d_array_t reconst_img,
                    __write_only image2d_array_t reconst_dest_img,
                    __read_only image2d_array_t rprj_img,
                    __constant float *angle, int sub, int ss){
    
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    
    float4 X_th;
    int X,Y;
    float4 xy_f;
    int4 xy_i;
    float aprj, rprj, bprj;
    float img;
    
    X_th.z = group_ID;
    
    //back projection from ratio
    for(int i=0;i<IMAGESIZE_X/localsize;i++){
        X=local_ID+i*localsize;
        for(int j=0; j<IMAGESIZE_Y; j++){
            Y=j;
            xy_f = (float4)(X,Y,group_ID,0);
            xy_i = (int4)(X,Y,group_ID,0);
            bprj=0.0;
            for(int th=sub;th<PRJ_ANGLESIZE;th+=ss){
                X_th.x =  (X-IMAGESIZE_X/2)*cos(angle[th]*PI/180)-(Y-IMAGESIZE_Y/2)*sin(angle[th]*PI/180) + PRJ_IMAGESIZE/2;
                X_th.y = th;
                
                bprj += read_imagef(rprj_img,s_linear,X_th).x;
            }
            
            //update assumed img
            img = read_imagef(reconst_img,s_linear,xy_f).x*bprj*ss/PRJ_ANGLESIZE;
            write_imagef(reconst_dest_img, xy_i, (float4)(img,0,0,1.0));
        }
    }
}