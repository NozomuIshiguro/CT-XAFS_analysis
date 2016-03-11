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

#ifndef NUM_TRIAL
#define NUM_TRIAL 20
#endif

#ifndef LAMBDA
#define LAMBDA 0.001f
#endif

__constant sampler_t s_linear = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;

__constant sampler_t s_linear_clampE = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE;


//reduction
inline float reduction(__local float *loc_mem, const size_t local_ID, const size_t localsize)
{
    for(size_t s=localsize/2;s>0;s>>=1){
        if(local_ID<s){
            loc_mem[local_ID]+=loc_mem[local_ID+s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    float res = loc_mem[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    return res;
}



__kernel void reslice(__read_only image2d_array_t mt_img, __write_only image2d_array_t prj_img,
                      __constant float* xshift, __constant float* yshift,
                      int iter, int n, float baseup){
    
    const size_t global_ID = get_global_id(0);
    
    float4 XYth;
    int4 XthY;
    float4 img;
    
    for(int i=0; i<IMAGESIZE_Y; i++){
        for(int j=0; j<PRJ_ANGLESIZE/iter;j++){
            XYth = (float4)(global_ID+xshift[j+n*PRJ_ANGLESIZE/iter],i+yshift[j+n*PRJ_ANGLESIZE/iter],j,0);
            XthY = (int4)(global_ID,j,i,0);
            
            img = read_imagef(mt_img,s_linear,XYth);
            img.x += baseup;
            write_imagef(prj_img, XthY,img);
        }
    }
}

__kernel void xprojection(__read_only image2d_array_t mt_img,__global float* xproj,
                          int startX, int endX, int iter){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t globalsizex = get_global_size(0);
    const size_t globalsizey = get_global_size(1);
    float4 XYth;
    float4 img;
    size_t ID;
    

    ID = global_x + (global_y + iter*globalsizey)*IMAGESIZE_Y;
    for(int j=startX;j<=endX;j++){
        XYth = (float4)(j,global_x,global_y,0);
        
        img = read_imagef(mt_img,s_linear,XYth);
        xproj[ID] += img.x;
    }
}

__kernel void zcorrection(__read_only image2d_t xprj_img, __global float *yshift,
                          __local float *target_xprj,__local float *loc_mem,
                          int startY, int endY){
    
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    const size_t groupsize = get_num_groups(0);
    
    int Y;
    float2 Yth;
    
    float s_xprj,s_xprj_p1,s_xprj_m1;
    float dF_dy, dF;
    float chi2;
    float tJJ, tJdF;
    
    float chi2_new, chi2_old;
    float lambda, rho, d_rho, nyu;
    float l_A, l_B;
    
    float delta_yshift, yshift_cnd;
    __local float yshift_loc;
    float mask;
    
    //copy target_xproj
    for(int i=0; i<IMAGESIZE_Y/localsize;i++){
        Y=local_ID+i*localsize;
        Yth=(float2)(Y,PRJ_ANGLESIZE/2);
        target_xprj[Y] = read_imagef(xprj_img,s_linear_clampE,Yth).x;
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    
    
    for(int j=0; j<PRJ_ANGLESIZE/groupsize; j++){
        
        //copy local yshift_loc to global yshift
        if(local_ID==0){
            yshift_loc = yshift[group_ID+j*groupsize];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        
        lambda = LAMBDA;
        for(size_t trial=0;trial<NUM_TRIAL;trial++){
            
            //estimate tJJ, tJdF, chi2
            tJJ  = 0.0f;
            tJdF = 0.0f;
            chi2 = 0.0f;
            for(int i=0; i<IMAGESIZE_Y/localsize;i++){
                Y=local_ID+i*localsize;
                Yth=(float2)(Y,group_ID+j*groupsize);
                s_xprj = read_imagef(xprj_img,s_linear_clampE,Yth).x;
                
                Yth.x=Y+yshift_loc-1;
                s_xprj_m1 = read_imagef(xprj_img,s_linear_clampE,Yth).x;
                Yth.x=Y+yshift_loc+1;
                s_xprj_p1 = read_imagef(xprj_img,s_linear_clampE,Yth).x;
                dF_dy = (s_xprj_p1 - s_xprj_m1)/2;
                
                mask = (Y>=startY&Y<=endY) ? 1:0;
                dF   = (target_xprj[Y]-s_xprj);
                
                tJJ  += dF_dy*dF_dy*mask;
                tJdF += dF_dy*dF*mask;
                
                chi2 += dF*dF*mask;
            }
            loc_mem[local_ID]= tJJ;
            barrier(CLK_LOCAL_MEM_FENCE);
            tJJ = reduction(loc_mem,local_ID,localsize);
            loc_mem[local_ID]= tJdF;
            barrier(CLK_LOCAL_MEM_FENCE);
            tJdF = reduction(loc_mem,local_ID,localsize);
            barrier(CLK_LOCAL_MEM_FENCE);
            loc_mem[local_ID] = chi2;
            barrier(CLK_LOCAL_MEM_FENCE);
            chi2_old = reduction(loc_mem,local_ID,localsize);
            barrier(CLK_LOCAL_MEM_FENCE);
            
            
            //estimate chi2_new = SIGMA(f_t(y) - f_s(y,ys_new))
            chi2=0.0f;
            yshift_cnd = yshift_loc + delta_yshift;
            for(int i=0; i<IMAGESIZE_Y/localsize;i++){
                Y=local_ID+i*localsize;
                Yth=(float2)(Y+yshift_cnd,group_ID+j*groupsize);
                float s_xproj = read_imagef(xprj_img,s_linear_clampE,Yth).x;
                
                mask = (Y>=startY&Y<=endY) ? 1:0;
                dF   = (target_xprj[Y]-s_xprj);
                chi2 += dF*dF*mask;
            }
            loc_mem[local_ID] = chi2;
            barrier(CLK_LOCAL_MEM_FENCE);
            chi2_new = reduction(loc_mem,local_ID,localsize);
            barrier(CLK_LOCAL_MEM_FENCE);
            
            
            //update lambda & delta_yshift(if rho>0)
            delta_yshift = tJdF/tJJ/(1+lambda);
            d_rho = delta_yshift*(lambda*delta_yshift+tJdF);
            rho = (chi2_old-chi2_new)/d_rho;
            l_A = 1.0f-(2.0f*rho-1.0f)*(2.0f*rho-1.0f)*(2.0f*rho-1.0f);
            l_B = lambda*max(0.333f,l_A);
            if(rho>0.0f){ //step accetable
                
                lambda = l_B;
                nyu    =  2.0f;
                
                if(local_ID==0){
                    yshift_loc += delta_yshift;
                }
            }else{
                lambda *= nyu;
                nyu    *= 2.0f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        
        //copy local yshift_loc to global yshift
        if(local_ID==0){
            yshift[group_ID+j*groupsize] = yshift_loc;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}