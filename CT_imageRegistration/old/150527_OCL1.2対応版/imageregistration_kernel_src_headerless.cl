#ifndef IMAGESIZE_X
#define IMAGESIZE_X 2048
#endif

#ifndef IMAGESIZE_Y
#define IMAGESIZE_Y 2048
#endif

#ifndef IMAGESIZE_M
#define IMAGESIZE_M 4194304 //2048*2048
#endif

#ifndef LAMBDA
#define LAMBDA 0.2f
#endif


__constant sampler_t s_linear = CLK_FILTER_NEAREST|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;

//reduction
inline void reduction (__local float *loc_mem, __local float *output,uint repeat)
{
    
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    
    for(size_t i=0;i<repeat;i++){
        
        for(size_t s=localsize/2;s>0;s>>=1){
            if(local_ID<s){
                loc_mem[local_ID+i*localsize]+=loc_mem[local_ID+s+i*localsize];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        if (local_ID==0){
            output[i]=loc_mem[i*localsize];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void dark_subtraction(__global float *dark,__global float *I)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    size_t X,Y,ID;
    
    for(int j=0;j<IMAGESIZE_Y;j++){
        for(int i=0;i<IMAGESIZE_X/localsize;i++){
            X=local_ID+i*localsize;
            Y=j;
            ID=X+IMAGESIZE_X*Y;
            
            I[group_ID*IMAGESIZE_M+ID]=I[group_ID*IMAGESIZE_M+ID]-dark[group_ID*IMAGESIZE_M+ID];
        }
    }
}


__kernel void mt_transform(__global float *I0,__global float *It,__write_only image2d_array_t mt)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    size_t X,Y,ID;
    int4 XYZ;
    float4 mt_f;
    
    for(int j=0;j<IMAGESIZE_Y;j++){
        for(int i=0;i<IMAGESIZE_X/localsize;i++){
            X=local_ID+i*localsize;
            Y=j;
            ID=X+IMAGESIZE_X*Y;
            XYZ=(int4)(X,Y,group_ID,0);
            
            mt_f = (float4)(log(I0[group_ID*IMAGESIZE_M+ID]/It[group_ID*IMAGESIZE_M+ID]),1.0,0,0);
            write_imagef(mt,XYZ,mt_f);
        }
    }
}

__kernel void imageRegistration(__read_only image2d_array_t mt_target,
                                __read_only image2d_array_t mt_sample,
                                __global float *mt_output,
                                __global float *transpara, __global float *transpara_err,
                                __local float *transpara_atE, __local float *delta_transpara,
                                __local float *reductPara, __local float *loc_mem,
                                __local char *merge_list)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    
    
    size_t X,Y;
    size_t ID;
    float4 XY_sample, XY_target;
    float4 pixel_sample, pixel_target;
    float mt_sample_pixel, mt_target_pixel;
    int Mask;
    float chi2;
    
    float4 XY_sample_xplus, XY_sample_xminus;
    float4 pixel_sample_xplus, pixel_sample_xminus;
    float4 XY_sample_yplus, XY_sample_yminus;
    float4 pixel_sample_yplus, pixel_sample_yminus;
    
    float Dimg_Dx, Dimg_Dy;
    float Jacob[NUM_TRANSPARA];
    float delta_Img;
    
    float Fx_new, Fx_old;
    float lambda, rho, delta_rho, nyu;
    float lambdaA, lambdaB, lambdaC;
    float nyuA, nyuB;
    
    float weight=0;
    
    /*loc_mem[local_ID]=0.0f;
     loc_mem[local_ID+localsize]=0.0f;
     reductPara[0]=0.0f;
     reductPara[1]=0.0f;
     for(int j=0;j<IMAGESIZE_Y;j++){
     for(int i=0;i<IMAGESIZE_X/localsize;i++){
     X=local_ID+i*localsize;
     Y=j;
     ID=X+Y*IMAGESIZE_X;
     
     XY_target = (float2)(X,Y);
     //mt_target_pixel = mt_target_N0_p[group_ID][ID];
     mt_target_pixel = mt_target_N0[group_ID*IMAGESIZE_M+ID];
     //mt_sample_pixel = mt_sample_N0_p[group_ID][ID];
     mt_sample_pixel = mt_sample_N0[group_ID*IMAGESIZE_M+ID];
     
     loc_mem[local_ID]+=mt_target_pixel;
     loc_mem[local_ID+localsize]+=mt_sample_pixel;
	    }
     }
     reduction (loc_mem, reductPara,2);
     weight=reductPara[1]/reductPara[0];*/
    
    
    //copy data from global transpara to local transpara_atE
    TRANSPARA_LOC_COPY(transpara_atE,transpara,local_ID,group_ID);
    
    for(int i=0;i<IMAGESIZE_X/localsize;i++){
        merge_list[local_ID+i*localsize]=0;
    }
    
    // mrege loop
    lambda=LAMBDA;
    for(size_t mergesize=8;mergesize>0;mergesize>>=1){
        
        if(local_ID==0){
            for(int i=0;i<IMAGESIZE_X;i+=mergesize){
                merge_list[i]=1;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        nyu=2.0f;
        rho=0.0f;
        delta_rho=1.0f;
        
        //initial Fx_old
        loc_mem[local_ID]=0.0f;
        reductPara[0]=0.0f;
        for(int j=0;j<IMAGESIZE_Y;j+=mergesize){
            for(int i=0;i<IMAGESIZE_X;i+=localsize){
                X=local_ID+i;
                Y=j;
                ID=X+Y*IMAGESIZE_X;
                
                XY_target = (float4)(X,Y,group_ID,0);
                pixel_target = read_imagef(mt_target,s_linear,XY_target);
                
                TRANS_XY(XY_sample,transpara_atE,X,Y,group_ID);
                pixel_sample = read_imagef(mt_sample,s_linear,XY_sample);
                
                Mask = (int)(pixel_target.y*pixel_sample.y);
                
                mt_sample_pixel=pixel_sample.x*Mask*merge_list[X];
                mt_target_pixel=pixel_target.x*Mask*merge_list[X];
                chi2=(mt_target_pixel-mt_sample_pixel)*(mt_target_pixel-mt_sample_pixel);
                
                loc_mem[local_ID]+=chi2;
            }
        }
        reduction (loc_mem, reductPara,1);
        Fx_old=reductPara[0];
        
        //trial loop (3 cycle)
        for(size_t trial=0;trial<3;trial++){
            LOCMEM_RESET(loc_mem,local_ID,localsize);
            REDUCTPARA_RESET(reductPara);
            
            //copy jacobian & delta_Img
            for(int j=0;j<IMAGESIZE_Y;j+=mergesize){
                for(int i=0;i<IMAGESIZE_X;i+=localsize){
                    X=local_ID+i;
                    Y=j;
                    ID=X+Y*IMAGESIZE_X;
                    
                    XY_target = (float4)(X,Y,group_ID,0);
                    pixel_target = read_imagef(mt_target,s_linear,XY_target);
                    
                    
                    TRANS_XY(XY_sample,transpara_atE,X,Y,group_ID);
                    pixel_sample = read_imagef(mt_sample,s_linear,XY_sample);
                    
                    Mask = (int)(pixel_target.y*pixel_sample.y);
                    
                    //Partial differential
                    if(X<mergesize){
                        TRANS_XY(XY_sample_xplus,transpara_atE,X+mergesize,Y,group_ID);
                        pixel_sample_xplus = read_imagef(mt_sample,s_linear,XY_sample_xplus);
                        
                        TRANS_XY(XY_sample_xminus,transpara_atE,X,Y,group_ID);
                        pixel_sample_xminus = read_imagef(mt_sample,s_linear,XY_sample_xminus);
                        
                    } else if(X>IMAGESIZE_X-mergesize) {
                        TRANS_XY(XY_sample_xplus,transpara_atE,X,Y,group_ID);
                        pixel_sample_xplus = read_imagef(mt_sample,s_linear,XY_sample_xplus);
                        
                        TRANS_XY(XY_sample_xminus,transpara_atE,X-mergesize,Y,group_ID);
                        pixel_sample_xminus = read_imagef(mt_sample,s_linear,XY_sample_xminus);
                    }else{
                        TRANS_XY(XY_sample_xplus,transpara_atE,X+mergesize,Y,group_ID);
                        pixel_sample_xplus = read_imagef(mt_sample,s_linear,XY_sample_xplus)/2;
                        
                        TRANS_XY(XY_sample_xminus,transpara_atE,X-mergesize,Y,group_ID);
                        pixel_sample_xminus = read_imagef(mt_sample,s_linear,XY_sample_xminus)/2;
                    }
                    Dimg_Dx = pixel_sample_xplus.x - pixel_sample_xminus.x;
                    if(Y<mergesize){
                        TRANS_XY(XY_sample_yplus,transpara_atE,X,Y+mergesize,group_ID);
                        pixel_sample_yplus = read_imagef(mt_sample,s_linear,XY_sample_yplus);
                        
                        TRANS_XY(XY_sample_yminus,transpara_atE,X,Y,group_ID);
                        pixel_sample_yminus = read_imagef(mt_sample,s_linear,XY_sample_yminus);
                    } else if(Y>IMAGESIZE_Y-mergesize) {
                        TRANS_XY(XY_sample_yplus,transpara_atE,X,Y,group_ID);
                        pixel_sample_yplus = read_imagef(mt_sample,s_linear,XY_sample_yplus);
                        
                        TRANS_XY(XY_sample_yminus,transpara_atE,X,Y-mergesize,group_ID);
                        pixel_sample_yminus = read_imagef(mt_sample,s_linear,XY_sample_yminus);
                    }else{
                        TRANS_XY(XY_sample_yplus,transpara_atE,X,Y+mergesize,group_ID);
                        pixel_sample_yplus = read_imagef(mt_sample,s_linear,XY_sample_yplus)/2;
                        
                        TRANS_XY(XY_sample_yminus,transpara_atE,X,Y-mergesize,group_ID);
                        pixel_sample_yminus = read_imagef(mt_sample,s_linear,XY_sample_yminus)/2;
                    }
                    Dimg_Dy = pixel_sample_yplus.x - pixel_sample_yminus.x;
                    barrier(CLK_LOCAL_MEM_FENCE);
                    
                    JACOBIAN(transpara_atE,X,Y,Jacob,Dimg_Dx,Dimg_Dy,Mask,1,merge_list[X]);
                    delta_Img = (pixel_target.x-pixel_sample.x)*Mask*merge_list[X];
                    mt_sample_pixel=pixel_sample.x*Mask*merge_list[X];
                    mt_target_pixel=pixel_target.x*Mask*merge_list[X];
                    chi2=(mt_target_pixel-mt_sample_pixel)*(mt_target_pixel-mt_sample_pixel);
                    
                    LOCMEM_COPY(loc_mem,Jacob,delta_Img,chi2,local_ID,localsize);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            //reduction
            REDUCTION(loc_mem,reductPara);
            
            //update dumping parameter (lamda)
            Fx_new = reductPara[NUM_REDUCTION-1];
            barrier(CLK_LOCAL_MEM_FENCE);
            
            //update transpara
            CALC_DELTA_TRANSPARA(reductPara, lambda, delta_transpara, delta_rho, local_ID);
            
            rho = (Fx_old-Fx_new)/delta_rho;
            lambdaA = 1.0f-(2.0f*rho-1.0f)*(2.0f*rho-1.0f)*(2.0f*rho-1.0f);
            lambdaB = lambda*max(0.333f,lambdaA);
            lambdaC = lambda*nyu;
            nyuA=2.0f;
            nyuB=nyu*2.0f;
            
            if(rho>0.0f){
                lambda = lambdaB;
                nyu=nyuA;
            }else{
                lambda = lambdaC;
                nyu=nyuB;
            }
            Fx_old = Fx_new;
            
            UPDATE_TRANSPARA(delta_transpara,transpara_atE,mergesize,local_ID);
        }
    }
    
    //output registrated mt image
    for(int j=0;j<IMAGESIZE_Y;j++){
        for(int i=0;i<IMAGESIZE_X/localsize;i++){
            X=local_ID+i*localsize;
            Y=j;
            
            TRANS_XY(XY_sample,transpara_atE,X,Y,group_ID);
            pixel_sample = read_imagef(mt_sample,s_linear,XY_sample);
            mt_output[group_ID*IMAGESIZE_M+X+Y*IMAGESIZE_X] = pixel_sample.x;
        }
    }
    
    //copy data from local transpara_atE to global transpara
    TRANSPARA_GLOB_COPY(transpara_atE, transpara, local_ID, group_ID);
    TRANSPARA_ERR_GLOB_COPY(reductPara, transpara_err, local_ID, group_ID);
    
}