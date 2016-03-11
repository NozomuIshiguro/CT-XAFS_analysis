#ifndef IMAGESIZE_X
#define IMAGESIZE_X 2048
#endif

#ifndef IMAGESIZE_Y
#define IMAGESIZE_Y 2048
#endif

#ifndef IMAGESIZE_M
#define IMAGESIZE_M 4194304 //2048*2048
#endif


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

//linear sampler
inline float2 mt_img_sampler_nearest(__global float *mt, float2 XY)
{
    size_t leftX;
    size_t rightX;
    if(XY.x<0){
        leftX = 0;
    }else if (XY.x>IMAGESIZE_X-1){
        leftX = IMAGESIZE_X-2;
    }else{
        leftX = floor(XY.x);
    }
    rightX = leftX+1;
    size_t downY;
    size_t upY;
    if(XY.y<0){
        downY = 0;
    }else if (XY.y>IMAGESIZE_Y-1){
        downY = IMAGESIZE_Y-2;
    }else{
        downY = floor(XY.y);
    }
    upY = downY+1;
    
    size_t ID;
    if(XY.x-leftX<rightX-XY.x) ID=leftX;
    else ID=rightX;
    if(XY.y-downY<upY-XY.y) ID+=downY*IMAGESIZE_X;
    else ID+=upY*IMAGESIZE_X;
    
    float x_inout = XY.x*(XY.x-IMAGESIZE_X+1);
    float y_inout = XY.y*(XY.y-IMAGESIZE_Y+1);
    
    float mask=1.0;
    if((x_inout<0)&(y_inout<0)){
        mask*=1.0;
    }else {
        mask*=0.0;
    }
    
    float mt_sampler = mt[ID]*mask;
    
    
    return (float2)(mt_sampler,mask);
}

//linear sampler
inline float2 mt_img_sampler_linear(__global float *mt, float2 XY)
{
    size_t leftX;
    size_t rightX;
    if(XY.x<0){
        leftX = 0;
    }else if (XY.x>IMAGESIZE_X-1){
        leftX = IMAGESIZE_X-2;
    }else{
        leftX = floor(XY.x);
    }
    rightX = leftX+1;
    size_t downY;
    size_t upY;
    if(XY.y<0){
        downY = 0;
    }else if (XY.y>IMAGESIZE_Y-1){
        downY = IMAGESIZE_Y-2;
    }else{
        downY = floor(XY.y);
    }
    upY = downY+1;
    
    size_t ID_dl = leftX+downY*IMAGESIZE_X;
    size_t ID_ul = leftX+upY*IMAGESIZE_X;
    size_t ID_dr = rightX+downY*IMAGESIZE_X;
    size_t ID_ur = rightX+upY*IMAGESIZE_X;
    
    float x_inout = XY.x*(XY.x-IMAGESIZE_X+1);
    float y_inout = XY.y*(XY.y-IMAGESIZE_Y+1);
    
    float mask=1.0;
    if((x_inout<0)&(y_inout<0)){
        mask*=1.0;
    }else {
        mask*=0.0;
    }
    
    
    float mt_dl=mt[ID_dl];
    barrier(CLK_GLOBAL_MEM_FENCE);
    float mt_ul=mt[ID_ul];
    barrier(CLK_GLOBAL_MEM_FENCE);
    float mt_dr=mt[ID_dr];
    barrier(CLK_GLOBAL_MEM_FENCE);
    float mt_ur=mt[ID_ur];
    barrier(CLK_GLOBAL_MEM_FENCE);
    float mt_down = (mt_dr-mt_dl)*(XY.x-leftX)+mt_dl;
    float mt_up = (mt_ur-mt_ul)*(XY.x-leftX)+mt_ul;
    
    float mt_sampler = ((mt_up-mt_down)*(XY.y-downY)+mt_down)*mask;
    
    
    return (float2)(mt_sampler,mask);
}

__kernel void dark_subtraction(DARK_DEF,I_DEF)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    size_t X,Y,ID;
    
    //DARK_P;
    //I_P;
    //barrier(CLK_GLOBAL_MEM_FENCE);
    
    for(int j=0;j<IMAGESIZE_Y;j++){
        for(int i=0;i<IMAGESIZE_X/localsize;i++){
            X=local_ID+i*localsize;
            Y=j;
            ID=X+IMAGESIZE_X*Y;
            
            //I_p[group_ID][ID]=I_p[group_ID][ID]-dark_p[group_ID][ID];
            I_N0[group_ID*IMAGESIZE_M+ID]=I_N0[group_ID*IMAGESIZE_M+ID]-dark_N0[group_ID*IMAGESIZE_M+ID];
        }
    }
}


__kernel void mt_transform(I0_DEF,IT2MT_DEF)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    size_t X,Y,ID;
    
    //I0_P;
    //IT2MT_P;
    //barrier(CLK_GLOBAL_MEM_FENCE);
    
    for(int j=0;j<IMAGESIZE_Y;j++){
        for(int i=0;i<IMAGESIZE_X/localsize;i++){
            X=local_ID+i*localsize;
            Y=j;
            ID=X+IMAGESIZE_X*Y;
            
            //It2mt_p[group_ID][ID] = log(I0_p[group_ID][ID]/It2mt_p[group_ID][ID]);
            It2mt_N0[group_ID*IMAGESIZE_M+ID] = log(I0_N0[group_ID*IMAGESIZE_M+ID]/It2mt_N0[group_ID*IMAGESIZE_M+ID]);
        }
    }
}

__kernel void imageRegistration(MT_TARGET_DEF, MT_SAMPLE_DEF, MT_OUTPUT_DEF,
                                __global float *transpara, __global float *transpara_err,
                                __local float *transpara_atE, __local float *delta_transpara,
                                __local float *reductPara, __local float *loc_mem,
                                __local char *merge_list)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    
    
    
    //MT_TARGET_P;
    //MT_SAMPLE_P;
    //MT_OUTPUT_P;
    //barrier(CLK_GLOBAL_MEM_FENCE);
    
    size_t X,Y;
    size_t ID;
    float2 XY_sample, XY_target;
    float2 pixel_sample, pixel_target;
    float mt_sample_pixel;
    int Mask;
    
    float2 XY_sample_xplus, XY_sample_xminus;
    float2 pixel_sample_xplus, pixel_sample_xminus;
    float2 XY_sample_yplus, XY_sample_yminus;
    float2 pixel_sample_yplus, pixel_sample_yminus;
    
    float Dimg_Dx, Dimg_Dy;
    float Jacob[NUM_TRANSPARA];
    float delta_Img;
    
    float Fx_new, Fx_old;
    float lambda, rho, delta_rho, nyu;
    float lambdaA, lambdaB, lambdaC;
    float nyuA, nyuB;
    
    
    
    
    //copy data from global transpara to local transpara_atE
    TRANSPARA_LOC_COPY(transpara_atE,transpara,local_ID,group_ID);
    
    for(int i=0;i<IMAGESIZE_X/localsize;i++){
        merge_list[local_ID+i*localsize]=0;
    }
    
    // mrege loop
    lambda=0.2f;
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
        for(int j=0;j<IMAGESIZE_Y;j++){
            for(int i=0;i<IMAGESIZE_X/localsize;i+=mergesize){
                X=local_ID+i*localsize;
                Y=j;
                ID=X+Y*IMAGESIZE_X;
                
                XY_target = (float2)(X,Y);
                //pixel_target = (float2)(mt_target_p[group_ID][ID],1);
                pixel_target = (float2)(mt_target_N0[group_ID*IMAGESIZE_M+ID],1);
                
                TRANS_XY(XY_sample,transpara_atE,X,Y);
                //pixel_sample = mt_img_sampler_nearest(mt_sample_p[group_ID],XY_sample);
                pixel_sample = mt_img_sampler_nearest(&mt_sample_N0[group_ID*IMAGESIZE_M],XY_sample);
                
                Mask = (int)(pixel_target.y*pixel_sample.y);
                
                mt_sample_pixel=pixel_sample.x*Mask/mergesize/mergesize*merge_list[X];
                
                loc_mem[local_ID]+=mt_sample_pixel;
                //atom_add_float(&loc_mem[local_ID], mt_sample_pixel);
            }
        }
        reduction (loc_mem, reductPara,1);
        Fx_old=reductPara[0];
        
        //trial loop (3 cycle)
        for(size_t trial=0;trial<3;trial++){
            LOCMEM_RESET(loc_mem,local_ID,localsize);
            REDUCTPARA_RESET(reductPara);
            
            //copy jacobian & delta_Img
            for(int j=0;j<IMAGESIZE_Y;j++){
                for(int i=0;i<IMAGESIZE_X/localsize;i+=mergesize){
                    X=local_ID+i*localsize;
                    Y=j;
                    ID=X+Y*IMAGESIZE_X;
                    
                    XY_target = (float2)(X,Y);
                    //pixel_target = (float2)(mt_target_p[group_ID][ID],1);
                    pixel_target = (float2)(mt_target_N0[group_ID*IMAGESIZE_M+ID],1);
                    
                    
                    TRANS_XY(XY_sample,transpara_atE,X,Y);
                    //pixel_sample = mt_img_sampler_nearest(mt_sample_p[group_ID],XY_sample);
                    pixel_sample = mt_img_sampler_nearest(&mt_sample_N0[group_ID*IMAGESIZE_M],XY_sample);
                    
                    Mask = (int)(pixel_target.y*pixel_sample.y);
                    
                    //Partial differential
                    if(X<mergesize){
                        TRANS_XY(XY_sample_xplus,transpara_atE,X+mergesize,Y);
                        //pixel_sample_xplus = mt_img_sampler_nearest(mt_sample_p[group_ID],XY_sample_xplus);
                        pixel_sample_xplus = mt_img_sampler_nearest(&mt_sample_N0[group_ID*IMAGESIZE_M],XY_sample_xplus);
                        TRANS_XY(XY_sample_xminus,transpara_atE,X,Y);
                        //pixel_sample_xminus = mt_img_sampler_nearest(mt_sample_p[group_ID],XY_sample_xminus);
                        pixel_sample_xminus = mt_img_sampler_nearest(&mt_sample_N0[group_ID*IMAGESIZE_M],XY_sample_xminus);
                        
                    } else if(X>IMAGESIZE_X-mergesize) {
                        TRANS_XY(XY_sample_xplus,transpara_atE,X,Y);
                        //pixel_sample_xplus = mt_img_sampler_nearest(mt_sample_p[group_ID],XY_sample_xplus);
                        pixel_sample_xplus = mt_img_sampler_nearest(&mt_sample_N0[group_ID*IMAGESIZE_M],XY_sample_xplus);
                        TRANS_XY(XY_sample_xminus,transpara_atE,X-mergesize,Y);
                        //pixel_sample_xminus = mt_img_sampler_nearest(mt_sample_p[group_ID],XY_sample_xminus);
                        pixel_sample_xminus = mt_img_sampler_nearest(&mt_sample_N0[group_ID*IMAGESIZE_M],XY_sample_xminus);
                    }else{
                        TRANS_XY(XY_sample_xplus,transpara_atE,X+mergesize,Y);
                        //pixel_sample_xplus = mt_img_sampler_nearest(mt_sample_p[group_ID],XY_sample_xplus)/2;
                        pixel_sample_xplus = mt_img_sampler_nearest(&mt_sample_N0[group_ID*IMAGESIZE_M],XY_sample_xplus)/2;
                        TRANS_XY(XY_sample_xminus,transpara_atE,X-mergesize,Y);
                        //pixel_sample_xminus = mt_img_sampler_nearest(mt_sample_p[group_ID],XY_sample_xminus)/2;
                        pixel_sample_xminus = mt_img_sampler_nearest(&mt_sample_N0[group_ID*IMAGESIZE_M],XY_sample_xminus)/2;
                    }
                    Dimg_Dx = pixel_sample_xplus.x - pixel_sample_xminus.x;
                    if(Y<mergesize){
                        TRANS_XY(XY_sample_yplus,transpara_atE,X,Y+mergesize);
                        //pixel_sample_yplus = mt_img_sampler_nearest(mt_sample_p[group_ID],XY_sample_yplus);
                        pixel_sample_yplus = mt_img_sampler_nearest(&mt_sample_N0[group_ID*IMAGESIZE_M],XY_sample_yplus);
                        TRANS_XY(XY_sample_yminus,transpara_atE,X,Y);
                        //pixel_sample_yminus = mt_img_sampler_nearest(mt_sample_p[group_ID],XY_sample_yminus);
                        pixel_sample_yminus = mt_img_sampler_nearest(&mt_sample_N0[group_ID*IMAGESIZE_M],XY_sample_yminus);
                    } else if(Y>IMAGESIZE_Y-mergesize) {
                        TRANS_XY(XY_sample_yplus,transpara_atE,X,Y);
                        //pixel_sample_yplus = mt_img_sampler_nearest(mt_sample_p[group_ID],XY_sample_yplus);
                        pixel_sample_yplus = mt_img_sampler_nearest(&mt_sample_N0[group_ID*IMAGESIZE_M],XY_sample_yplus);
                        TRANS_XY(XY_sample_yminus,transpara_atE,X,Y-mergesize);
                        //pixel_sample_yminus = mt_img_sampler_nearest(mt_sample_p[group_ID],XY_sample_yminus);
                        pixel_sample_yminus = mt_img_sampler_nearest(&mt_sample_N0[group_ID*IMAGESIZE_M],XY_sample_yminus);
                    }else{
                        TRANS_XY(XY_sample_yplus,transpara_atE,X,Y+mergesize);
                        //pixel_sample_yplus = mt_img_sampler_nearest(mt_sample_p[group_ID],XY_sample_yplus)/2;
                        pixel_sample_yplus = mt_img_sampler_nearest(&mt_sample_N0[group_ID*IMAGESIZE_M],XY_sample_yplus)/2;
                        TRANS_XY(XY_sample_yminus,transpara_atE,X,Y-mergesize);
                        //pixel_sample_yminus = mt_img_sampler_nearest(mt_sample_p[group_ID],XY_sample_yminus)/2;
                        pixel_sample_yminus = mt_img_sampler_nearest(&mt_sample_N0[group_ID*IMAGESIZE_M],XY_sample_yminus)/2;
                    }
                    Dimg_Dy = pixel_sample_yplus.x - pixel_sample_yminus.x;
                    barrier(CLK_LOCAL_MEM_FENCE);
                    
                    JACOBIAN(transpara_atE,X,Y,Jacob,Dimg_Dx,Dimg_Dy,Mask,mergesize,merge_list[X]);
                    delta_Img = (pixel_target.x-pixel_sample.x)*Mask/mergesize/mergesize*merge_list[X];
                    mt_sample_pixel=pixel_sample.x*Mask/mergesize/mergesize*merge_list[X];
                    
                    LOCMEM_COPY(loc_mem,Jacob,delta_Img,mt_sample_pixel,local_ID,localsize);
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
            
            TRANS_XY(XY_sample,transpara_atE,X,Y);
            //pixel_sample = mt_img_sampler_linear(mt_sample_p[group_ID],XY_sample);
            pixel_sample = mt_img_sampler_linear(&mt_sample_N0[group_ID*IMAGESIZE_M],XY_sample);
            //mt_sample_p[group_ID][X+Y*IMAGESIZE_X] = pixel_sample.x;
            mt_output_N0[group_ID*IMAGESIZE_M+X+Y*IMAGESIZE_X] = pixel_sample.x;
        }
    }
    
    //copy data from local transpara_atE to global transpara
    TRANSPARA_GLOB_COPY(transpara_atE, transpara, local_ID, group_ID);
    TRANSPARA_GLOB_COPY(delta_transpara, transpara_err, local_ID, group_ID);
    
    
}