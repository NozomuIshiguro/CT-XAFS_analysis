#ifndef IMAGESIZE_X
#define IMAGESIZE_X 2048
#endif

#ifndef IMAGESIZE_Y
#define IMAGESIZE_Y 2048
#endif

#ifndef IMAGESIZE_M
#define IMAGESIZE_M 4194304 //2048*2048
#endif

#ifndef NUM_TRANSPARA
#define NUM_TRANSPARA 1
#endif

#ifndef NUM_REDUCTION
#define NUM_REDUCTION 3
#endif

#ifndef LAMBDA
#define LAMBDA 0.001f
#endif

#ifndef NUM_TRIAL
#define NUM_TRIAL 3
#endif

#ifdef DEFINITE
#define READ_PX mergesize
#else
#define READ_PX 1
#endif


//transform XY
#ifndef TRANS_XY
#define TRANS_XY(XY,t,x,y,z) (XY) = (float4)((x)+((__local float*)(t))[0],(y),(z),0)
#endif



//calculate Jacobian
#ifndef JACOBIAN
#define JACOBIAN(t,X,Y,j,dx,dy,msk,ms,ml) {\
((float*)(j))[0] = (dx)*(msk)/(ms)/(ms)*(ml);\
}
#endif



//sampler
__constant sampler_t s_linear = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;
__constant sampler_t s_linear_CLMP_EDGE = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE;



//target merge reading
inline float4 merge_reading_target(__read_only image2d_array_t mt_target,
                                   size_t X,size_t Y,size_t Z,size_t mergesize){
    float4 pixel_target=(float4)(0,0,0,0);
    float4 XYZ_target;
    XYZ_target.z=Z;
    
    for(int i=0;i<mergesize; i++){
        if(X+i<IMAGESIZE_X){
            XYZ_target.x=X+i;
        }else{
            XYZ_target.x=X+i-IMAGESIZE_X;
        }
        for(int j=0;j<mergesize; j++){
            if(Y+j<IMAGESIZE_Y){
                XYZ_target.y=Y+j;
            }else{
                XYZ_target.y=Y+j-IMAGESIZE_Y;
            }
            
            pixel_target += read_imagef(mt_target,s_linear,XYZ_target);
        }
    }
    
    pixel_target /= mergesize*mergesize;
    pixel_target.y=(pixel_target.y>=0.5) ? 1.0:0.0;
    
    return pixel_target;
}

//sample merge reading
inline float4 merge_reading_sample(__read_only image2d_array_t mt_sample,
                                   __local float *transpara_atE,
                                   size_t X,size_t Y,size_t Z,size_t mergesize){
    float4 pixel_sample=(float4)(0,0,0,0);
    float4 XYZ_sample;
    float delta_x, delta_y;
    
    for(int i=0;i<mergesize; i++){
        if(X+i<IMAGESIZE_X){
            delta_x=i;
        }else{
            delta_x=i-IMAGESIZE_X;
        }
        for(int j=0;j<mergesize; j++){
            if(Y+j<IMAGESIZE_Y){
                delta_y=j;
            }else{
                delta_y=j-IMAGESIZE_Y;
            }
            
            TRANS_XY(XYZ_sample,transpara_atE,X+delta_x,Y+delta_y,Z);
            pixel_sample += read_imagef(mt_sample,s_linear,XYZ_sample);
        }
    }
    
    pixel_sample /= mergesize*mergesize;
    pixel_sample.y=(pixel_sample.y>=0.5) ? 1.0:0.0;
    
    return pixel_sample;
}

//copy data from global transpara to local transpara_atE
inline void transpara_copy2local(__local float *transpara_atE, __global float *transpara,
                                 const size_t local_ID,const size_t group_ID){
    
    if(local_ID==0){
        for(size_t i=0;i<NUM_TRANSPARA;i++){
            transpara_atE[i]=transpara[i+group_ID*NUM_TRANSPARA];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
}

//copy data from local transpara_atE to global transpara
inline void transpara_copy2global(__local float *transpara_atE, __global float *transpara,
                                  const size_t local_ID,const size_t group_ID){
    
    if(local_ID==0){
        for(size_t i=0;i<NUM_TRANSPARA;i++){
            transpara[i+group_ID*NUM_TRANSPARA]=transpara_atE[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
}

//Local memory (for reduction) reset
inline void localMemReset(__local float *loc_mem,
                          const size_t local_ID,const size_t localsize){
    
    for(size_t i=0;i<NUM_REDUCTION;i++){
        loc_mem[local_ID+i*localsize]=0;
    }
}

//reset reduction results
inline void reductParaReset(__local float *reductPara){
    for(size_t i=0;i<NUM_REDUCTION;i++){
        reductPara[i]=0;
    }
}

//copy Jacobian & dimg to local memory
inline void copy_Jacab_Dimg(__local float *loc_mem, float *Jacob, float delta_Img, float chi2,
                            const size_t local_ID,const size_t localsize){
    
    size_t t=0;
    for(size_t i=0;i<NUM_TRANSPARA;i++){
        for(size_t j=i;j<NUM_TRANSPARA;j++){
            loc_mem[local_ID+localsize*t]+=Jacob[i]*Jacob[j];
            t++;
        }
    }
    for(size_t i=0;i<NUM_TRANSPARA;i++){
        loc_mem[local_ID+localsize*t]+=Jacob[i]*delta_Img;
        t++;
    }
    loc_mem[local_ID+localsize*t]+=chi2;
    
    barrier(CLK_LOCAL_MEM_FENCE);
}

//reduction
inline void reduction (__local float *loc_mem, __local float *output, uint repeat,
                       const size_t local_ID,const size_t localsize)
{
    
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

//calculate delta_transpara
inline void calc_delta_transpara(__local float *reductPara, float lambda,
                                 __local float *delta_transpara, float delta_rho, char *lambda_init,//caution
                                 const size_t local_ID){
    
    float tJJ[NUM_TRANSPARA][NUM_TRANSPARA];
    float delta_transpara_p[NUM_TRANSPARA];
    float a=0;
    
    
    if(*lambda_init){//caution
        for(size_t i=1;i<NUM_TRANSPARA;i++){
            lambda = fmax(lambda,reductPara[(NUM_TRANSPARA+1)*i]);
        }
        lambda *= LAMBDA;
        *lambda_init=0;//caution
    }
    
    size_t t=0;
    for(size_t i=0;i<NUM_TRANSPARA;i++){
        
        tJJ[i][i]=reductPara[t]*(1 + lambda);
        t++;
        
        for(size_t j=i+1;j<NUM_TRANSPARA;j++){
            tJJ[i][j]=reductPara[t];
            tJJ[j][i]=reductPara[t];
            t++;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(size_t i=0;i<NUM_TRANSPARA;i++){
        
        delta_transpara_p[i]=reductPara[t];
        t++;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //inversed matrix
    for(int i=0; i<NUM_TRANSPARA; i++){
        //devide (i,i) to 1
        a = 1/tJJ[i][i];
        
        for(int j=0; j<NUM_TRANSPARA; j++){
            tJJ[i][j] *= a;
        }
        delta_transpara_p[i] *= a;
        
        //erase (j,i) (i!=j) to 0
        for(int j=i+1; j<NUM_TRANSPARA; j++){
            a = tJJ[j][i];
            for(int k=0; k<NUM_TRANSPARA; k++){
                tJJ[j][k] -= a*tJJ[i][k];
            }
            delta_transpara_p[j] -= a*delta_transpara_p[i];
        }
    }
    for(int i=0; i<NUM_TRANSPARA; i++){
        for(int j=i+1; j<NUM_TRANSPARA; j++){
            a = tJJ[i][j];
            for(int k=0; k<NUM_TRANSPARA; k++){
                tJJ[k][j] -= a*tJJ[k][i];
            }
            delta_transpara_p[i] -= a*delta_transpara_p[j];
        }
    }
    

    if(local_ID==0){
        delta_rho=0.0;
        for(size_t i=0;i<NUM_TRANSPARA;i++){
            delta_transpara[i]=delta_transpara_p[i];
            
            delta_rho += delta_transpara_p[i]*(lambda*delta_transpara_p[i]+reductPara[NUM_REDUCTION-NUM_TRANSPARA-1+i]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

//update transpara
inline void updateTranspara(__local float *delta_transpara,__local float *transpara_atE,
                            size_t mergesize,const size_t local_ID){
    
    if(local_ID==0){
        for(size_t i=0;i<NUM_TRANSPARA;i++){
            transpara_atE[i] += mergesize*delta_transpara[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

//output transpara_err
inline void output_TransparaError(__local float *reductPara,__global float *transpara_err,
                                  const size_t local_ID,const size_t group_ID){
    
    if(local_ID==0){
        for(size_t i=0;i<NUM_TRANSPARA;i++){
            transpara_err[group_ID*2+i]=0.5/fabs(reductPara[NUM_REDUCTION-NUM_TRANSPARA-1+i]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);\
}



//kernel
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
            
            if(I[group_ID*IMAGESIZE_M+ID]<=0) {
                I[group_ID*IMAGESIZE_M+ID]=1.0;
            }
        }
    }
}


__kernel void mt_transform(__global float *I0,__global float *It,
                           __write_only image2d_array_t mt_img,
                           int shapeNo,int startpntX, int startpntY, int width, int height)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    size_t X,Y,ID;
    int4 XYZ;
    float4 mt_f;
    float mask=1.0;
    float radius2=0;
    float mt;
    
    for(int j=0;j<IMAGESIZE_Y;j++){
        for(int i=0;i<IMAGESIZE_X/localsize;i++){
            X=local_ID+i*localsize;
            Y=j;
            ID=X+IMAGESIZE_X*Y;
            XYZ=(int4)(X,Y,group_ID,0);
            
            switch(shapeNo){
                case 0: //square or rectangle
                    if(abs(X-startpntX)<=width/2 & abs(Y-startpntY)<=height/2){
                        mask=1.0;
                    }else{
                        mask=0.0;
                    }
                    break;
                    
                case 1: //circle or orval
                    radius2=(X-startpntX)*(X-startpntX)/width/width+(Y-startpntY)*(Y-startpntY)/height/height;
                    if(radius2<=4){
                        mask=1.0;
                    }else{
                        mask=0.0;
                    }
                    break;
                    
                default:
                    mask=1.0;
                    break;
            }
            
            mt = log(I0[group_ID*IMAGESIZE_M+ID]/It[group_ID*IMAGESIZE_M+ID]);
            mt = (isnan(mt)==1) ? 0:mt;
            
			mt_f = (float4)(mt,mask,0,0);
            write_imagef(mt_img,XYZ,mt_f);
            It[group_ID*IMAGESIZE_M+ID]=mt;
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
    
    char lambda_init=1;
    
    //copy data from global transpara to local transpara_atE
    transpara_copy2local(transpara_atE,transpara,local_ID,group_ID);
    
    for(int i=0;i<IMAGESIZE_X/localsize;i++){
        merge_list[local_ID+i*localsize]=0;
    }
    
    // mrege loop
	//lambda=LAMBDA;
    lambda=0.0;
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
                
            
                pixel_target = merge_reading_target(mt_target,X,Y,group_ID,READ_PX);
                pixel_sample = merge_reading_sample(mt_sample,transpara_atE,X,Y,group_ID,READ_PX);
                
                
                Mask = (int)(pixel_target.y*pixel_sample.y);
				
                mt_sample_pixel=pixel_sample.x*Mask*merge_list[X];
				mt_target_pixel=pixel_target.x*Mask*merge_list[X];
				chi2=(mt_target_pixel-mt_sample_pixel)*(mt_target_pixel-mt_sample_pixel);
                
                loc_mem[local_ID]+=chi2;
            }
        }
        reduction(loc_mem, reductPara,1,local_ID,localsize);
        Fx_old=reductPara[0];
        
        //trial loop (default 3 cycle)
        for(size_t trial=0;trial<NUM_TRIAL;trial++){
            localMemReset(loc_mem,local_ID,localsize);
            reductParaReset(reductPara);
            
            //copy jacobian & delta_Img
            for(int j=0;j<IMAGESIZE_Y;j+=mergesize){
                for(int i=0;i<IMAGESIZE_X;i+=localsize){
                    X=local_ID+i;
                    Y=j;
                    ID=X+Y*IMAGESIZE_X;
                    
                    pixel_target = merge_reading_target(mt_target,X,Y,group_ID,READ_PX);
                    pixel_sample =merge_reading_sample(mt_sample,transpara_atE,X,Y,group_ID,READ_PX);
                    
                    Mask = (int)(pixel_target.y*pixel_sample.y);
                    
                    //Partial differential
                    if(X<mergesize){
                        pixel_sample_xplus=merge_reading_sample(mt_sample,transpara_atE,X+mergesize,Y,group_ID,READ_PX);
                        pixel_sample_xminus =merge_reading_sample(mt_sample,transpara_atE,X,Y,group_ID,READ_PX);

                    } else if(X>IMAGESIZE_X-mergesize) {
						pixel_sample_xplus =merge_reading_sample(mt_sample,transpara_atE,X,Y,group_ID,READ_PX);
                        pixel_sample_xminus =merge_reading_sample(mt_sample,transpara_atE,X-mergesize,Y,group_ID,READ_PX);
                    }else{
                        pixel_sample_xplus =merge_reading_sample(mt_sample,transpara_atE,X+mergesize,Y,group_ID,READ_PX)/2;
                        pixel_sample_xminus =merge_reading_sample(mt_sample,transpara_atE,X-mergesize,Y,group_ID,READ_PX)/2;
                    }
                    Dimg_Dx = pixel_sample_xplus.x - pixel_sample_xminus.x;
                    if(Y<mergesize){
                        pixel_sample_yplus =merge_reading_sample(mt_sample,transpara_atE,X,Y+mergesize,group_ID,READ_PX);
                        pixel_sample_yminus =merge_reading_sample(mt_sample,transpara_atE,X,Y,group_ID,READ_PX);
                    } else if(Y>IMAGESIZE_Y-mergesize) {
                        pixel_sample_yplus =merge_reading_sample(mt_sample,transpara_atE,X,Y,group_ID,READ_PX);
                        pixel_sample_yminus =merge_reading_sample(mt_sample,transpara_atE,X,Y-mergesize,group_ID,READ_PX);
                    }else{
                        pixel_sample_yplus =merge_reading_sample(mt_sample,transpara_atE,X,Y+mergesize,group_ID,READ_PX)/2;
                        pixel_sample_yminus =merge_reading_sample(mt_sample,transpara_atE,X,Y-mergesize,group_ID,READ_PX)/2;
                    }
                    Dimg_Dy = pixel_sample_yplus.x - pixel_sample_yminus.x;
                    barrier(CLK_LOCAL_MEM_FENCE);
                    
                    JACOBIAN(transpara_atE,X,Y,Jacob,Dimg_Dx,Dimg_Dy,Mask,1,merge_list[X]);
                    delta_Img = (pixel_target.x-pixel_sample.x)*Mask*merge_list[X];
                    mt_sample_pixel=pixel_sample.x*Mask*merge_list[X];
					mt_target_pixel=pixel_target.x*Mask*merge_list[X];
					chi2=(mt_target_pixel-mt_sample_pixel)*(mt_target_pixel-mt_sample_pixel);
                    
                    copy_Jacab_Dimg(loc_mem,Jacob,delta_Img,chi2,local_ID,localsize);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            
            //reduction
            reduction(loc_mem,reductPara,NUM_REDUCTION,local_ID,localsize);
            
            //update dumping parameter (lamda)
            Fx_new = reductPara[NUM_REDUCTION-1];
            barrier(CLK_LOCAL_MEM_FENCE);
            
            //update transpara
            calc_delta_transpara(reductPara, lambda, delta_transpara, delta_rho, &lambda_init, local_ID);
            /*caution lambda_init */
            
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
            
            updateTranspara(delta_transpara,transpara_atE,mergesize,local_ID);
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
    transpara_copy2global(transpara_atE, transpara,local_ID,group_ID);
    //output transpara_err
    output_TransparaError(reductPara, transpara_err,local_ID,group_ID);
   
}

__kernel void merge_mt(__read_only image2d_array_t mt_sample, __global float *mt_output)
{
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t globalsize_x = get_global_size(0);
    const size_t globalsize_y = get_global_size(1);
    const size_t global_ID = global_x+globalsize_x*global_y;
    float4 XYZ;
    float mt=0;
    
    
    const size_t mergesize = get_image_array_size(mt_sample);
    for(int i=0;i<mergesize;i++){
        XYZ=(float4)(global_x,global_y,i,0);
        mt+=read_imagef(mt_sample,s_linear,XYZ).x;
    }
    mt_output[global_ID]=mt;
}
