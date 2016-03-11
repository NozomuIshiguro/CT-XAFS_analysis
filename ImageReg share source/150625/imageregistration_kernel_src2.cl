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
#define LAMBDA 1.0f
#endif

#ifndef NUM_TRIAL
#define NUM_TRIAL 3
#endif

//transform XY
#ifndef TRANS_XY
#define TRANS_XY(XY,t,x,y,z,mr) (XY) = (float4)((x)+((__local float*)(t))[0]/(mr),(y),(z),0)
#endif

//calculate Jacobian
#ifndef JACOBIAN
#define JACOBIAN(t,X,Y,j,dx,dy,msk,mr) {\
((float*)(j))[0] = (dx)*(msk)/(mr);\
}
#endif

//sampler
__constant sampler_t s_linear = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;
__constant sampler_t s_linear_CLMP_EDGE = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE;

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
                                 __local float *delta_transpara, float delta_rho, char lambda_init,
                                 const size_t local_ID){
    
    float tJJ[NUM_TRANSPARA][NUM_TRANSPARA];
    float delta_transpara_p[NUM_TRANSPARA];
    float a=0;
    
    //initialize lambda
    if(lambda_init){
        for(size_t i=1;i<NUM_TRANSPARA;i++){
            lambda = fmax(lambda,reductPara[(NUM_TRANSPARA+1)*i]);
        }
        lambda *= LAMBDA;
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
                            const size_t local_ID){
    
    if(local_ID==0){
        for(size_t i=0;i<NUM_TRANSPARA;i++){
            transpara_atE[i] += delta_transpara[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}


//create merged image
__kernel void merge(__read_only image2d_array_t input_img, __write_only image2d_array_t output_img,
                    int mergeN){
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    
    float4 img;
    float4 XYZ_in;
    int4 XYZ_out;
    int X, Y;
    
    for(int i=0;i<IMAGESIZE_Y/mergeN;i++){
        for(int j=0;j<IMAGESIZE_X/mergeN/localsize;j++){
            X = local_ID+j*localsize;
            Y = i;
            img = (float4)(0,0,0,0);
            XYZ_out = (int4)(X,Y,group_ID,0);
            for(int k=0; k<mergeN; k++){
                for(int l=0; l<mergeN; l++){
                    XYZ_in = (float4)(k+X*mergeN,l+mergeN,group_ID,0);
                    img += read_imagef(input_img,s_linear,XYZ_in);
                }
            }
            img /= mergeN*mergeN;
            img.y = (img.y>0.5) ? 1:0;
            write_imagef(output_img,XYZ_out,img);
        }
    }
}

//image registration
__kernel void imageRegistration(__read_only image2d_array_t mt_target,__read_only image2d_array_t mt_sample,
                                __global float *transpara, __global float *transpara_err,
                                __local float *transpara_atE, __local float *delta_transpara,
                                __local float *reductPara, __local float *loc_mem,int mergeN)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    
    
    size_t X,Y;
    //size_t ID;
    float4 XYZ_sample, XYZ_target;
    float4 pixel_sample, pixel_target;
    float mt_sample_pixel, mt_target_pixel;
    int Mask;
    float chi2;
    
    float4 XYZ_sample_xp, XYZ_sample_xm;
    float4 pixel_sample_xp, pixel_sample_xm;
    float4 XYZ_sample_yp, XYZ_sample_ym;
    float4 pixel_sample_yp, pixel_sample_ym;
    
    float Df_Dx, Df_Dy;
    float Jacob[NUM_TRANSPARA];
    float delta_Img;
    
    float Fx_new, Fx_old;
    float lambda, rho, delta_rho, nyu;
    float lambdaA, lambdaB, lambdaC;
    float nyuA, nyuB;
    
    float weight=0;
    
    char lambda_init;
    
    //copy data from global transpara to local transpara_atE
    transpara_copy2local(transpara_atE,transpara,local_ID,group_ID);
    
    
    //initialize LM parameter
    lambda=0.0;
    nyu=2.0f;
    rho=0.0f;
    delta_rho=1.0f;
        
    //estimate initial Fx_old
    loc_mem[local_ID]=0.0f;
    reductPara[0]=0.0f;
    for(int j=0;j<IMAGESIZE_Y/mergeN;j++){
        for(int i=0;i<IMAGESIZE_X/mergeN/localsize;i++){
            X=local_ID+i*localsize;
            Y=j;
            XYZ_target = (float4)(X,Y,group_ID,0);
            TRANS_XY(XYZ_sample,transpara_atE,X,Y,group_ID,mergeN);
                
            pixel_target = read_imagef(mt_target,s_linear,XYZ_target);
            pixel_sample = read_imagef(mt_sample,s_linear,XYZ_sample);
                
            Mask = (int)(pixel_target.y*pixel_sample.y);
                
            mt_sample_pixel=pixel_sample.x*Mask;
            mt_target_pixel=pixel_target.x*Mask;
            chi2=(mt_target_pixel-mt_sample_pixel);
            chi2*=chi2;
            
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
        for(int j=0;j<IMAGESIZE_Y/mergeN;j++){
            for(int i=0;i<IMAGESIZE_X/mergeN/localsize;i++){
                X=local_ID+i*localsize;
                Y=j;
                    
                XYZ_target = (float4)(X,Y,group_ID,0);
                TRANS_XY(XYZ_sample,transpara_atE,X,Y,group_ID,mergeN);
                
                pixel_target = read_imagef(mt_target,s_linear,XYZ_target);
                pixel_sample = read_imagef(mt_sample,s_linear,XYZ_sample);
                    
                Mask = (int)(pixel_target.y*pixel_sample.y);
                    
                //Partial differential
                TRANS_XY(XYZ_sample_xp,transpara_atE,X+1,Y,group_ID,mergeN);
                TRANS_XY(XYZ_sample_xm,transpara_atE,X-1,Y,group_ID,mergeN);
                pixel_sample_xp  = read_imagef(mt_sample,s_linear,XYZ_sample_xp);
                pixel_sample_xm = read_imagef(mt_sample,s_linear,XYZ_sample_xm);
                Df_Dx = (pixel_sample_xp.x - pixel_sample_xm.x)*Mask/2;
                    
                TRANS_XY(XYZ_sample_yp,transpara_atE,X,Y+1,group_ID,mergeN);
                TRANS_XY(XYZ_sample_ym,transpara_atE,X,Y-1,group_ID,mergeN);
                pixel_sample_yp  = read_imagef(mt_sample,s_linear,XYZ_sample_yp);
                pixel_sample_ym = read_imagef(mt_sample,s_linear,XYZ_sample_ym);
                Df_Dy = (pixel_sample_yp.x - pixel_sample_xm.x)*Mask/2;
                
                JACOBIAN(transpara_atE,X,Y,Jacob,Df_Dx,Df_Dy,Mask,mergeN);
                delta_Img = (pixel_target.x-pixel_sample.x)*Mask;
                mt_sample_pixel=pixel_sample.x*Mask;
                mt_target_pixel=pixel_target.x*Mask;
                chi2=(mt_target_pixel-mt_sample_pixel);
                chi2*=chi2;
                    
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
        lambda_init = (lambda_init==0) ? 1:0;
        calc_delta_transpara(reductPara, lambda, delta_transpara, delta_rho, lambda_init, local_ID);
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
            
        updateTranspara(delta_transpara,transpara_atE,local_ID);
    }
    
    
    //copy data from local transpara_atE to global transpara
    transpara_copy2global(transpara_atE,transpara,local_ID,group_ID);
    //output transpara_err
    output_TransparaError(reductPara,transpara_err,local_ID,group_ID);
    
}