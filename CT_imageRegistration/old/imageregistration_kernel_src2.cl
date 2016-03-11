#ifndef ROTATION
#define ROTATION 0
#endif

#ifndef IMAGESIZE_X
#define IMAGESIZE_X 2048
#endif

#ifndef IMAGESIZE_Y
#define IMAGESIZE_Y 2048
#endif

#ifndef IMAGESIZE_M
#define IMAGESIZE_M 4194304 //2048*2048
#endif

#ifndef IT_TARGET_BUF_DEF
#define IT_TARGET_BUF_DEF __global float *It_target,__global float* __local *It_target_p
#endif

#ifndef IT_TARGET_BUF_P
#define IT_TARGET_BUF_P It_target_p[0]=It_target
#endif

#ifndef MT_TARGET_BUF_DEF
#define MT_TARGET_BUF_DEF __global float *mt_target,__global float* __local *mt_target_p
#endif

#ifndef MT_TARGET_BUF_P
#define MT_TARGET_BUF_P mt_target_p[0]=mt_target
#endif

#ifndef I0_SAMPLE_BUF_DEF
#define I0_SAMPLE_BUF_DEF __global float *I0_sample, __global float* __local *I0_sample_p
#endif

#ifndef I0_SAMPLE_BUF_P
#define I0_SAMPLE_BUF_P I0_sample_p[0]=I0_sample
#endif

#ifndef IT_SAMPLE_BUF_DEF
#define IT_SAMPLE_BUF_DEF __global float *It_sample, __global float* __local *It_sample_p
#endif

#ifndef IT_SAMPLE_BUF_P
#define IT_SAMPLE_BUF_P It_sample_p[0]=It_sample
#endif

#ifndef MT_SAMPLE_BUF_DEF
#define MT_SAMPLE_BUF_DEF __global float *mt_sample, __global float* __local *mt_sample_p
#endif

#ifndef MT_SAMPLE_BUF_P
#define MT_SAMPLE_BUF_P mt_sample_p[0]=mt_sample
#endif

#ifndef MT_SAMPLE_OUTPUT_BUF_DEF
#define MT_SAMPLE_OUTPUT_BUF_DEF __global float *mt_sample_output, __global float* __local *mt_sample_output_p
#endif

#ifndef MT_SAMPLE_OUTPUT_BUF_P
#define MT_SAMPLE_OUTPUT_BUF_P mt_sample_output_p[0]=mt_sample_output
#endif

// rotation, x/y shift, or ...
#if ROTATION==1  //rotation+xy shift
#define TRANS_XY(XY,t,x,y)\
(XY) = (float2)( cos(((__local float*)(t))[2])*(x)\
                -sin(((__local float*)(t))[2])*(y)\
                +((__local float*)(t))[0],\
                 sin(((__local float*)(t))[2])*(x)\
                +cos(((__local float*)(t))[2])*(y)\
                +((__local float*)(t))[1])

#define Dtheta(DfxDth,DfyDth,t,x,y,m) \
{\
(DfxDth)= (-(x)*sin(((__local float*)(t))[2])-(y)*cos(((__local float*)(t))[2]))/(m);\
(DfyDth)= ((x)*cos(((__local float*)(t))[2])-(y)*sin(((__local float*)(t))[2]))/(m);\
}


#define JACOB_r(dx,dy,DfxDth,DfyDth,m)\
(dx)*(DfxDth)+(dy)*(DfxDth)*m

#define LOCMEM_RESET(l,lid,lsz)\
{\
((__local float*)(l))[lid]=0;\
((__local float*)(l))[lid+lsz]=0;\
((__local float*)(l))[lid+lsz*2]=0;\
((__local float*)(l))[lid+lsz*3]=0;\
((__local float*)(l))[lid+lsz*4]=0;\
((__local float*)(l))[lid+lsz*5]=0;\
((__local float*)(l))[lid+lsz*6]=0;\
((__local float*)(l))[lid+lsz*7]=0;\
((__local float*)(l))[lid+lsz*8]=0;\
((__local float*)(l))[lid+lsz*9]=0;\
}

#define LOCMEM_COPY(l,j,dimg,p,lid,lsz)\
{\
float e[9];\
((__local float*)(l))[lid]+=((float*)(j))[0]*((float*)(j))[0];\
((__local float*)(l))[lid+lsz]+=((float*)(j))[1]*((float*)(j))[0];\
((__local float*)(l))[lid+lsz*2]+=((float*)(j))[2]*((float*)(j))[0];\
((__local float*)(l))[lid+lsz*3]+=((float*)(j))[1]*((float*)(j))[0];\
((__local float*)(l))[lid+lsz*4]+=((float*)(j))[2]*((float*)(j))[0];\
((__local float*)(l))[lid+lsz*5]+=((float*)(j))[2]*((float*)(j))[2];\
((__local float*)(l))[lid+lsz*6]+=((float*)(j))[0]*(dimg);\
((__local float*)(l))[lid+lsz*7]+=((float*)(j))[1]*(dimg);\
((__local float*)(l))[lid+lsz*8]+=((float*)(j))[2]*(dimg);\
((__local float*)(l))[lid+lsz*9]+=(p);\
}

#define REDUCTION(l,rp)\
{\
reduction((__local float*)(l),(__local float*)(rp),(10));\
}

#define NUM_REDUCTION 10

#define CALC_TRANSPARA(rp,l,dtp,tp,m,lid) \
{\
float tJJ[3][3];\
tJJ[0][0]=((__local float*)(rp))[0]+(l);\
tJJ[0][1]=((__local float*)(rp))[1];\
tJJ[0][2]=((__local float*)(rp))[2];\
tJJ[1][0]=((__local float*)(rp))[1];\
tJJ[1][1]=((__local float*)(rp))[3]+(l);\
tJJ[1][2]=((__local float*)(rp))[4]+(l);\
tJJ[2][0]=((__local float*)(rp))[2]+(l);\
tJJ[2][1]=((__local float*)(rp))[4]+(l);\
tJJ[2][2]=((__local float*)(rp))[5]+(l);\
\
float det_tJJ;\
det_tJJ = tJJ[0][0]*tJJ[1][1]*tJJ[2][2] +tJJ[0][1]*tJJ[1][2]*tJJ[2][0] +tJJ[0][2]*tJJ[1][0]*tJJ[2][1]\
-tJJ[0][0]*tJJ[1][2]*tJJ[2][1] -tJJ[0][1]*tJJ[1][0]*tJJ[2][2] -tJJ[0][2]*tJJ[1][1]*tJJ[2][0];\
\
float inv_tJJ[3][3];\
inv_tJJ[0][0] =  (tJJ[1][1]*tJJ[2][2]-tJJ[1][2]*tJJ[2][1])/(det_tJJ);\
inv_tJJ[0][1] = -(tJJ[1][0]*tJJ[2][2]-tJJ[1][2]*tJJ[2][0])/(det_tJJ);\
inv_tJJ[0][2] =  (tJJ[1][0]*tJJ[2][1]-tJJ[1][1]*tJJ[2][0])/(det_tJJ);\
inv_tJJ[1][0] = -(tJJ[0][1]*tJJ[2][2]-tJJ[0][2]*tJJ[2][1])/(det_tJJ);\
inv_tJJ[1][1] =  (tJJ[0][0]*tJJ[2][2]-tJJ[0][2]*tJJ[2][0])/(det_tJJ);\
inv_tJJ[1][2] = -(tJJ[0][0]*tJJ[2][1]-tJJ[0][1]*tJJ[2][0])/(det_tJJ);\
inv_tJJ[2][0] =  (tJJ[0][1]*tJJ[1][2]-tJJ[0][2]*tJJ[1][1])/(det_tJJ);\
inv_tJJ[2][1] = -(tJJ[0][0]*tJJ[1][2]-tJJ[0][2]*tJJ[1][0])/(det_tJJ);\
inv_tJJ[2][2] =  (tJJ[0][0]*tJJ[1][1]-tJJ[0][1]*tJJ[1][0])/(det_tJJ);\
\
if((lid)==0){\
    ((__local float*)(dtp))[0]=inv_tJJ[0][0]*((__local float*)(rp))[6]+inv_tJJ[0][1]*((__local float*)(rp))[7]+inv_tJJ[0][2]*((float*)(rp))[8];\
    ((__local float*)(dtp))[1]=inv_tJJ[1][0]*((__local float*)(rp))[6]+inv_tJJ[1][1]*((__local float*)(rp))[7]+inv_tJJ[1][2]*((float*)(rp))[8];\
    ((__local float*)(dtp))[2]=inv_tJJ[2][0]*((__local float*)(rp))[6]+inv_tJJ[2][1]*((__local float*)(rp))[7]+inv_tJJ[2][2]*((float*)(rp))[8];\
    \
    ((__local float*)(tp))[0] += (m)*((__local float*)(dtp))[0];\
    ((__local float*)(tp))[1] += (m)*((__local float*)(dtp))[1];\
    ((__local float*)(tp))[2] += (m)*((__local float*)(dtp))[2];\
    \
    (dr) = (((__local float*)(dtp))[0]*((l)*((__local float*)(dtp))[0]-((__local float*)(rp))[6])\
    +((__local float*)(dtp))[1]*((l)*((__local float*)(dtp))[1]-((__local float*)(rp))[7])\
    +((__local float*)(dtp))[2]*((l)*((__local float*)(dtp))[2]-((__local float*)(rp))[8]))/2;\
}\
barrier(CLK_LOCAL_MEM_FENCE);\
\
}

#define REDUCTPARA_RESET(rp)\
{\
((__local float*)(rp))[0]=0;\
((__local float*)(rp))[1]=0;\
((__local float*)(rp))[2]=0;\
((__local float*)(rp))[3]=0;\
((__local float*)(rp))[4]=0;\
((__local float*)(rp))[5]=0;\
((__local float*)(rp))[6]=0;\
((__local float*)(rp))[7]=0;\
((__local float*)(rp))[8]=0;\
((__local float*)(rp))[9]=0;\
}

#define TRANSPARA_LOC_COPY(tp,tpg,lid,gid)\
{\
if((lid)==0){\
((__local float*)(tp))[0]=((__global float*)(tpg))[gid*3];\
((__local float*)(tp))[1]=((__global float*)(tpg))[gid*3+1];\
((__local float*)(tp))[2]=((__global float*)(tpg))[gid*3+2];\
}\
barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);\
}

#define TRANSPARA_GLOB_COPY(tp,tpg,lid,gid)\
{\
if((lid)==0){\
((__global float*)(tpg))[gid*3]=((__local float*)(tp))[0];\
((__global float*)(tpg))[gid*3+1]=((__local float*)(tp))[1];\
((__global float*)(tpg))[gid*3+2]=((__local float*)(tp))[2];\
}\
barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);\
}

#define NUM_TRANSPARA 3

#else //xy shift
#define TRANS_XY(XY,t,x,y)\
(XY) = (float2)((x)+((__local float*)(t))[0],\
                (y)+((__local float*)(t))[1])

#define Dtheta(DfxDth,DfyDth,t,x,y,m) \
{\
(DfxDth)= 0.0;\
(DfyDth)= 0.0;\
}

#define JACOB_r(dx,dy,DfxDth,DfyDth,m) 0.0

#define LOCMEM_RESET(l,lid,lsz)\
{\
((__local float*)(l))[lid]=0;\
((__local float*)(l))[lid+lsz]=0;\
((__local float*)(l))[lid+lsz*2]=0;\
((__local float*)(l))[lid+lsz*3]=0;\
((__local float*)(l))[lid+lsz*4]=0;\
((__local float*)(l))[lid+lsz*5]=0;\
}

#define LOCMEM_COPY(l,j,dimg,p,lid,lsz)\
{\
((__local float*)(l))[lid]+=((float*)(j))[0]*((float*)(j))[0];\
((__local float*)(l))[lid+lsz]+=((float*)(j))[1]*((float*)(j))[0];\
((__local float*)(l))[lid+lsz*2]+=((float*)(j))[1]*((float*)(j))[1];\
((__local float*)(l))[lid+lsz*3]+=((float*)(j))[0]*(dimg);\
((__local float*)(l))[lid+lsz*4]+=((float*)(j))[1]*(dimg);\
((__local float*)(l))[lid+lsz*5]+=(p);\
}

#define REDUCTION(l,rp)\
{\
reduction((__local float*)(l),(__local float*)(rp),(6));\
}

#define NUM_REDUCTION 6

#define CALC_TRANSPARA(rp,l,dtp,tp,m,dr,lid) \
{\
float tJJ[2][2];\
tJJ[0][0]=((__local float*)(rp))[0]+(l);\
tJJ[0][1]=((__local float*)(rp))[1];\
tJJ[1][0]=((__local float*)(rp))[1];\
tJJ[1][1]=((__local float*)(rp))[2]+(l);\
\
float det_tJJ;\
det_tJJ = tJJ[0][0]*tJJ[1][1] - tJJ[1][0]*tJJ[0][1];\
\
float inv_tJJ[2][2];\
inv_tJJ[0][0] =  tJJ[1][1]/(det_tJJ);\
inv_tJJ[0][1] = -tJJ[0][1]/(det_tJJ);\
inv_tJJ[1][0] = -tJJ[1][0]/(det_tJJ);\
inv_tJJ[1][1] =  tJJ[0][0]/(det_tJJ);\
\
if((lid)==0){\
    ((__local float*)(dtp))[0]=inv_tJJ[0][0]*((__local float*)(rp))[3]+inv_tJJ[0][1]*((__local float*)(rp))[4];\
    ((__local float*)(dtp))[1]=inv_tJJ[1][0]*((__local float*)(rp))[3]+inv_tJJ[1][1]*((__local float*)(rp))[4];\
    \
    ((__local float*)(tp))[0] += (m)*((__local float*)(dtp))[0];\
    ((__local float*)(tp))[1] += (m)*((__local float*)(dtp))[1];\
    \
    (dr) = (((__local float*)(dtp))[0]*((l)*((__local float*)(dtp))[0]-((__local float*)(rp))[3])\
    +((__local float*)(dtp))[1]*((l)*((__local float*)(dtp))[1]-((__local float*)(rp))[4]))/2.0;\
}\
barrier(CLK_LOCAL_MEM_FENCE);\
\
}

#define REDUCTPARA_RESET(rp)\
{\
((__local float*)(rp))[0]=0;\
((__local float*)(rp))[1]=0;\
((__local float*)(rp))[2]=0;\
((__local float*)(rp))[3]=0;\
((__local float*)(rp))[4]=0;\
((__local float*)(rp))[5]=0;\
}

#define TRANSPARA_LOC_COPY(tp,tpg,lid,gid)\
{\
if((lid)==0){\
((__local float*)(tp))[0]=((__global float*)(tpg))[gid*2];\
((__local float*)(tp))[1]=((__global float*)(tpg))[gid*2+1];\
}\
barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);\
}

#define TRANSPARA_GLOB_COPY(tp,tpg,lid,gid)\
{\
if((lid)==0){\
((__global float*)(tpg))[gid*2]=((__local float*)(tp))[0];\
((__global float*)(tpg))[gid*2+1]=((__local float*)(tp))[1];\
}\
barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);\
}

#define NUM_TRANSPARA 2

#endif


/*Atomic float add*/
float atom_add_float(__local float* const address, const float value)
{
    uint oldval, newval, readback;
    
    *(float*)&oldval = *address;
    *(float*)&newval = (*(float*)&oldval + value);
    while ((readback = atom_cmpxchg((__local uint*)address, oldval, newval)) != oldval) {
        oldval = readback;
        *(float*)&newval = (*(float*)&oldval + value);
    }
    return *(float*)&oldval;
}


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
                atom_add_float(&output[i], loc_mem[i*localsize]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


//linear sampler
inline float2 mt_img_sampler(__global float *mt, float2 XY)
{
    size_t leftX = floor(XY.x);
    size_t rightX = leftX+1;
    size_t downY =floor(XY.y);
    size_t upY = downY+1;
    size_t ID_ul = leftX+upY*IMAGESIZE_X;
    size_t ID_ur = rightX+upY*IMAGESIZE_X;
    size_t ID_dl = leftX+downY*IMAGESIZE_X;
    size_t ID_dr = rightX+downY*IMAGESIZE_X;
    float mt_sampler, mt1, mt2;
    
    if((XY.x*(XY.x-IMAGESIZE_X+1)<=0)&(XY.y*(XY.y-IMAGESIZE_Y+1)<=0)){
        
        mt1 = (mt[ID_ur]-mt[ID_ul])*(XY.x-leftX)+mt[ID_ul];
        mt2 = (mt[ID_dr]-mt[ID_dl])*(XY.x-leftX)+mt[ID_dl];
        mt_sampler = (mt1-mt2)*(XY.y-downY)+mt1;

        return (float2)(mt_sampler,1.0f);
    } else {
        return (float2)(0.0f,0.0f);
    }
}


__kernel void mt_target_transform(__global float *dark,
                                  __global float *I0_target,
                                  IT_TARGET_BUF_DEF,
                                  MT_TARGET_BUF_DEF)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    size_t X,Y,ID;
    int2 XY;
    if(local_ID==0){
        IT_TARGET_BUF_P;
        MT_TARGET_BUF_P;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(int j=0;j<IMAGESIZE_Y;j++){
        for(int i=0;i<IMAGESIZE_X/localsize;i++){
            X=local_ID+i*localsize;
            Y=j;
            ID=X+IMAGESIZE_X*Y;
            XY = (int2)(X,Y);
            
            //target image data
            mt_target_p[group_ID][ID] = log((I0_target[ID]-dark[ID])/(It_target_p[group_ID][ID]-dark[ID]));
        }
    }
}

__kernel void mt_sample_transform(__global float *dark,
                                  I0_SAMPLE_BUF_DEF,
                                  IT_SAMPLE_BUF_DEF,
                                  MT_SAMPLE_BUF_DEF,
                                  int num_E)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    size_t X,Y,ID;
    int2 XY;
    
    if(local_ID==0){
       I0_SAMPLE_BUF_P;
       IT_SAMPLE_BUF_P;
       MT_SAMPLE_BUF_P;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //energy loop
    for(size_t energyID=0;energyID<num_E;energyID++){
        for(int j=0;j<IMAGESIZE_Y;j++){
            for(int i=0;i<IMAGESIZE_X/localsize;i++){
                X=local_ID+i*localsize;
                Y=j;
                ID=X+IMAGESIZE_X*Y;
                XY = (int2)(X,Y);
                
                mt_sample_p[energyID+num_E*group_ID][ID] = log((I0_sample_p[energyID][ID]-dark[ID])/(It_sample_p[energyID+num_E*group_ID][ID]-dark[ID]));
            }
        }
    }
    
}


__kernel void imageRegistration(MT_SAMPLE_BUF_DEF,
                                MT_TARGET_BUF_DEF,
                                MT_SAMPLE_OUTPUT_BUF_DEF,
                                __global float *transpara,
                                __local float *transpara_atE,
                                __local float *delta_transpara,
                                __local float *reductPara,
                                __local float *loc_mem,
                                int num_E)
{
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    if(local_ID==0){
        MT_SAMPLE_BUF_P;
        MT_TARGET_BUF_P;
        MT_SAMPLE_OUTPUT_BUF_P;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    size_t X,Y;
    size_t ID;
    float2 XY_sample, XY_target;
    float2 pixel_sample, pixel_target;
    int Mask;
    
    float2 XY_sample_xplus, XY_sample_xminus;
    float2 pixel_sample_xplus, pixel_sample_xminus;
    float2 XY_sample_yplus, XY_sample_yminus;
    float2 pixel_sample_yplus, pixel_sample_yminus;
    
    float Dimg_Dx, Dimg_Dy, Dfx_Dtheta, Dfy_Dtheta;
    float Jacob[3];
    float delta_Img;
    
    float Fx_new, Fx_old;
    float lambda, rho, delta_rho;
    
    float deltaShiftSize;
    
    TRANSPARA_LOC_COPY(transpara_atE,transpara,local_ID,num_E*group_ID);
    
    //energy loop
    for(size_t energyID=0;energyID<num_E;energyID++){
        // mrege loop
        for(size_t mergesize=8;mergesize>0;){
            lambda=0.2;
            rho=0.0;
            delta_rho=1.0;
            
            //trial loop
            for(size_t trial=0;trial<3;trial++){
                LOCMEM_RESET(loc_mem,local_ID,localsize);
                REDUCTPARA_RESET(reductPara);
                
                //copy jacobian & delta_Img
                for(int j=0;j<IMAGESIZE_Y;j++){
                    for(int i=0;i<IMAGESIZE_X/localsize;i++){
                        X=local_ID+i*localsize;
                        Y=j;
                        ID=X+Y*IMAGESIZE_X;
                        
                        XY_target = (float2)(X,Y);
                        pixel_target = (float2)(mt_target_p[group_ID][ID],1);
                        
                        TRANS_XY(XY_sample,transpara_atE,X,Y);
                        pixel_sample = mt_img_sampler(mt_sample_p[energyID+num_E*group_ID],XY_sample);
                        
                        Mask = (int)(pixel_target.y*pixel_sample.y);
                        
                        //Partial differential
                        if(X<mergesize){
                            TRANS_XY(XY_sample_xplus,transpara_atE,X+mergesize,Y);
                            pixel_sample_xplus = mt_img_sampler(mt_sample_p[energyID+num_E*group_ID],XY_sample_xplus);
                            TRANS_XY(XY_sample_xminus,transpara,X,Y);
                            pixel_sample_xminus = mt_img_sampler(mt_sample_p[energyID+num_E*group_ID],XY_sample_xminus);
                        } else if(X>IMAGESIZE_X-mergesize) {
                            TRANS_XY(XY_sample_xplus,transpara_atE,X,Y);
                            pixel_sample_xplus = mt_img_sampler(mt_sample_p[energyID+num_E*group_ID],XY_sample_xplus);
                            TRANS_XY(XY_sample_xminus,transpara_atE,X-mergesize,Y);
                            pixel_sample_xminus = mt_img_sampler(mt_sample_p[energyID+num_E*group_ID],XY_sample_xminus);
                        }else{
                            TRANS_XY(XY_sample_xplus,transpara_atE,X+mergesize,Y);
                            pixel_sample_xplus = mt_img_sampler(mt_sample_p[energyID+num_E*group_ID],XY_sample_xplus)/2;
                            TRANS_XY(XY_sample_xminus,transpara_atE,X-mergesize,Y);
                            pixel_sample_xminus = mt_img_sampler(mt_sample_p[energyID+num_E*group_ID],XY_sample_xminus)/2;
                        }
                        Dimg_Dx = pixel_sample_xplus.x - pixel_sample_xminus.x;
                        if(Y<mergesize){
                            TRANS_XY(XY_sample_yplus,transpara_atE,X,Y+mergesize);
                            pixel_sample_yplus = mt_img_sampler(mt_sample_p[energyID+num_E*group_ID],XY_sample_yplus);
                            TRANS_XY(XY_sample_yminus,transpara_atE,X,Y);
                            pixel_sample_yminus = mt_img_sampler(mt_sample_p[energyID+num_E*group_ID],XY_sample_yminus);
                        } else if(Y>IMAGESIZE_Y-mergesize) {
                            TRANS_XY(XY_sample_yplus,transpara_atE,X,Y);
                            pixel_sample_yplus = mt_img_sampler(mt_sample_p[energyID+num_E*group_ID],XY_sample_yplus);
                            TRANS_XY(XY_sample_yminus,transpara_atE,X,Y-mergesize);
                            pixel_sample_yminus = mt_img_sampler(mt_sample_p[energyID+num_E*group_ID],XY_sample_yminus);
                        }else{
                            TRANS_XY(XY_sample_yplus,transpara_atE,X,Y+mergesize);
                            pixel_sample_yplus = mt_img_sampler(mt_sample_p[energyID+num_E*group_ID],XY_sample_yplus)/2;
                            TRANS_XY(XY_sample_yminus,transpara_atE,X,Y-mergesize);
                            pixel_sample_yminus = mt_img_sampler(mt_sample_p[energyID+num_E*group_ID],XY_sample_yminus)/2;
                        }
                        Dimg_Dy = pixel_sample_yplus.x - pixel_sample_yminus.x;
                        Dtheta(Dfx_Dtheta,Dfy_Dtheta,transpara_atE,X,Y,mergesize);
                        barrier(CLK_GLOBAL_MEM_FENCE);
                        
                        //Create Jacobian matrix
                        if((X%mergesize==0)&(Y%mergesize==0)){
                            Jacob[0] = Dimg_Dx*Mask/mergesize/mergesize;
                            Jacob[1] = Dimg_Dy*Mask/mergesize/mergesize;
                            Jacob[2] = JACOB_r(Dimg_Dx,Dimg_Dy,Dfx_Dtheta,Dfy_Dtheta,Mask);
                            delta_Img = (pixel_target.x-pixel_sample.x)*Mask/mergesize/mergesize;
                        } else {
                            Jacob[0] = 0.0;
                            Jacob[1] = 0.0;
                            Jacob[2] = 0.0;
                            delta_Img = 0.0;
                        }
                        
                        LOCMEM_COPY(loc_mem,Jacob,delta_Img,pixel_sample.x,local_ID,localsize);
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                
                //reduction
                REDUCTION(loc_mem,reductPara);
                
                //update dumping parameter (lamda)
                Fx_new = reductPara[NUM_REDUCTION-1];
                if(trial>0){
                    rho = (Fx_old-Fx_new)/delta_rho;
                    if(rho>0.0f){
                        rho = 1.0f-(2.0f*rho-1.0f)*(2.0f*rho-1.0f)*(2.0f*rho-1.0f);
                        lambda = lambda*fmax(0.333f,rho);
                    }else{
                        lambda = lambda*2.0f;
                    }
                }
                //not used now
                //lambda = fmax(fmax(reductPara[0],reductPara[1]),reductPara[5]);
                //while(lambda>1){
                //lambda/=10.0f;
                //}
                
                //update transpara
                CALC_TRANSPARA(reductPara,lambda,delta_transpara,transpara_atE,mergesize,delta_rho,local_ID);
                Fx_old = Fx_new;
                
                //judging trial loop cutoff
                deltaShiftSize = delta_transpara[0]*delta_transpara[0]+delta_transpara[1]*delta_transpara[1];
                if (deltaShiftSize<1) break;
            }
            
            //judging merge loop cutoff
            if ((mergesize>1)&(deltaShiftSize<1/mergesize/mergesize)) {
                mergesize=1;
                continue;
            }else{
                mergesize>>=1;
            }
        }
        
        //output registrated mt image
        for(int j=0;j<IMAGESIZE_Y;j++){
            for(int i=0;i<IMAGESIZE_X/localsize;i++){
                X=local_ID+i*localsize;
                Y=j;
                
                TRANS_XY(XY_sample,transpara_atE,X,Y);
                pixel_sample = mt_img_sampler(mt_sample_p[energyID+num_E*group_ID],XY_sample);
                
                mt_sample_output_p[energyID+num_E*group_ID][X+Y*IMAGESIZE_X] = pixel_sample.x;
            }
        }
        
        TRANSPARA_GLOB_COPY(transpara_atE,transpara,local_ID,energyID+num_E*group_ID);
    }
}