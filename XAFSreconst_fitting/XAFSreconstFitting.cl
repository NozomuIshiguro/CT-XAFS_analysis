
#define PI 3.14159265358979323846

#define PI_180 0.01745329252

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

#ifndef PRJ_IMAGESIZE_M
#define PRJ_IMAGESIZE_M 327680
#endif

#ifndef SS
#define SS 16
#endif

#ifndef SS_ANGLESIZE
#define SS_ANGLESIZE 10
#endif

#ifndef PARA_NUM
#define PARA_NUM 1
#endif

#ifndef PARA_NUM_SQ
#define PARA_NUM_SQ 1
#endif

#ifndef ENERGY_NUM
#define ENERGY_NUM 1
#endif

#ifndef E0
#define E0 11559.0f
#endif

#ifndef CONSTRAIN_NUM
#define CONSTRAIN_NUM 0
#endif

#ifndef FIT
#define FIT(x,y,fp){\
\
(y) = ((float*)(fp))[0] + ((float*)(fp))[1]*(x);\
\
}
#endif

#ifndef JACOBIAN
#define JACOBIAN(x,j,fp){\
\
((float*)(j))[0]=1.0;\
((float*)(j))[1]=(x);\
\
}
#endif

#ifndef AMP_FACTOR
#define AMP_FACTOR 1.0f
#endif

#ifndef AART_FACTOR
#define AART_FACTOR 0.1f
#endif

__constant sampler_t s_linear = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;
__constant sampler_t s_nearest = CLK_FILTER_NEAREST|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;


//reduction
inline void reduction(__local float *loc_mem, const size_t local_ID, const size_t localsize)
{
    for(size_t s=localsize/2;s>0;s>>=1){
        if(local_ID<s){
            loc_mem[local_ID]+=loc_mem[local_ID+s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

//simultaneous linear equation
static inline void sim_linear_eq_global(__global float *A, float *x,
                                        size_t ID, size_t imgSize,
                                        size_t dim,__constant char *p_fix){
    
    float a,b,c,d;
    size_t i,j,k;
    
    for(i=0; i<dim; i++){
        if(p_fix[i]==48) continue;
        
        //devide (i,i) to 1
        a = 1.0f/A[ID+(i+i*dim)*IMAGESIZE_M];
        for(int j=0; j<dim; j++){
            if(p_fix[j]== 48) continue;
            A[ID+(i+j*dim)*IMAGESIZE_M] *= a;
        }
        x[i] *= a;
        
        //erase (j,i) (i!=j) to 0
        for(int j=i+1; j<dim; j++){
            if(p_fix[j]== 48) continue;
            
            a = A[ID+(j+i*dim)*IMAGESIZE_M];
            for(int k=0; k<dim; k++){
                if(p_fix[k]== 48) continue;
                
                A[ID+(j+k*dim)*IMAGESIZE_M] -= a*A[ID+(i+k*dim)*IMAGESIZE_M];
            }
            x[j] -= a*x[i];
        }
    
    }
    
    for(int i=0; i<dim; i++){
        if(p_fix[i]== 48) continue;
        
        for(int j=i+1; j<dim; j++){
            if(p_fix[j] == 48) continue;
            
            a = A[ID+(i+j*dim)*IMAGESIZE_M];
            for(int k=0; k<dim; k++){
                if(p_fix[k] == 48) continue;
                
                A[ID+(k+j*dim)*IMAGESIZE_M] -= a*A[ID+(k+i*dim)*IMAGESIZE_M];
            }
            x[i] -= a*x[j];
        }
    }
}


//assign parameter to Fitting eq.
__kernel void assign2FittingEq(__global float* p, __write_only image2d_t mt_fit_img,
                               __constant float *energyList, int Enum){
    
    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    const size_t ID = X+Y*IMAGESIZE_X;
    const int2 XY = (int2)(X,Y);
    float energy = energyList[Enum];
    float fp[PARA_NUM];
    int i;
    
    //copy global p to private fp
    for(i=0;i<PARA_NUM;i++){
        fp[i] = p[ID+i*IMAGESIZE_M];
    }
    float mt_fit;
    FIT(energy,mt_fit,fp);
    
    write_imagef(mt_fit_img,XY,(float4)(mt_fit,0.0f,0.0f,1.0f));
}

//assign parameter to Jacobian
__kernel void assign2Jacobian(__global float* p,__write_only image2d_array_t jacob_img,
                               __constant float *energyList, int Enum){
    
    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    const size_t ID = X+Y*IMAGESIZE_X;
    const int4 XYP = (int4)(X,Y,0,0);
    float energy = energyList[Enum];
    float fp[PARA_NUM];
    float J[PARA_NUM];
    int i;
    
    //copy global p to private fp
    for(i=0;i<PARA_NUM;i++){
        fp[i] = p[ID+i*IMAGESIZE_M];
    }
    JACOBIAN(energy,J,fp);
    
    for(i=0;i<PARA_NUM;i++){
        XYP.z=i;
        write_imagef(jacob_img,XYP,(float4)(J[i],0.0f,0.0f,1.0f));
    }
}

//projectoion from mt_fit to prj_delta_mt
__kernel void projectionToDeltaMt(__read_only image2d_t mt_fit_img,
                                  __read_only image2d_array_t prj_mt_img,
                                  __write_only image2d_t prj_delta_mt_img,
                                  __constant float *anglelist, int sub, int Enum){

    const int X = get_global_id(0);
    const int th = get_global_id(1);

    float2 XY;
    float4 XthE_f= (float4)(X,sub+th*SS,Enum,0);
    int2 Xth_i = (int2)(X,th);
    float prj =0.0f;
    float angle = anglelist[sub+th*SS]*PI_180;
    
    for(int Y=0;Y<PRJ_IMAGESIZE;Y++){
        XY.x =  (X-PRJ_IMAGESIZE*0.5f)*cos(angle)+(Y-PRJ_IMAGESIZE*0.5f)*sin(angle) + IMAGESIZE_X*0.5f;
        XY.y = -(X-PRJ_IMAGESIZE*0.5f)*sin(angle)+(Y-PRJ_IMAGESIZE*0.5f)*cos(angle) + IMAGESIZE_Y*0.5f;
    
        prj += read_imagef(mt_fit_img,s_linear,XY).x;
    }
    float4 img = (float4)(prj,0.0f,0.0f,1.0f);
    img.x = read_imagef(prj_mt_img,s_linear,XthE_f).x - prj/AMP_FACTOR;
    
    write_imagef(prj_delta_mt_img, Xth_i, img);
}

//projectoion array image
__kernel void projectionArray(__read_only image2d_array_t src_img,
                              __write_only image2d_array_t prj_img,
                              __constant float *anglelist, int sub){
    
    const int X = get_global_id(0);
    const int th = get_global_id(1);
    const int Z = get_global_id(2);
    
    float4 XYZ;
    XYZ.z = Z;
    int4 XthZ_i = (int4)(X,th,Z,0);
    float prj =0.0f;
    float angle = anglelist[sub+th*SS]*PI_180;
    
    int Y;
    for(Y=0;Y<PRJ_IMAGESIZE;Y++){
        XYZ.x =  (X-PRJ_IMAGESIZE*0.5f)*cos(angle)+(Y-PRJ_IMAGESIZE*0.5f)*sin(angle) + IMAGESIZE_X*0.5f;
        XYZ.y = -(X-PRJ_IMAGESIZE*0.5f)*sin(angle)+(Y-PRJ_IMAGESIZE*0.5f)*cos(angle) + IMAGESIZE_Y*0.5f;
        prj += read_imagef(src_img,s_linear,XYZ).x;
    }
    float4 img = (float4)(prj/AMP_FACTOR,0.0f,0.0f,1.0f);
    
    write_imagef(prj_img, XthZ_i, img);
}


//back projection of prj image (subset considered)
//"For loop" for th is conducted at outside kernel
__kernel void backProjectionSingle(__global float* bprj_img,
                                   __read_only image2d_t prj_img,
                                   __constant float *anglelist, int sub){
    
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const size_t ID =(size_t)X+(size_t)Y*IMAGESIZE_X;
    
    float2 Xth;
    
    float angle;
    float bprj = 0.0f;
    int th;
    for(th=0;th<SS_ANGLESIZE;th++){
        angle = anglelist[sub+th*SS]*PI_180;
        Xth.x = (X-IMAGESIZE_X*0.5f)*cos(angle)-(Y-IMAGESIZE_Y*0.5f)*sin(angle)+PRJ_IMAGESIZE*0.5f;
        Xth.y = th;
        bprj += read_imagef(prj_img,s_nearest,Xth).x;
    }
    bprj_img[ID] = bprj;
}


//back projection of prj image array (subset considered)
//"For loop" for th is conducted at outside kernel
__kernel void backProjectionArray(__global float* bprj_img,
                                  __read_only image2d_array_t prj_img,
                                  __constant float *anglelist, int sub){
    
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int P = get_global_id(2);
    const size_t ID =(size_t)X+(size_t)Y*IMAGESIZE_X+(size_t)P*IMAGESIZE_M;
    
    float4 XthP;
    
    float angle;
    float bprj = 0.0f;
    
    int th;
    for(th=0;th<SS_ANGLESIZE;th++){
        angle = anglelist[sub+th*SS]*PI_180;
        XthP.x = (X-IMAGESIZE_X*0.5f)*cos(angle)-(Y-IMAGESIZE_Y*0.5f)*sin(angle)+PRJ_IMAGESIZE*0.5f;
        XthP.y = th;
        XthP.z = P;
    
        bprj += read_imagef(prj_img,s_nearest,XthP).x;
    }
    bprj_img[ID] = bprj;
}

//back projection of prj image array (full angle)
//"For loop" for th is conducted at outside kernel
__kernel void backProjectionArrayFull(__global float* bprj_img,
                                      __read_only image2d_array_t prj_img,
                                      __constant float *anglelist){
    
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int E = get_global_id(2);
    const size_t ID =(size_t)X+(size_t)Y*IMAGESIZE_X+(size_t)E*IMAGESIZE_M;
    float4 XthE;
    XthE.z = E;
    
    float angle;
    float bprj = 0.0f;
    int th;
    
    for(th=0;th<SS_ANGLESIZE;th++){
        angle = anglelist[th]*PI_180;
        XthE.x = (X-IMAGESIZE_X*0.5f)*cos(angle)-(Y-IMAGESIZE_Y*0.5f)*sin(angle)+PRJ_IMAGESIZE*0.5f;
        XthE.y = th;
        bprj += read_imagef(prj_img,s_nearest,XthE).x;
    }
    bprj_img[ID] = bprj;
}


//calculation of chi2
//"For loop" for E is conducted at outside kernel
__kernel void calcChi2_1(__global float *chi2,
                       __read_only image2d_t prj_delta_mt_img){
    
    const size_t X = get_global_id(0);
    const size_t th = get_global_id(1);
    float2 Xth = (float2)(X,th);
    const size_t ID = X +th*PRJ_IMAGESIZE;
    float chi;

    chi = read_imagef(prj_delta_mt_img,s_nearest,Xth).x;
    chi2[ID] += chi*chi;
}

//calculation of chi2 (reduction of projection image pixel)
__kernel void calcChi2_2(__global float *chi2, __local float *loc_mem){
    
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    size_t ID, X, th;
    int i;
    
    loc_mem[local_ID]=0.0f;
    for(i=0;i<PRJ_IMAGESIZE;i+=localsize){
        X = local_ID + i;
        for(th=0;th<SS_ANGLESIZE;th++){
            ID = X + th*PRJ_IMAGESIZE;
            loc_mem[local_ID] += chi2[ID];
        }
    }
    reduction(loc_mem,local_ID,localsize);
    
    chi2[local_ID] = loc_mem[local_ID];
}



//calculate tJJ
//calculate tJdF
//"For loop" for E is conducted at outside kernel
__kernel void calc_tJJ_tJdF(__global float* bprj_jacob, __global float* bprj_delta_mt,
                            __global float* tJJ, __global float* tJdF){
    
    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    const size_t ID = X+Y*IMAGESIZE_X;
    
    float J[PARA_NUM];
    int i,j;
    for(i=0;i<PARA_NUM;i++){
        J[i] = bprj_jacob[ID+i*IMAGESIZE_M];
    }
    float dF = bprj_delta_mt[ID];
    
    //calculate tJJ and tJdF
    for(i=0;i<PARA_NUM;i++){
        tJdF[ID+i*IMAGESIZE_M] += J[i]*dF;
        tJJ[ID+(i+i*PARA_NUM)*IMAGESIZE_M] += J[i]*J[i];
        for(j=i+1;j<PARA_NUM;j++){
            tJJ[ID+(j+i*PARA_NUM)*IMAGESIZE_M] += J[i]*J[j];
            tJJ[ID+(i+j*PARA_NUM)*IMAGESIZE_M] += J[i]*J[j];
        }
    }
}




//calculate new parameter candidate
__kernel void calc_pCandidate(__global float* p_cnd,__global float* tJJ,__global float* tJdF,
                              float lambda, __global float *dL,__constant char *p_fix){
    
    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    const size_t ID = X+Y*IMAGESIZE_X;
    
    float dp[PARA_NUM];
    float fp[PARA_NUM];
    float diag_tJJ[PARA_NUM];
    
    int i,j;
    //copy fp from p_cnd(=p in present)
    //copy dp from tJdF
    //add L-M factor to tJJ
    for(i=0;i<PARA_NUM;i++){
        fp[i] = p_cnd[ID+i*IMAGESIZE_M];
        dp[i] = (p_fix[i]==48) ? 0.0f:tJdF[ID+i*IMAGESIZE_M];
        diag_tJJ[i] = tJJ[ID+(i+i*PARA_NUM)*IMAGESIZE_M];
        tJJ[ID+(i+i*PARA_NUM)*IMAGESIZE_M] *= (1.0f+lambda);
    }
     
    //solve dp by sim. linear eq. [ tJJ x dp = tJdF ]
    sim_linear_eq_global(tJJ,dp,ID,IMAGESIZE_M,PARA_NUM,p_fix);
    
    
    //calc dL
    //calc new parameter candidate
    float dL_pr = 0.0f;
    for(i=0; i<PARA_NUM; i++){
        dL_pr += dp[i]*(dp[i]*lambda*diag_tJJ[i] + tJdF[ID+i*IMAGESIZE_M]);
        p_cnd[ID+i*IMAGESIZE_M] = fp[i]+AART_FACTOR*dp[i];
    }
    dL[ID]=dL_pr;
}

//reduction of dL (1 st time: reduction in X direction, 2nd time reduction in Y direction)
__kernel void calc_dL(__global float *dL, __local float *loc_mem){
    
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Y = get_local_id(1);
    float2 XY;
    int X,i;
    size_t ID;
    
    
    loc_mem[local_ID]=0.0f;
    for(i=0;i<PRJ_IMAGESIZE;i+=localsize){
        X = local_ID + i;
        ID = X + Y*IMAGESIZE_X;
        loc_mem[local_ID] += dL[ID];
    }
    reduction(loc_mem,local_ID,localsize);
    
    dL[Y+local_ID*IMAGESIZE_X] = loc_mem[local_ID];
}


//set constrain to new parameter candidate
__kernel void setConstrain(__global float* p_cnd,__constant float *C_mat, __constant float *D_vec){

    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    const size_t ID = X+Y*IMAGESIZE_X;
    
    float fp[PARA_NUM];
    int i,j;
    for(i=0;i<PARA_NUM;i++){
        fp[i]= p_cnd[ID+i*IMAGESIZE_M];
    }
    
    float eval,C2,h;
    bool eval_b;
    for(j=0;j<CONSTRAIN_NUM;j++){
        eval=0.0f;
        C2 =0.0f;
        for(i=0; i<PARA_NUM; i++){
            eval += C_mat[j*PARA_NUM+i]*fp[i];
            C2 += C_mat[j*PARA_NUM+i]*C_mat[j*PARA_NUM+i];
        }
        eval_b = (eval>D_vec[j]);
        h = (eval-D_vec[j])/sqrt(C2);
        for(i=0; i<PARA_NUM; i++){
            fp[i] = (eval_b) ? fp[i]-h*C_mat[j*PARA_NUM+i]:fp[i];
        }
    }
    
    //copy p_cnd
    for(i=0; i<PARA_NUM; i++){
        p_cnd[ID+i*IMAGESIZE_M] = fp[i];
    }

}



//sinogram correction
__kernel void sinogramCorrection(__read_only image2d_t prj_img_src, __write_only image2d_array_t prj_img_dst, __constant float *angle, int mode, int Enum){
    
    const size_t X = get_global_id(0);
    const size_t th = get_global_id(1);
    
    float2 Xth_f= (float2)(X,th);
    int4 XthE_i= (int4)(X,th,Enum,0);
    float4 img;
            
    //update assumed img
    img = read_imagef(prj_img_src,s_nearest,Xth_f);
    float theta = 2.0f*fabs((float)X/IMAGESIZE_X-0.5f);
    float angle_pr = angle[th]*PI_180;
    switch(mode){
        case 3: //intensity correction for x + theta
            img.x *= cos(asin(theta))*cos(angle_pr);
            break;
        case 2: //intensity correction for theta
            img.x *= cos(angle_pr);
            break;
        case 1: //intensity correction for x
            img.x *= cos(asin(theta));
            break;
        default:
            break;
    }
    write_imagef(prj_img_dst, XthE_i, img);

}

__kernel void circleAttenuator(__global float *p, __constant float *attenuator){
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    const size_t ID = X + Y*IMAGESIZE_X + Z*IMAGESIZE_M;
    
    float p_pr = p[ID];
    float radius = (X-IMAGESIZE_X*0.5f)*(X-IMAGESIZE_X*0.5f) + (Y-IMAGESIZE_Y*0.5f)*(Y-IMAGESIZE_Y*0.5f);
    radius = sqrt(radius);
    p_pr = (radius<=IMAGESIZE_X*0.5f) ? p_pr:p_pr*attenuator[Z];
    p[ID] = p_pr;
}


__kernel void parameterMask(__global float *p, __global float *mask_img,
                            __constant float *attenuator){
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    const size_t ID = X + Y*IMAGESIZE_X + Z*IMAGESIZE_M;
    const size_t IDxy = X + Y*IMAGESIZE_X;
    
    float p_pr = p[ID];
    float mask = mask_img[IDxy];
    p_pr = (attenuator[Z]!=1.0f) ? p_pr*mask*SS/PRJ_IMAGESIZE_M:p_pr;
    p[ID] = p_pr;

}






//assign parameter to Fitting eq. (Energy array ver.)
__kernel void assign2FittingEq_EArray(__global float* p, __write_only image2d_array_t mt_fit_img,
                               __constant float *energyList){
    
    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    const size_t Enum = get_global_id(2);
    const size_t ID = X+Y*IMAGESIZE_X;
    const int4 XYEnum = (int4)(X,Y,Enum,0);
    float energy = energyList[Enum];
    float fp[PARA_NUM];
    int i;
    
    //copy global p to private fp
    for(i=0;i<PARA_NUM;i++){
        fp[i] = p[ID+i*IMAGESIZE_M];
    }
    float mt_fit;
    FIT(energy,mt_fit,fp);
    
    write_imagef(mt_fit_img,XYEnum,(float4)(mt_fit,0.0f,0.0f,1.0f));
}


//assign parameter to Jacobian (Energy array ver.)
__kernel void assign2Jacobian_EArray(__global float* p,__write_only image2d_array_t jacob_img,
                              __constant float *energyList){
    
    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    const size_t Enum = get_global_id(2);
    const size_t ID = X+Y*IMAGESIZE_X;
    const int4 XYPE = (int4)(X,Y,0,0);
    float energy = energyList[Enum];
    float fp[PARA_NUM];
    float J[PARA_NUM];
    int i;
    
    //copy global p to private fp
    for(i=0;i<PARA_NUM;i++){
        fp[i] = p[ID+i*IMAGESIZE_M];
    }
    JACOBIAN(energy,J,fp);
    
    for(i=0;i<PARA_NUM;i++){
        XYPE.z=i+Enum*PARA_NUM;
        write_imagef(jacob_img,XYPE,(float4)(J[i],0.0f,0.0f,1.0f));
    }
}

//projectoion from mt_fit to prj_delta_mt
__kernel void projectionToDeltaMt_Earray(__read_only image2d_array_t mt_fit_img,
                                         __read_only image2d_array_t prj_mt_img,
                                         __write_only image2d_array_t prj_delta_mt_img,
                                         __constant float *anglelist, int sub){
    
    const int X = get_global_id(0);
    const int th = get_global_id(1);
    const int Enum = get_global_id(2);
    
    float4 XYE;
    XYE.z = Enum;
    float4 XthE_f= (float4)(X,sub+th*SS,Enum,0);
    int4 XthE_i = (int4)(X,th,Enum,0);
    float prj =0.0f;
    float angle = anglelist[sub+th*SS]*PI_180;
    
    for(int Y=0;Y<PRJ_IMAGESIZE;Y++){
        XYE.x =  (X-PRJ_IMAGESIZE*0.5f)*cos(angle)+(Y-PRJ_IMAGESIZE*0.5f)*sin(angle) + IMAGESIZE_X*0.5f;
        XYE.y = -(X-PRJ_IMAGESIZE*0.5f)*sin(angle)+(Y-PRJ_IMAGESIZE*0.5f)*cos(angle) + IMAGESIZE_Y*0.5f;
        
        prj += read_imagef(mt_fit_img,s_linear,XYE).x;
    }
    float4 img = (float4)(prj,0.0f,0.0f,1.0f);
    img.x = read_imagef(prj_mt_img,s_linear,XthE_f).x - prj/AMP_FACTOR;
    
    write_imagef(prj_delta_mt_img, XthE_i, img);
}

//calculation of chi2
__kernel void calcChi2_1_EArray(__global float *chi2,
                                __read_only image2d_array_t prj_delta_mt_img){
    
    const size_t X = get_global_id(0);
    const size_t th = get_global_id(1);
    float4 XthE = (float4)(X,th,0.0f,0.0f);
    const size_t ID = X +th*PRJ_IMAGESIZE;
    float chi;
    int Enum;
    
    for(Enum=0;Enum<ENERGY_NUM;Enum++){
        XthE.z=Enum;
        chi = read_imagef(prj_delta_mt_img,s_nearest,XthE).x;
        chi2[ID] += chi*chi;
    }
}


//calculate tJJ
//calculate tJdF
//"For loop" for E is conducted at outside kernel
__kernel void calc_tJJ_tJdF_EArray(__global float* bprj_jacob, __global float* bprj_delta_mt,
                            __global float* tJJ, __global float* tJdF, int Enum){
    
    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    const size_t ID = X+Y*IMAGESIZE_X;
    
    float J[PARA_NUM];
    int i,j;
    
    for(i=0;i<PARA_NUM;i++){
        J[i] = bprj_jacob[ID+i*IMAGESIZE_M];
    }
    float dF = bprj_delta_mt[ID+Enum*IMAGESIZE_M];
    
    //calculate tJJ and tJdF
    for(i=0;i<PARA_NUM;i++){
        tJdF[ID+i*IMAGESIZE_M] += J[i]*dF;
        tJJ[ID+(i+i*PARA_NUM)*IMAGESIZE_M] += J[i]*J[i];
        for(j=i+1;j<PARA_NUM;j++){
            tJJ[ID+(j+i*PARA_NUM)*IMAGESIZE_M] += J[i]*J[j];
            tJJ[ID+(i+j*PARA_NUM)*IMAGESIZE_M] += J[i]*J[j];
        }
    }
}

//assign parameter to Jacobian
__kernel void assign2Jacobian_2(__global float* p,__global float *jacob_buff,
                              __constant float *energyList, int Enum){
    
    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    const size_t ID = X+Y*IMAGESIZE_X;
    float energy = energyList[Enum];
    float fp[PARA_NUM];
    float J[PARA_NUM];
    int i;
    
    //copy global p to private fp
    for(i=0;i<PARA_NUM;i++){
        fp[i] = p[ID+i*IMAGESIZE_M];
    }
    JACOBIAN(energy,J,fp);
    
    for(i=0;i<PARA_NUM;i++){
        jacob_buff[ID+i*IMAGESIZE_M] = J[i];
    }
}
