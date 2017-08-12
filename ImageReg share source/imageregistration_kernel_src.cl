#define PI 3.14159265358979323846f
#define PI_2 1.57079632679489661923f

#ifndef IMAGESIZE_X
#define IMAGESIZE_X 2048
#endif

#ifndef IMAGESIZE_Y
#define IMAGESIZE_Y 2048
#endif

#ifndef IMAGESIZE_M
#define IMAGESIZE_M 4194304 //2048*2048
#endif

#ifndef REGMODE
#define REGMODE 0
#endif

#ifndef PARA_NUM
#define PARA_NUM 2
#endif


#ifndef DIFFSTEP
#define DIFFSTEP 2
#endif



//sampler
__constant sampler_t s_linear = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;
__constant sampler_t s_linear_cEdge = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE;


//reduction
inline static float reduction(__local float *loc_mem, const size_t local_ID, const size_t localsize)
{
    for(size_t s=localsize/2;s>0;s>>=1){
        if(local_ID<s){
            loc_mem[local_ID]+=loc_mem[local_ID+s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    return loc_mem[local_ID];
}


inline float4 XYshift(float4 XYZ, float* p, float mergeN){
    return (float4)(XYZ.x+p[0]/mergeN, XYZ.y+p[1]/mergeN, XYZ.z, 0.0f);
}


inline float4 Rot(float4 XYZ, float* p, float mergeN){
    return (float4)(cos(p[2])*XYZ.x - sin(p[2])*XYZ.y + p[0]/mergeN,
                    sin(p[2])*XYZ.x + cos(p[2])*XYZ.y + p[1]/mergeN, XYZ.z, 0.0f);
}


inline float4 Scale(float4 XYZ, float* p, float mergeN){
    return (float4)(exp(p[2])*XYZ.x + p[0]/mergeN, exp(p[2])*XYZ.y + p[1]/mergeN, XYZ.z, 0.0f);
}


inline float4 RotScale(float4 XYZ, float* p, float mergeN){
    return (float4)((cos(p[2])*XYZ.x - sin(p[2])*XYZ.y)*exp(p[3]) + p[0]/mergeN,
                    (sin(p[2])*XYZ.x + cos(p[2])*XYZ.y)*exp(p[3]) + p[1]/mergeN, XYZ.z, 0.0f);
}


inline float4 Affine(float4 XYZ, float* p, float mergeN){
    return (float4)((1.0f + p[2])*XYZ.x - p[3]*XYZ.y + p[0]/mergeN,
                    p[4]*XYZ.x + (1.0f + p[5])*XYZ.y + p[1]/mergeN, XYZ.z, 0.0f);
}


inline float4 transXY(float4 XYZ, float* p, float mergeN, int transMode){
    switch(transMode){
        case 0: //XY shift
            return XYshift(XYZ,p,mergeN);
        case 1: //rotation + XY shift
            return Rot(XYZ,p,mergeN);
        case 2: //scale + XY shift
            return Scale(XYZ,p,mergeN);
        case 3: //rotation + scale + XY shift
            return RotScale(XYZ,p,mergeN);
        case 4: //affine + XY shift
            return Affine(XYZ,p,mergeN);
    }
}


inline void Jacobian_XYshift(float* J,float4 XYZ,float* p,float dfdx,float dfdy,float mask,float mergeN){
    J[0] = dfdx*mask/mergeN;
    J[1] = dfdy*mask/mergeN;
}


inline void Jacobian_Rot(float* J,float4 XYZ,float* p,float dfdx,float dfdy,float mask,float mergeN){
    
    float DxDth = -XYZ.x*sin(p[2])-XYZ.y*cos(p[2]);
    float DyDth =  XYZ.x*cos(p[2])-XYZ.y*sin(p[2]);
    
    J[0] = dfdx*mask/mergeN;
    J[1] = dfdy*mask/mergeN;
    J[2] = (dfdx*DxDth+dfdy*DyDth)*mask;
}


inline void Jacobian_Scale(float* J,float4 XYZ,float* p,float dfdx,float dfdy,float mask,float mergeN){
    J[0] = dfdx*mask/mergeN;
    J[1] = dfdy*mask/mergeN;
    J[2] = exp(p[2])*(dfdx*XYZ.x+dfdy*XYZ.y)*mask;
}


inline void Jacobian_RotScale(float* J,float4 XYZ,float* p,float dfdx,float dfdy,float mask,float mergeN){
    
    float g = p[3];
    float DxDth = (-XYZ.x*sin(p[2])-XYZ.y*cos(p[2]))*exp(g);
    float DyDth = ( XYZ.x*cos(p[2])-XYZ.y*sin(p[2]))*exp(g);
    float DxDs  = ( XYZ.x*cos(p[2])-XYZ.y*sin(p[2]))*exp(g);
    float DyDs  = ( XYZ.x*sin(p[2])+XYZ.y*cos(p[2]))*exp(g);
    
    J[0] = dfdx*mask/mergeN;
    J[1] = dfdy*mask/mergeN;
    J[2] = (dfdx*DxDth+dfdy*DxDth)*mask;
    J[3] = (dfdx*DxDs+dfdy*DyDs)*mask;
}


inline void Jacobian_Affine(float* J,float4 XYZ,float* p,float dfdx,float dfdy,float mask,float mergeN){
    J[0] = dfdx*mask/mergeN;
    J[1] = dfdy*mask/mergeN;
    J[2] = dfdx*XYZ.x*mask;
    J[3] = dfdx*XYZ.y*mask;
    J[4] = dfdy*XYZ.x*mask;
    J[5] = dfdy*XYZ.y*mask;
}


inline void Jacobian_transXY(float* J,float4 XYZ,float* p,float dfdx,float dfdy,float mask,float mergeN,int transMode){
    switch(transMode){
        case 0: //XY shift
            Jacobian_XYshift(J,XYZ,p,dfdx,dfdy,mask,mergeN);
            break;
        case 1: //rotation + XY shift
            Jacobian_Rot(J,XYZ,p,dfdx,dfdy,mask,mergeN);
            break;
        case 2: //scale + XY shift
            Jacobian_Scale(J,XYZ,p,dfdx,dfdy,mask,mergeN);
            break;
        case 3: //rotation + scale + XY shift
            Jacobian_RotScale(J,XYZ,p,dfdx,dfdy,mask,mergeN);
            break;
        case 4: //affine + XY shift
            Jacobian_Affine(J,XYZ,p,dfdx,dfdy,mask,mergeN);
            break;
    }
}


//kernel
//mt conversion
__kernel void mt_conversion(__global float *dark, __global float *I0,
                            __global ushort *It_buffer, __global float *mt_buffer,
                            __write_only image2d_array_t mt_img1,
                            __write_only image2d_array_t mt_img2,
                            int shapeNo,int startpntX, int startpntY,
                            uint width, uint height, float angle, int evaluatemode){
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_group_id(0);
    const size_t Y = get_global_id(1);
    int X,ID,IDxy;
    int4 XYZ;
    float4 mt_f1,mt_f2;
    float mask=1.0f;
    float radius2=0;
    float trans, mt, It, dark1;
    float absX, absY;
    
    
    for(size_t i=0;i<IMAGESIZE_X/localsize;i++){
        X=local_ID+i*localsize;
        ID=X+IMAGESIZE_X*Y+Z*IMAGESIZE_M;
        IDxy=X+IMAGESIZE_X*Y;
        XYZ=(int4)(X,Y,Z,0);
        
        absX = (X-startpntX)*cos(angle/180*(float)PI)-(Y-startpntY)*sin(angle/180*(float)PI);
        absX = fabs(absX);
        absY = (X-startpntX)*sin(angle/180*(float)PI)+(Y-startpntY)*cos(angle/180*(float)PI);
        absY = fabs(absY);
        switch(shapeNo){
            case 0: //square or rectangle
                mask=(absX<=width/2 & absY<=height/2) ? 1.0f:0.0f;
                break;
                
            case 1: //circle or orval
                radius2=(float)(absX*absX)/width/width+(float)(absY*absY)/height/height;
                mask=(radius2<=0.25f) ? 1.0f:0.0f;
                break;
                
            default:
                mask=1.0f;
                break;
        }
        
        It = (float)It_buffer[ID+32*Z];
        dark1 = dark[IDxy];
        trans = (I0[IDxy]-dark1)/(It-dark1);
        trans = (trans < 1.0E-5f) ? 1.0E-5f:trans;
        trans = (trans > 1.0E7f) ? 1.0E7f:trans;
        trans = isnan(trans) ? 1.0f:trans;
        mt = log(trans);
        
        switch(evaluatemode){
            case 0: //mt
                mt_f1 = (float4)(mt,mask,0.0f,0.0f);
                break;
                
            case 1: // trans^(-1)-1
                mt_f1 = (float4)(1/trans-1,mask,0.0f,0.0f);
                break;
                
            case 2: // It
                mt_f1 = (float4)(1/It,mask,0.0f,0.0f);
                break;
                
            default: //mt
                mt_f1 = (float4)(mt,mask,0.0f,0.0f);
                break;
        }
        mt_f1 = (float4)(mt,mask,0.0f,0.0f);
        mt_f2 = (float4)(mt,1.0f,0.0f,0.0f);
        write_imagef(mt_img1,XYZ,mt_f1);
        write_imagef(mt_img2,XYZ,mt_f2);
        mt_buffer[ID]=mt;
    }
    
}


//mt transfer
__kernel void mt_transfer(__global float *mt_buffer,
                          __write_only image2d_array_t mt_img1,
                          __write_only image2d_array_t mt_img2,
                          int shapeNo,int startpntX, int startpntY,
                          uint width, uint height, float angle){
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const int Z = get_group_id(0);
    const int Y = get_global_id(1);
    int X,ID;
    int4 XYZ;
    float4 mt_f1,mt_f2;
    float mask=1.0f;
    float radius2=0.0f;
    float mt;
    float absX, absY;
    float angle_rad = angle/180.0f*PI;
    
    
    for(size_t i=0;i<IMAGESIZE_X/localsize;i++){
        X=local_ID+i*localsize;
        ID=X+IMAGESIZE_X*Y+Z*IMAGESIZE_M;
        XYZ=(int4)(X,Y,Z,0);
        
        absX = (float)(X-startpntX)*cos(angle_rad)-(float)(Y-startpntY)*sin(angle_rad);
        absX = fabs(absX);
        absY = (float)(X-startpntX)*sin(angle_rad)+(float)(Y-startpntY)*cos(angle_rad);
        absY = fabs(absY);
        switch(shapeNo){
            case 0: //square or rectangle
                mask=(absX<=width/2 & absY<=height/2) ? 1.0f:0.0f;
                break;
                
            case 1: //circle or orval
                radius2=(float)(absX*absX)/width/width+(float)(absY*absY)/height/height;
                mask=(radius2<=0.25f) ? 1.0f:0.0f;
                break;
                
            default:
                mask=1.0f;
                break;
        }
        
        mt = (float)mt_buffer[ID];
        mt_f1 = (float4)(mt,mask,0.0f,0.0f);
        mt_f2 = (float4)(mt,1.0f,0.0f,0.0f);
        write_imagef(mt_img1,XYZ,mt_f1);
        write_imagef(mt_img2,XYZ,mt_f2);
        //mt_buffer[ID]=mt;
    }
    
}


//create merged image
__kernel void merge(__read_only image2d_array_t input_img, __write_only image2d_array_t output_img,
                    unsigned int mergeN){
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t group_ID = get_group_id(0);
    
    float4 img;
    float4 XYZ_in;
    int4 XYZ_out;
    int X, Y;
    
    for(size_t i=0;i<IMAGESIZE_Y/mergeN;i++){
        for(size_t j=0;j<IMAGESIZE_X/mergeN/localsize;j++){
            X = local_ID+j*localsize;
            Y = i;
            img = (float4)(0.0f,0.0f,0.0f,0.0f);
            XYZ_out = (int4)(X,Y,group_ID,0);
            for(size_t k=0; k<mergeN; k++){
                for(size_t l=0; l<mergeN; l++){
                    XYZ_in = (float4)(k+X*mergeN,l+Y*mergeN,group_ID,0.0f);
                    img += read_imagef(input_img,s_linear,XYZ_in);
                }
            }
            //if(img.y>1) printf("X%dY%d: %f,%f \n",X,Y,img.x,img.y);
            img /= mergeN*mergeN;
            img.y = (img.y>0.5f) ? 1.0f:0.0f;
            write_imagef(output_img,XYZ_out,img);
        }
    }
}


//imageReg1: estimate dF2(old), tJJ, tJdF
__kernel void imageReg1(__read_only image2d_array_t mt_t_img,__read_only image2d_array_t mt_s_img,
                        __constant float *p,__constant float *p_target,__constant char *p_fix,
                        __global float *dF2, __global float *tJdF, __global float *tJJ,
                        int mergeN, __local float* loc_mem){
    
    const size_t local_ID = get_local_id(0);
    const size_t globalsize = get_global_size(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_group_id(0);
    const size_t Zsize = globalsize/localsize;
    
    
    //copy data from global p to private p_pr
    float p_pr[PARA_NUM], p_target_pr[PARA_NUM];
    for(int i=0;i<PARA_NUM;i++){
        p_pr[i]=p[Z+i*Zsize];
        p_target_pr[i]=p_target[Z+i*Zsize];
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    
    size_t X,Y;
    float4 XYZ, XYZ_s, XYZ_t;
    float4 mt_s, mt_t;
    float mt_xp, mt_xm, mt_yp, mt_ym;
    float diffweight = 3.0f/(float)DIFFSTEP/(DIFFSTEP+1.0f)/(2.0f*DIFFSTEP+1.0f)/(float)mergeN;
    float dfdx, dfdy;
    float mask;
    float J[PARA_NUM];
    float dF, dF2_pr=0.0f;
    float tJdF_pr[PARA_NUM];
    float tJJ_pr[PARA_NUM*(PARA_NUM+1)/2];
    float dev=0.0f;
    
    
    //initailize tJJ,tJdF
    for(int i=0; i<PARA_NUM; i++){
        tJdF_pr[i]=0.0f;
    }
    for(int i=0; i<PARA_NUM*(PARA_NUM+1)/2; i++){
        tJJ_pr[i]=0.0f;
    }
    
    
    //estimate tJJ, tJdF, dF2
    for(int i=0;i<IMAGESIZE_Y/mergeN;i++){
        Y=i;
        for(int j=0; j<IMAGESIZE_X/mergeN; j+=localsize){
            X=local_ID+j*localsize;
            
            
            mt_xp = 0.0f;
            mt_xm = 0.0f;
            mt_yp = 0.0f;
            mt_ym = 0.0f;
            for(int k=1;k<=DIFFSTEP;k++){
                //Partial differential dF/dx
                XYZ = (float4)(X+k,Y,Z,0.0f);
                XYZ_s = k*transXY(XYZ, p_pr, (float)mergeN, REGMODE);
                mt_xp += read_imagef(mt_s_img,s_linear_cEdge,XYZ_s).x;
                XYZ = (float4)(X-k,Y,Z,0.0f);
                XYZ_s = transXY(XYZ, p_pr, (float)mergeN, REGMODE);
                mt_xm += k*read_imagef(mt_s_img,s_linear_cEdge,XYZ_s).x;
                
                //Partial differential dF/dy
                XYZ = (float4)(X,Y+k,Z,0.0f);
                XYZ_s = transXY(XYZ, p_pr, (float)mergeN, REGMODE);
                mt_yp += k*read_imagef(mt_s_img,s_linear_cEdge,XYZ_s).x;
                XYZ = (float4)(X,Y-k,Z,0.0f);
                XYZ_s = transXY(XYZ, p_pr, (float)mergeN, REGMODE);
                mt_ym += k*read_imagef(mt_s_img,s_linear_cEdge,XYZ_s).x;
            }
            dfdx = (mt_xp - mt_xm)*diffweight;
            dfdy = (mt_yp - mt_ym)*diffweight;
            
            
            //dF
            XYZ = (float4)(X,Y,Z,0.0f);
            XYZ_s = transXY(XYZ, p_pr, (float)mergeN, REGMODE);
            XYZ_t = transXY(XYZ, p_target_pr, (float)mergeN, REGMODE);
            mt_s = read_imagef(mt_s_img,s_linear_cEdge,XYZ_s);
            mt_t = read_imagef(mt_t_img,s_linear_cEdge,XYZ_t);
            mask = mt_t.y*mt_s.y;
            dF = (mt_t.x-mt_s.x)*mask;
            
            
            //Jacobian
            Jacobian_transXY(J,XYZ_s,p_pr,dfdx,dfdy,mask,(float)mergeN,REGMODE);
            
            
            dev += mask;
            dF2_pr += dF*dF;
            int k=0;
            for(int n=0;n<PARA_NUM;n++){
                if(p_fix[n]==48) continue;
                tJdF_pr[n] += J[n]*dF;
                for(int m=n; m<PARA_NUM;m++){
                    if(p_fix[m]==48) continue;
                    tJJ_pr[k] += J[n]*J[m];
                    k++;
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    //reduction of dev
    loc_mem[local_ID] = dev;
    barrier(CLK_LOCAL_MEM_FENCE);
    dev = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    //reduction of dF2
    loc_mem[local_ID] = dF2_pr;
    barrier(CLK_LOCAL_MEM_FENCE);
    dF2_pr = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    //reduction of tJdF
    for(int i=0;i<PARA_NUM;i++){
        loc_mem[local_ID] = tJdF_pr[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        tJdF_pr[i] = reduction(loc_mem,local_ID,localsize);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //reduction of tJJ
    for(int i=0;i<PARA_NUM*(PARA_NUM+1)/2;i++){
        loc_mem[local_ID] = tJJ_pr[i];
        barrier(CLK_LOCAL_MEM_FENCE);
        tJJ_pr[i] = reduction(loc_mem,local_ID,localsize);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    
    //output dF2, tJJ, tJdF to global memory
    if(local_ID==0){
        dF2[Z] = dF2_pr/dev;
        for(int i=0;i<PARA_NUM;i++){
            tJdF[Z+i*Zsize] = tJdF_pr[i]/dev;
        }
        for(int i=0;i<PARA_NUM*(PARA_NUM+1)/2;i++){
            tJJ[Z+i*Zsize] = tJJ_pr[i]/dev;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
}



//imageReg2: estimate dF2(new)
__kernel void imageReg2(__read_only image2d_array_t mt_t_img,__read_only image2d_array_t mt_s_img,
                        __constant float *p,__constant float *p_target,
                        __global float *dF2, int mergeN, __local float* loc_mem){
    
    const size_t local_ID = get_local_id(0);
    const size_t globalsize = get_global_size(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_group_id(0);
    const size_t Zsize = globalsize/localsize;
    
    
    //copy data from global p to private p_pr
    float p_pr[PARA_NUM],p_target_pr[PARA_NUM];
    for(int i=0;i<PARA_NUM;i++){
        p_pr[i]=p[Z+i*Zsize];
        p_target_pr[i]=p_target[Z+i*Zsize];
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    
    size_t X,Y;
    float4 XYZ, XYZ_s, XYZ_t;
    float4 mt_s, mt_t;
    float mask;
    float dF, dF2_pr=0.0f;
    float dev=0.0f;
    
    //estimate tJJ, tJdF, dF2
    for(int i=0;i<IMAGESIZE_Y/mergeN;i++){
        Y=i;
        for(int j=0; j<IMAGESIZE_X/mergeN; j+=localsize){
            X=local_ID+j*localsize;
            
            //dF
            XYZ = (float4)(X,Y,Z,0.0f);
            XYZ_s = transXY(XYZ, p_pr, (float)mergeN, REGMODE);
            XYZ_t = transXY(XYZ, p_target_pr, (float)mergeN, REGMODE);
            mt_s = read_imagef(mt_s_img,s_linear_cEdge,XYZ_s);
            mt_t = read_imagef(mt_t_img,s_linear_cEdge,XYZ_t);
            mask = mt_t.y*mt_s.y;
            dF = (mt_t.x-mt_s.x)*mask;
            dF2_pr += dF*dF;
            
            dev += mask;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    //reduction of dev
    loc_mem[local_ID] = dev;
    barrier(CLK_LOCAL_MEM_FENCE);
    dev = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    //reduction of dF2
    loc_mem[local_ID] = dF2_pr;
    barrier(CLK_LOCAL_MEM_FENCE);
    dF2_pr = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    //output dF2 to global memory
    if(local_ID==0){
        dF2[Z] = dF2_pr/dev;
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
}


__kernel void estimateParaError(__global float* p_error, __global float* tJJ){
    const size_t X = get_global_id(0); //angle dimension
    const size_t Y = get_global_id(1);
    const size_t Z = get_global_id(2); //parameter dimension
    const size_t sizeX = get_global_size(0);
    const size_t sizeY = get_global_size(1);
    const size_t ID = X + Y*sizeX;
    const size_t ID1 = ID + Z*sizeX*sizeY;
    const size_t ID2 = ID + Z*(2*PARA_NUM-Z+1)/2*sizeX*sizeY;
    
    
    p_error[ID1] = fmax(sqrt(1.0f/tJJ[ID2]),0.01f);
}


//output final image reg results
__kernel void output_imgReg_result(__read_only image2d_array_t mt_img, __global float *mt_buf,
                                   __global float *p){
    
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_group_id(0);
    const size_t Y = get_global_id(1);
    const size_t globalsize = get_global_size(0);
    const size_t Zsize = globalsize/localsize;
    
    size_t X,ID;
    float4 XYZ, XYZ_s, img;
    float p_pr[PARA_NUM];
    
    
    //copy data from global transpara to local transpara_atE
    for(size_t i=0;i<PARA_NUM;i++){
        p_pr[i]=p[Z+i*Zsize];
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    }
    
    
    //convert image reg results
    for(size_t i=0;i<IMAGESIZE_X/localsize;i++){
        X = local_ID+i*localsize;
        XYZ = (float4)(X,Y,Z,0.0f);
        XYZ_s = transXY(XYZ, p_pr, 1, REGMODE);
        ID = X+Y*IMAGESIZE_X+Z*IMAGESIZE_M;
        
        img = read_imagef(mt_img,s_linear,XYZ_s);
        mt_buf[ID]=img.x*img.y;
    }
}


//merge image reg images
__kernel void merge_mt(__read_only image2d_array_t mt_sample, __global float *mt_output)
{
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t globalsize_x = get_global_size(0);
    const size_t global_ID = global_x+globalsize_x*global_y;
    float4 XYZ;
    float mt=0;
    
    
    const size_t mergesize = get_image_array_size(mt_sample);
    for(size_t i=0;i<mergesize;i++){
        XYZ=(float4)(global_x,global_y,i,0);
        mt+=read_imagef(mt_sample,s_linear,XYZ).x;
    }
    mt_output[global_ID]=mt;
}


//merge raw his data
__kernel void merge_rawhisdata(__global ushort *rawhisdata, __global float *outputdata,
                               int mergeN) {
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t globalsize_x = get_global_size(0);
    const size_t global_ID = global_x+globalsize_x*global_y;
    
    float value = 0.0f;
    for(int i=0;i<mergeN;i++){
        value += (float)rawhisdata[global_ID+(IMAGESIZE_M+32)*i];
    }
    value /= mergeN;
    outputdata[global_ID]=value;
}


//QXAFS smoothing
__kernel void imQXAFS_smoothing(__global float *rawmtdata, __global float *outputdata,
                                int mergeN) {
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t globalsize_x = get_global_size(0);
    const size_t globalsize_y = get_global_size(1);
    const size_t globalsize_z = get_global_size(2);
    const size_t global_ID = global_x+globalsize_x*global_y+globalsize_x*globalsize_y*global_z;
    const size_t imagesize = globalsize_x*globalsize_y*globalsize_z;
    
    float value = 0.0f;
    for(int i=0;i<mergeN;i++){
        value += rawmtdata[global_ID+imagesize*i];
    }
    value /= mergeN;
    outputdata[global_ID]=value;
}

