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
#define PARA_NUM 4
#endif


//sampler
__constant sampler_t s_linear = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP;
__constant sampler_t s_linear_cEdge = CLK_FILTER_LINEAR|CLK_NORMALIZED_COORDS_FALSE|CLK_ADDRESS_CLAMP_TO_EDGE;


//reduction
inline static float reduction(__local float *loc_mem, const size_t local_ID, const size_t localsize)
{
    for(size_t s=localsize;s>0;s>>=1){
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
                            uint width, uint height, float angle){
    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    const size_t Z = get_global_id(2);
    const size_t Zoffset = get_global_offset(2);
    
    
    
    int ID,IDxy,IDIt;
    int4 XYZ;
    float4 mt_f1,mt_f2;
    float mask=1.0f;
    float radius2=0;
    float trans, mt, It, dark1;
    float absX, absY;
    
    
    IDxy = X + IMAGESIZE_X*Y;
    IDIt = IDxy + (Z-Zoffset)*(IMAGESIZE_M+32);
    ID   = IDxy + Z*IMAGESIZE_M;
    XYZ=(int4)(X,Y,Z,0);
    
    //mask
    absX = (X-startpntX)*cos(angle/180.0f*(float)PI)-(Y-startpntY)*sin(angle/180.0f*(float)PI);
    absX = fabs(absX);
    absY = (X-startpntX)*sin(angle/180.0f*(float)PI)+(Y-startpntY)*cos(angle/180.0f*(float)PI);
    absY = fabs(absY);
    switch(shapeNo){
        case 0: //square or rectangle
            mask=(absX<=width/2.0f & absY<=height/2.0f) ? 1.0f:0.0f;
            break;
                
        case 1: //circle or orval
            radius2=(float)(absX*absX)/width/width+(float)(absY*absY)/height/height;
            mask=(radius2<=0.25f) ? 1.0f:0.0f;
            break;
                
        default:
            mask=1.0f;
            break;
    }
        
    It = (float)It_buffer[IDIt];
    dark1 = dark[IDxy];
    trans = (I0[IDxy]-dark1)/(It-dark1);
    trans = (trans < 1.0E-5f) ? 1.0E-5f:trans;
    trans = (trans > 1.0E7f) ? 1.0E7f:trans;
    trans = isnan(trans) ? 1.0f:trans;
    mt = log(trans);
    mt = isnan(mt) ? 0.0f:mt;
    
    mt_f1 = (float4)(mt,mask,0.0f,0.0f);
    mt_f2 = (float4)(mt,1.0f,0.0f,0.0f);
    write_imagef(mt_img1,XYZ,mt_f1);
    write_imagef(mt_img2,XYZ,mt_f2);
    mt_buffer[ID]=mt;

}


__kernel void mt_conversion_binning(__global float *dark_buffer, __global float *I0_buffer,
                                    __global ushort *It_buffer, __global float *mt_buffer,
                                    __write_only image2d_array_t mt_img1,
                                    __write_only image2d_array_t mt_img2,
                                    int shapeNo,int startpntX, int startpntY,
                                    uint width, uint height, float angle, int binningN){
    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    const size_t Z = get_global_id(2);
    const size_t Zoffset = get_global_offset(2);
    
    
    
    int ID,IDxy,IDIt;
    int4 XYZ;
    float4 mt_f1,mt_f2;
    float mask=1.0f;
    float radius2=0;
    float trans, mt, It, I0, dark;
    float absX, absY;
    
    
    IDxy = X + IMAGESIZE_X*Y;
    IDIt = IDxy + (Z-Zoffset)*(IMAGESIZE_M+32);
    ID   = IDxy + Z*IMAGESIZE_M;
    XYZ=(int4)(X,Y,Z,0);
    
    //mask
    absX = (X-startpntX)*cos(angle/180.0f*(float)PI)-(Y-startpntY)*sin(angle/180.0f*(float)PI);
    absX = fabs(absX);
    absY = (X-startpntX)*sin(angle/180.0f*(float)PI)+(Y-startpntY)*cos(angle/180.0f*(float)PI);
    absY = fabs(absY);
    switch(shapeNo){
        case 0: //square or rectangle
            mask=(absX<=width/2.0f & absY<=height/2.0f) ? 1.0f:0.0f;
            break;
            
        case 1: //circle or orval
            radius2=(float)(absX*absX)/width/width+(float)(absY*absY)/height/height;
            mask=(radius2<=0.25f) ? 1.0f:0.0f;
            break;
            
        default:
            mask=1.0f;
            break;
    }
    
    It=0.0f;
    I0=0.0f;
    for(int i=0;i<binningN;i++){
        for(int j=0;j<binningN;j++){
            IDxy = (X*binningN+j) + IMAGESIZE_X*(Y*binningN+i);
            IDIt = IDxy + (Z-Zoffset)*(IMAGESIZE_M+32);
            ID   = IDxy + Z*IMAGESIZE_M;
            XYZ=(int4)(X,Y,Z,0);
            
            dark = dark_buffer[IDxy];
            I0 += I0_buffer[IDxy] - dark;
            It += (float)It_buffer[IDIt] - dark;
        }
    }

    trans = I0/It;
    trans = (trans < 1.0E-5f) ? 1.0E-5f:trans;
    trans = (trans > 1.0E7f) ? 1.0E7f:trans;
    trans = isnan(trans) ? 1.0f:trans;
    mt = log(trans);
    mt = isnan(mt) ? 0.0f:mt;
    
    mt_f1 = (float4)(mt,mask,0.0f,0.0f);
    mt_f2 = (float4)(mt,1.0f,0.0f,0.0f);
    write_imagef(mt_img1,XYZ,mt_f1);
    write_imagef(mt_img2,XYZ,mt_f2);
    mt_buffer[ID]=mt;
    
}

//mt transfer
__kernel void mt_transfer(__global float *mt_buffer,
                          __write_only image2d_array_t mt_img1,
                          __write_only image2d_array_t mt_img2,
                          int shapeNo,int startpntX, int startpntY,
                          uint width, uint height, float angle){
    const int X = get_global_id(0);
    const int Y = get_global_id(1);
    const int Z = get_global_id(2);
    
    int ID;
    int4 XYZ;
    float4 mt_f1,mt_f2;
    float mask=1.0f;
    float radius2=0.0f;
    float mt;
    float absX, absY;
    float angle_rad = angle/180.0f*PI;
    
    
    ID=X+IMAGESIZE_X*Y+Z*IMAGESIZE_M;
    XYZ=(int4)(X,Y,Z,0);
    
    //mask
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
}


//create merged image (mean)
__kernel void merge(__read_only image2d_array_t input_img, __write_only image2d_array_t output_img,
                    unsigned int mergeN){
    
    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    const size_t Z = get_global_id(2);
    
    float4 img;
    float4 XYZ_in;
    int4 XYZ_out;
    
    img = (float4)(0.0f,0.0f,0.0f,0.0f);
    XYZ_out = (int4)(X,Y,Z,0);
    for(size_t k=0; k<mergeN; k++){
        for(size_t l=0; l<mergeN; l++){
            XYZ_in = (float4)(k+X*mergeN,l+Y*mergeN,Z,0.0f);
            img += read_imagef(input_img,s_linear,XYZ_in);
        }
    }
    //if(img.y>1) printf("X%dY%d: %f,%f \n",X,Y,img.x,img.y);
    img /= mergeN*mergeN;
    img.y = (img.y>0.5f) ? 1.0f:0.0f;
    write_imagef(output_img,XYZ_out,img);
    
}


//imageReg1: estimate dF2(old), tJJ, tJdF
__kernel void imageReg1X(__read_only image2d_array_t mt_t_img,__read_only image2d_array_t mt_s_img,
                         __write_only image2d_array_t weight_img,
                         __constant float* dF2, __constant float* dF, float CI,
                         __constant float *p,__constant float *p_target,__constant char *p_fix,
                         __global float *dF2X, __global float *tJdFX, __global float *tJJX,
                         __global float* devX, int mergeN, __local float* loc_mem, int difstep){
    
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Y = get_global_id(1);
    const size_t Z = get_global_id(2);
    const size_t Zsize = get_global_size(2);
    
    
    //copy data from global p to private p_pr
    float p_pr[PARA_NUM], p_target_pr[PARA_NUM];
    for(int i=0;i<PARA_NUM;i++){
        p_pr[i]=p[Z+i*Zsize];
        p_target_pr[i]=p_target[Z+i*Zsize];
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    
    size_t X;
    float4 XYZ, XYZ_s, XYZ_t;
    int4 XYZ_i;
    float4 mt_s, mt_t;
    float Smtx, Smty, Sxx, Syy;
    float dfdx, dfdy;
    float mask;
    float J[PARA_NUM];
    float dF_pr;
    float dF2_pr[2];
    float tJdF_pr[PARA_NUM*2];
    float tJJ_pr[PARA_NUM*(PARA_NUM+1)];
    float dev[2];
    float weight;
    float weight_thd = max(0.5f,fabs(dF[Z]) + sqrt(max(0.0f,dF2[Z] - dF[Z]*dF[Z]))*CI);
    
    dF2_pr[0]=0.0f;
    dF2_pr[1]=0.0f;
    dev[0]=0.0f;
    dev[1]=0.0f;
    
    //initailize tJJ,tJdF
    for(int i=0; i<PARA_NUM*2; i++){
        tJdF_pr[i]=0.0f;
    }
    for(int i=0; i<PARA_NUM*(PARA_NUM+1); i++){
        tJJ_pr[i]=0.0f;
    }
    
    
    //estimate tJJ, tJdF, dF2
    for(int j=0; j<IMAGESIZE_X/mergeN; j+=2*localsize){
        for(int i=0;i<2;i++){
            X=local_ID + i*localsize + j;
            
            Smtx = 0.0f;
            Smty = 0.0f;
            Sxx  = 0.0f;
            Syy  = 0.0f;
            for(int k=-difstep;k<=difstep;k++){
                //Partial differential dF/dx
                XYZ = (float4)(X+k,Y,Z,0.0f);
                XYZ_s = transXY(XYZ, p_pr, (float)mergeN, REGMODE);
                Smtx += k*read_imagef(mt_s_img,s_linear_cEdge,XYZ_s).x;
                Sxx  += k*k;
                
                //Partial differential dF/dy
                XYZ = (float4)(X,Y+k,Z,0.0f);
                XYZ_s = transXY(XYZ, p_pr, (float)mergeN, REGMODE);
                Smty += k*read_imagef(mt_s_img,s_linear_cEdge,XYZ_s).x;
                Syy  += k*k;
            }
            dfdx = Smtx/Sxx*exp(p_pr[PARA_NUM-2]);
            dfdy = Smty/Syy*exp(p_pr[PARA_NUM-2]);
            
            
            //dF
            XYZ = (float4)(X,Y,Z,0.0f);
            XYZ_s = transXY(XYZ, p_pr, (float)mergeN, REGMODE);
            XYZ_t = transXY(XYZ, p_target_pr, (float)mergeN, REGMODE);
            mt_s = read_imagef(mt_s_img,s_linear_cEdge,XYZ_s);
            mt_t = read_imagef(mt_t_img,s_linear_cEdge,XYZ_t);
            mask = mt_t.y*mt_s.y;
            dF_pr = (mt_t.x - (mt_s.x+p_pr[PARA_NUM-1])*exp(p_pr[PARA_NUM-2]))*mask;
            
            //weight
            XYZ_i = (int4)(X,Y,Z,0);
            weight = dF_pr/weight_thd;
            //weight = (fabs(weight)<=1.0f) ? (1.0f - weight*weight):0.0f;
            //weight = weight*weight*mask;
            weight = exp(-weight*weight)*mask;
            write_imagef(weight_img,XYZ_i,(float4)(weight,0.0f,0.0f,0.0f));
            
            //Jacobian
            Jacobian_transXY(J,XYZ_s,p_pr,dfdx,dfdy,mask,(float)mergeN,REGMODE);
            J[PARA_NUM-2] = (mt_s.x+p_pr[PARA_NUM-1])*exp(p_pr[PARA_NUM-2])*mask;
            J[PARA_NUM-1] = exp(p_pr[PARA_NUM-2])*mask;
            
            
            dev[i] += weight;
            dF2_pr[i] += weight*dF_pr*dF_pr;
            for(int n=0;n<PARA_NUM;n++){
                if(p_fix[n]==48) continue;
                tJdF_pr[n + i*PARA_NUM] += weight*J[n]*dF_pr;
                for(int m=n; m<PARA_NUM;m++){
                    if(p_fix[m]==48) continue;
                    tJJ_pr[n*PARA_NUM-n*(n+1)/2+m + i*PARA_NUM*(PARA_NUM+1)/2] += weight*J[n]*J[m];
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    
    
    //reduction of dev
    loc_mem[local_ID] = dev[0];
    loc_mem[local_ID+localsize] = dev[1];
    barrier(CLK_LOCAL_MEM_FENCE);
    dev[0] = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    //reduction of dF2
    loc_mem[local_ID] = dF2_pr[0];
    loc_mem[local_ID+localsize] = dF2_pr[1];
    barrier(CLK_LOCAL_MEM_FENCE);
    dF2_pr[0] = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    //reduction of tJdF
    for(int i=0;i<PARA_NUM;i++){
        loc_mem[local_ID] = tJdF_pr[i];
        loc_mem[local_ID+localsize] = tJdF_pr[i + PARA_NUM];
        barrier(CLK_LOCAL_MEM_FENCE);
        tJdF_pr[i] = reduction(loc_mem,local_ID,localsize);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //reduction of tJJ
    for(int i=0;i<PARA_NUM*(PARA_NUM+1)/2;i++){
        loc_mem[local_ID] = tJJ_pr[i];
        loc_mem[local_ID+localsize] = tJJ_pr[i + PARA_NUM*(PARA_NUM+1)/2];
        barrier(CLK_LOCAL_MEM_FENCE);
        tJJ_pr[i] = reduction(loc_mem,local_ID,localsize);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    
    //output dF2, tJJ, tJdF to global memory
    if(local_ID==0){
        dF2X[Y + Z*IMAGESIZE_Y/mergeN] = dF2_pr[0];
        devX[Y + Z*IMAGESIZE_Y/mergeN] = dev[0];
        for(int i=0;i<PARA_NUM;i++){
            tJdFX[Y + Z*IMAGESIZE_Y/mergeN + i*IMAGESIZE_Y/mergeN*Zsize] = tJdF_pr[i];
        }
        for(int i=0;i<PARA_NUM*(PARA_NUM+1)/2;i++){
            tJJX[Y + Z*IMAGESIZE_Y/mergeN + i*IMAGESIZE_Y/mergeN*Zsize] = tJJ_pr[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
}


__kernel void imageReg1Y(__global float *dF2X, __global float *tJdFX, __global float *tJJX,
                         __global float* devX, __global float *dF2, __global float *tJdF,
                         __global float *tJJ, __global float *dev, int mergeN, __local float* loc_mem){
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_global_id(1);
    const size_t Zsize = get_global_size(1);
    
    
    size_t ID1, ID2;
    float dF2_pr[2];
    float tJdF_pr[PARA_NUM*2];
    float tJJ_pr[PARA_NUM*(PARA_NUM+1)];
    float dev_pr[2];
    dF2_pr[0]=0.0f;
    dF2_pr[1]=0.0f;
    dev_pr[0]=0.0f;
    dev_pr[1]=0.0f;
    
    //initailize tJJ,tJdF
    for(int i=0; i<PARA_NUM*2; i++){
        tJdF_pr[i]=0.0f;
    }
    for(int i=0; i<PARA_NUM*(PARA_NUM+1); i++){
        tJJ_pr[i]=0.0f;
    }
    
    const size_t imageSizeY = IMAGESIZE_Y/mergeN;
    const size_t imageSizeM = imageSizeY*Zsize;
    
    for(int j=0;j<imageSizeY;j+=localsize*2){
        ID1 = j             + Z*imageSizeY;
        ID2 = j + localsize + Z*imageSizeY;
        
        dF2_pr[0] += dF2X[ID1];
        dF2_pr[1] += dF2X[ID2];
        dev_pr[0] += devX[ID1];
        dev_pr[1] += devX[ID2];
        
        for(int i=0;i<PARA_NUM;i++){
            tJdF_pr[i]          += tJdFX[ID1 + i*imageSizeM];
            tJdF_pr[i+PARA_NUM] += tJdFX[ID2 + i*imageSizeM];
        }
        for(int i=0;i<PARA_NUM*(PARA_NUM+1)/2;i++){
            tJJ_pr[i]                           += tJJX[ID1 + i*imageSizeM];
            tJJ_pr[i+PARA_NUM*(PARA_NUM+1)/2]   += tJJX[ID2 + i*imageSizeM];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    
    
    //reduction of dev
    loc_mem[local_ID] = dev_pr[0];
    loc_mem[local_ID+localsize] = dev_pr[1];
    barrier(CLK_LOCAL_MEM_FENCE);
    dev_pr[0] = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    //reduction of dF2
    loc_mem[local_ID] = dF2_pr[0];
    loc_mem[local_ID+localsize] = dF2_pr[1];
    barrier(CLK_LOCAL_MEM_FENCE);
    dF2_pr[0] = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    //reduction of tJdF
    for(int i=0;i<PARA_NUM;i++){
        loc_mem[local_ID] = tJdF_pr[i];
        loc_mem[local_ID+localsize] = tJdF_pr[i + PARA_NUM];
        barrier(CLK_LOCAL_MEM_FENCE);
        tJdF_pr[i] = reduction(loc_mem,local_ID,localsize);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    //reduction of tJJ
    for(int i=0;i<PARA_NUM*(PARA_NUM+1)/2;i++){
        loc_mem[local_ID] = tJJ_pr[i];
        loc_mem[local_ID+localsize] = tJJ_pr[i + PARA_NUM*(PARA_NUM+1)/2];
        barrier(CLK_LOCAL_MEM_FENCE);
        tJJ_pr[i] = reduction(loc_mem,local_ID,localsize);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    
    //output dF2, tJJ, tJdF to global memory
    if(local_ID==0){
        dF2[Z] = dF2_pr[0]/max(dev_pr[0],1.0f);
        dev[Z] = dev_pr[0];
        for(int i=0;i<PARA_NUM;i++){
            tJdF[Z + i*Zsize] = tJdF_pr[i]/max(dev_pr[0],1.0f);
        }
        for(int i=0;i<PARA_NUM*(PARA_NUM+1)/2;i++){
            tJJ[Z + i*Zsize] = tJJ_pr[i]/max(dev_pr[0],1.0f);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
}


//imageReg2: estimate dF2(new)
__kernel void imageReg2X(__read_only image2d_array_t mt_t_img,__read_only image2d_array_t mt_s_img,
                         __read_only image2d_array_t weight_img,
                        __constant float *p,__constant float *p_target,
                        __global float *dF2X, __global float *dFX, __global float *devX,
                         int mergeN, __local float* loc_mem){
    
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Y = get_global_id(1);
    const size_t Z = get_global_id(2);
    const size_t Zsize = get_global_size(2);
    
    
    //copy data from global p to private p_pr
    float p_pr[PARA_NUM],p_target_pr[PARA_NUM];
    for(int i=0;i<PARA_NUM;i++){
        p_pr[i]=p[Z+i*Zsize];
        p_target_pr[i]=p_target[Z+i*Zsize];
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    
    size_t X;
    float4 XYZ, XYZ_s, XYZ_t;
    float4 mt_s, mt_t;
    float mask;
    float dF;
    float sum_dF[2];
    float dF2[2];
    float dev[2];
    sum_dF[0]=0.0f;
    sum_dF[1]=0.0f;
    dF2[0]=0.0f;
    dF2[1]=0.0f;
    dev[0]=0.0f;
    dev[1]=0.0f;
    float weight;
    
    
    //estimate tJJ, tJdF, dF2
    for(int j=0; j<IMAGESIZE_X/mergeN; j+=2*localsize){
        for(int i=0;i<2;i++){
            X = local_ID + i*localsize + j;
            
            //dF
            XYZ = (float4)(X,Y,Z,0.0f);
            XYZ_s = transXY(XYZ, p_pr, (float)mergeN, REGMODE);
            XYZ_t = transXY(XYZ, p_target_pr, (float)mergeN, REGMODE);
            mt_s = read_imagef(mt_s_img,s_linear_cEdge,XYZ_s);
            mt_t = read_imagef(mt_t_img,s_linear_cEdge,XYZ_t);
            mask = mt_t.y*mt_s.y;
            dF = (mt_t.x - (mt_s.x+p_pr[PARA_NUM-1])*exp(p_pr[PARA_NUM-2]))*mask;
            weight = read_imagef(weight_img,s_linear_cEdge,XYZ).x*mask;
            
            sum_dF[i] += dF*weight;
            dF2[i] += dF*dF*weight;
            dev[i] += mask*weight;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    
    
    //reduction of dev
    loc_mem[local_ID] = dev[0];
    loc_mem[local_ID+localsize] = dev[1];
    barrier(CLK_LOCAL_MEM_FENCE);
    dev[0] = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    //reduction of dF2
    loc_mem[local_ID] = dF2[0];
    loc_mem[local_ID+localsize] = dF2[1];
    barrier(CLK_LOCAL_MEM_FENCE);
    dF2[0] = reduction(loc_mem,local_ID,localsize);
    //reduction of dF
    loc_mem[local_ID] = sum_dF[0];
    loc_mem[local_ID+localsize] = sum_dF[1];
    barrier(CLK_LOCAL_MEM_FENCE);
    sum_dF[0] = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    //output dF2 to global memory
    if(local_ID==0){
        dFX[Y + Z*IMAGESIZE_Y/mergeN] = sum_dF[0];
        dF2X[Y + Z*IMAGESIZE_Y/mergeN] = dF2[0];
        devX[Y + Z*IMAGESIZE_Y/mergeN] = dev[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
}


__kernel void imageReg2Y(__global float *dF2X, __global float *dFX, __global float* devX,
                         __global float *dF2, __global float *dF, int mergeN, __local float* loc_mem){
    const size_t local_ID = get_local_id(0);
    const size_t localsize = get_local_size(0);
    const size_t Z = get_global_id(1);
    
    
    size_t ID1, ID2;
    float dF_pr[2];
    float dF2_pr[2];
    float dev_pr[2];
    dF_pr[0]=0.0f;
    dF_pr[1]=0.0f;
    dF2_pr[0]=0.0f;
    dF2_pr[1]=0.0f;
    dev_pr[0]=0.0f;
    dev_pr[1]=0.0f;
    
    const size_t imageSizeY = IMAGESIZE_Y/mergeN;
    for(int i=0;i<imageSizeY;i+=localsize*2){
        ID1 = i             + Z*imageSizeY;
        ID2 = i + localsize + Z*imageSizeY;
        
        dF_pr[0] += dFX[ID1];
        dF_pr[1] += dFX[ID2];
        dF2_pr[0] += dF2X[ID1];
        dF2_pr[1] += dF2X[ID2];
        dev_pr[0] += devX[ID1];
        dev_pr[1] += devX[ID2];
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    
    
    //reduction of dev
    loc_mem[local_ID] = dev_pr[0];
    loc_mem[local_ID+localsize] = dev_pr[1];
    barrier(CLK_LOCAL_MEM_FENCE);
    dev_pr[0] = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    //reduction of dF2
    loc_mem[local_ID] = dF2_pr[0];
    loc_mem[local_ID+localsize] = dF2_pr[1];
    barrier(CLK_LOCAL_MEM_FENCE);
    dF2_pr[0] = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    //reduction of dF
    loc_mem[local_ID] = dF_pr[0];
    loc_mem[local_ID+localsize] = dF_pr[1];
    barrier(CLK_LOCAL_MEM_FENCE);
    dF_pr[0] = reduction(loc_mem,local_ID,localsize);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //output dF2, tJJ, tJdF to global memory
    if(local_ID==0){
        dF2[Z] = dF2_pr[0]/max(dev_pr[0],1.0f);
        dF[Z]  = dF_pr[0]/max(dev_pr[0],1.0f);
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
}


__kernel void estimateParaError(__global float* p_error, __global float* tJdF_img,
                                __global float* tJJ_img, __global float* dev_img){
    const size_t X = get_global_id(0); //angle dimension
    const size_t Y = get_global_id(1);
    const size_t Z = get_global_id(2); //parameter dimension
    const size_t sizeX = get_global_size(0);
    const size_t sizeY = get_global_size(1);
    const size_t ID = X + Y*sizeX;
    const size_t ID1 = ID + Z*sizeX*sizeY;
    const size_t ID2 = ID + Z*(2*PARA_NUM-Z+1)/2*sizeX*sizeY;
    
    float tJdF = tJdF_img[ID1];
    float tJJ  = tJJ_img[ID2];
    float dev  = dev_img[ID];
    float error = sqrt(tJdF*tJdF + tJJ/dev);
    float error1 = fabs(tJdF + error)/tJJ;
    float error2 = fabs(tJdF - error)/tJJ;
    //p_error[ID1] = fmax(error,0.01f);
    //p_error[ID1] = fmax(sqrt(1.0f/dev/tJJ),0.01f);
    p_error[ID1] = fmin(fmax(fmax(error1,error2),0.01f),1000.0f);
}


//output final image reg results
__kernel void output_imgReg_result(__read_only image2d_array_t mt_img, __global float *mt_buf,
                                   __constant float *p){
    
    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    const size_t Z = get_global_id(2);
    const size_t Zsize = get_global_size(2);
    
    size_t ID;
    float4 XYZ, XYZ_s, img;
    float p_pr[PARA_NUM];
    
    
    //copy data from global transpara to local transpara_atE
    for(size_t i=0;i<PARA_NUM;i++){
        p_pr[i]=p[Z+i*Zsize];
        barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    }
    
    
    //convert image reg results
    XYZ = (float4)(X,Y,Z,0.0f);
    XYZ_s = transXY(XYZ, p_pr, 1, REGMODE);
    ID = X+Y*IMAGESIZE_X+Z*IMAGESIZE_M;
        
    img = read_imagef(mt_img,s_linear,XYZ_s);
    mt_buf[ID]=img.x*img.y;
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





//update weight
__kernel void updateWeight(__read_only image2d_array_t mt_t_img,__read_only image2d_array_t mt_s_img,
                           __write_only image2d_array_t weight_img,
                           __constant float* dF2, __constant float* dF, float CI,
                           __constant float *p,__constant float *p_target, int mergeN){
    
    const size_t X = get_global_id(0);
    const size_t Y = get_global_id(1);
    const size_t Z = get_global_id(2);
    const size_t Zsize = get_global_size(2);
    
    
    //copy data from global p to private p_pr
    float p_pr[PARA_NUM], p_target_pr[PARA_NUM];
    for(int i=0;i<PARA_NUM;i++){
        p_pr[i]=p[Z+i*Zsize];
        p_target_pr[i]=p_target[Z+i*Zsize];
    }
    barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
    
    
    float4 XYZ, XYZ_s, XYZ_t;
    int4 XYZ_i;
    float4 mt_s, mt_t;
    float mask;
    float weight, dF_pr;
    float weight_thd = sqrt(dF2[Z] - dF[Z]*dF[Z])*CI;
    
        
    //dF
    XYZ = (float4)(X,Y,Z,0.0f);
    XYZ_s = transXY(XYZ, p_pr, (float)mergeN, REGMODE);
    XYZ_t = transXY(XYZ, p_target_pr, (float)mergeN, REGMODE);
    mt_s = read_imagef(mt_s_img,s_linear_cEdge,XYZ_s);
    mt_t = read_imagef(mt_t_img,s_linear_cEdge,XYZ_t);
    mask = mt_t.y*mt_s.y;
    dF_pr = (mt_t.x - (mt_s.x+p_pr[PARA_NUM-1])*exp(p_pr[PARA_NUM-2]))*mask;
    
    
    //weight
    XYZ_i = (int4)(X,Y,Z,0);
    weight = (dF_pr-dF[Z])/weight_thd;
    weight = (fabs(weight)<=1.0f) ? (1.0f - weight*weight):0.0f;
    weight = weight*weight*mask;
    write_imagef(weight_img,XYZ_i,(float4)(weight,0.0f,0.0f,0.0f));
}
