#define NUM_TRANSPARA 2
#define NUM_REDUCTION 9



//copy data from global transpara to local transpara_atE
#define TRANSPARA_LOC_COPY(tp,tpg,lid,gid)\
{\
if((lid)==0){\
((__local float*)(tp))[0]=((__global float*)(tpg))[gid*2];\
((__local float*)(tp))[1]=((__global float*)(tpg))[gid*2+1];\
}\
barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);\
}



//transform XY
#define TRANS_XY(XY,t,x,y)\
(XY) = (float2)((x)+((__local float*)(t))[0],(y)+((__local float*)(t))[1])



//Local memory (for reduction) reset
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
}



//reset reduction results
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
}

//calculate Jacobian
#define JACOBIAN(t,X,Y,j,dx,dy,msk,ms) \
{\
((float*)(j))[0] = (dx)*(msk)/(ms)/(ms);\
((float*)(j))[1] = (dy)*(msk)/(ms)/(ms);\
}


//copy Jacobian & dimg to local memory
#define LOCMEM_COPY(l,j,dimg,m,jm,lid,lsz)\
{\
((__local float*)(l))[lid]+=((float*)(j))[0]*((float*)(j))[0];\
((__local float*)(l))[lid+lsz]+=((float*)(j))[1]*((float*)(j))[0];\
((__local float*)(l))[lid+lsz*2]+=((float*)(j))[1]*((float*)(j))[1];\
((__local float*)(l))[lid+lsz*3]+=((float*)(j))[0]*(dimg);\
((__local float*)(l))[lid+lsz*4]+=((float*)(j))[1]*(dimg);\
((__local float*)(l))[lid+lsz*5]+=(dimg)*(dimg);\
((__local float*)(l))[lid+lsz*6]+=(float)(m);\
((__local float*)(l))[lid+lsz*7]+=((float*)(jm))[0];\
((__local float*)(l))[lid+lsz*8]+=((float*)(jm))[1];\
barrier(CLK_LOCAL_MEM_FENCE);\
}



//reduction
#define REDUCTION(l,rp) reduction((__local float*)(l),(__local float*)(rp),(9));



//calculate delta_transpara
#define CALC_DELTA_TRANSPARA(rp,l,dtp,dr,lid) \
{\
float tJJ[2][2];\
tJJ[0][0]=((__local float*)(rp))[0]+((__local float*)(rp))[3]*((__local float*)(rp))[7]/((__local float*)(rp))[6]+(l);\
tJJ[0][1]=((__local float*)(rp))[1]+((__local float*)(rp))[3]*((__local float*)(rp))[8]/((__local float*)(rp))[6]\
                                   +((__local float*)(rp))[4]*((__local float*)(rp))[7]/((__local float*)(rp))[7];\
tJJ[1][0]=((__local float*)(rp))[1]+((__local float*)(rp))[3]*((__local float*)(rp))[8]/((__local float*)(rp))[6]\
                                   +((__local float*)(rp))[4]*((__local float*)(rp))[7]/((__local float*)(rp))[7];\
tJJ[1][1]=((__local float*)(rp))[2]+((__local float*)(rp))[4]*((__local float*)(rp))[8]/((__local float*)(rp))[6]+(l);\
barrier(CLK_LOCAL_MEM_FENCE);\
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
float tJdImg[2];\
tJdImg[0] = ((__local float*)(rp))[3]+((__local float*)(rp))[5]*((__local float*)(rp))[7]/((__local float*)(rp))[6]/2;\
tJdImg[1] = ((__local float*)(rp))[4]+((__local float*)(rp))[5]*((__local float*)(rp))[8]/((__local float*)(rp))[6]/2;\
\
if((lid)==0){\
((__local float*)(dtp))[0]=inv_tJJ[0][0]*tJdImg[0]+inv_tJJ[0][1]*tJdImg[1];\
((__local float*)(dtp))[1]=inv_tJJ[1][0]*tJdImg[0]+inv_tJJ[1][1]*tJdImg[1];\
\
(dr) = (((__local float*)(dtp))[0]*((l)*((__local float*)(dtp))[0]+tJdImg[0])\
+((__local float*)(dtp))[1]*((l)*((__local float*)(dtp))[1]+tJdImg[1]));\
}\
barrier(CLK_LOCAL_MEM_FENCE);\
\
}



//update transpara
#define UPDATE_TRANSPARA(dtp,tp,m,lid)\
{\
if((lid)==0){\
((__local float*)(tp))[0] += (m)*((__local float*)(dtp))[0];\
((__local float*)(tp))[1] += (m)*((__local float*)(dtp))[1];\
}\
barrier(CLK_LOCAL_MEM_FENCE);\
\
}



//copy data from local transpara_atE to global transpara
#define TRANSPARA_GLOB_COPY(tp,tpg,lid,gid)\
{\
if((lid)==0){\
((__global float*)(tpg))[gid*2]=((__local float*)(tp))[0];\
((__global float*)(tpg))[gid*2+1]=((__local float*)(tp))[1];\
}\
barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);\
}

