#define NUM_TRANSPARA 3
#define NUM_REDUCTION 10



//copy data from global transpara to local transpara_atE
#define TRANSPARA_LOC_COPY(tp,tpg,lid,gid)\
{\
if((lid)==0){\
((__local float*)(tp))[0]=((__global float*)(tpg))[gid*3];\
((__local float*)(tp))[1]=((__global float*)(tpg))[gid*3+1];\
((__local float*)(tp))[2]=((__global float*)(tpg))[gid*3+2];\
}\
barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);\
}



//transform XY
#define TRANS_XY(XY,t,x,y)\
(XY) = (float2)((x)*((__local float*)(t))[2]\
+IMAGESIZE_X*(1-((__local float*)(t))[2])/2\
+((__local float*)(t))[0],\
(y)*((__local float*)(t))[2]\
+IMAGESIZE_Y*(1-((__local float*)(t))[2])/2\
+((__local float*)(t))[1])



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
((__local float*)(l))[lid+lsz*9]=0;\
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
((__local float*)(rp))[9]=0;\
}



//calculate Jacobian
#define JACOBIAN(t,x,y,j,dx,dy,msk,ms,ml)\
{\
((float*)(j))[0] = (dx)*(msk)/(ms)/(ms)*(ml);\
((float*)(j))[1] = (dy)*(msk)/(ms)/(ms)*(ml);\
((float*)(j))[2] = ((dx)*(x-IMAZESIZE_X/2)+(dy)*(y-IMAZESIZE_Y/2))*(ml);\
}



//copy Jacobian & dimg to local memory
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
barrier(CLK_LOCAL_MEM_FENCE);\
}



//reduction
#define REDUCTION(l,rp) reduction((__local float*)(l),(__local float*)(rp),(10));




//calculate delta_transpara
#define CALC_DELTA_TRANSPARA(rp,l,dtp,dr,lid) \
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
barrier(CLK_LOCAL_MEM_FENCE);\
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
(dr) = (((__local float*)(dtp))[0]*((l)*((__local float*)(dtp))[0]+((__local float*)(rp))[6])\
+((__local float*)(dtp))[1]*((l)*((__local float*)(dtp))[1]+((__local float*)(rp))[7])\
+((__local float*)(dtp))[2]*((l)*((__local float*)(dtp))[2]+((__local float*)(rp))[8]));\
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
((__local float*)(tp))[2] += (m)*((__local float*)(dtp))[2];\
}\
barrier(CLK_LOCAL_MEM_FENCE);\
\
}



//copy data from local transpara_atE to global transpara
#define TRANSPARA_GLOB_COPY(tp,tpg,lid,gid)\
{\
if((lid)==0){\
((__global float*)(tpg))[gid*3]=((__local float*)(tp))[0];\
((__global float*)(tpg))[gid*3+1]=((__local float*)(tp))[1];\
((__global float*)(tpg))[gid*3+2]=((__local float*)(tp))[2];\
}\
barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);\
}

//copy data from local transpara_err to global memory
#define TRANSPARA_ERR_GLOB_COPY(rp,tp_err,lid,gid)\
{\
if((lid)==0){\
((__global float*)(tp_err))[gid*3]=0.5/fabs(((__local float*)(rp))[6]);\
((__global float*)(tp_err))[gid*3+1]=0.5/fabs(((__local float*)(rp))[7]);\
((__global float*)(tp_err))[gid*3+2]=0.5/fabs(((__local float*)(rp))[8]);\
}\
barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);\
}

