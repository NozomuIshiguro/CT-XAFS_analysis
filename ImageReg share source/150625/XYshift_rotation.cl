#define NUM_TRANSPARA 3
#define NUM_REDUCTION 10



//transform XY
#define TRANS_XY(XY,t,x,y,z)\
(XY) = (float4)(\
cos(((__local float*)(t))[2])*(x)-sin(((__local float*)(t))[2])*(y)+((__local float*)(t))[0],\
sin(((__local float*)(t))[2])*(x)+cos(((__local float*)(t))[2])*(y)+((__local float*)(t))[1],\
(z),0)




//calculate Jacobian
#define JACOBIAN(t,x,y,j,dx,dy,msk,ms,ml)\
{\
float DfxDth = (-(x)*sin(((__local float*)(t))[2])-(y)*cos(((__local float*)(t))[2]));\
float DfyDth = ((x)*cos(((__local float*)(t))[2])-(y)*sin(((__local float*)(t))[2]));\
barrier(CLK_LOCAL_MEM_FENCE);\
\
((float*)(j))[0] = (dx)*(msk)/(ms)/(ms)*(ml);\
((float*)(j))[1] = (dy)*(msk)/(ms)/(ms)*(ml);\
((float*)(j))[2] = ((dx)*DfxDth+(dy)*DfxDth)*(msk)/(ms)/(ms)*(ml);\
}

