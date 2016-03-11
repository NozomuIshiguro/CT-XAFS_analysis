#define NUM_TRANSPARA 3
#define NUM_REDUCTION 10




//transform XY
#define TRANS_XY(XY,t,x,y,z)\
(XY) = (float4)(\
((__local float*)(t))[2]*(x)+((__local float*)(t))[0],\
((__local float*)(t))[2]*(y)+((__local float*)(t))[1],\
(z),0)




//calculate Jacobian
#define JACOBIAN(t,x,y,j,dx,dy,msk,ms,ml)\
{\
((float*)(j))[0] = (dx)*(msk)/(ms)/(ms)*(ml);\
((float*)(j))[1] = (dy)*(msk)/(ms)/(ms)*(ml);\
((float*)(j))[2] = ((dx)*(x)+(dy)*(y))*(msk)/(ms)/(ms)*(ml);\
}

