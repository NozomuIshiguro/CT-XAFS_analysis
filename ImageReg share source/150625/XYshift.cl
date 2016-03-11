#define NUM_TRANSPARA 2
#define NUM_REDUCTION 6




//transform XY
#define TRANS_XY(XY,t,x,y,z)\
(XY) = (float4)((x)+((__local float*)(t))[0],(y)+((__local float*)(t))[1],(z),0)



//calculate Jacobian
#define JACOBIAN(t,X,Y,j,dx,dy,msk,ms,ml) \
{\
((float*)(j))[0] = (dx)*(msk)/(ms)/(ms)*(ml);\
((float*)(j))[1] = (dy)*(msk)/(ms)/(ms)*(ml);\
}

