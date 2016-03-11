//
//  atan_lor_linear_fitting.hpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/03/13.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#ifndef CT_XANES_analysis_atan_lor_linear_fitting_hpp
#define CT_XANES_analysis_atan_lor_linear_fitting_hpp

const string atanlorlinear_preprocessor = "#define FIT_AND_JACOBIAN(x,mt,j,fp,msk){\\\n\
\\\n\
float a_X = ((x)-((float*)(fp))[3])/((float*)(fp))[4];\\\n\
float l_X = ((x)-((float*)(fp))[6])/((float*)(fp))[7];\\\n\
\\\n\
(mt) = ((float*)(fp))[0] + ((float*)(fp))[1]*(x)\\\n\
        +((float*)(fp))[2]*(0.5f+atanpi(a_X))\\\n\
        +((float*)(fp))[5]/(1+l_X*l_X);\\\n\
(mt)*=(msk);\\\n\
\\\n\
((float*)(j))[0]=1.0;\\\n\
((float*)(j))[1]=(x);\\\n\
((float*)(j))[2]=0.5f+atanpi(a_X);\\\n\
((float*)(j))[3]=-((float*)(fp))[2]/PI/((float*)(fp))[4]/(1+a_X*a_X);\\\n\
((float*)(j))[4]=-a_X*((float*)(fp))[2]/PI/((float*)(fp))[4]/(1+a_X*a_X);\\\n\
((float*)(j))[5]=1/(1+l_X*l_X);\\\n\
((float*)(j))[6]=l_X*2*((float*)(fp))[5]/((float*)(fp))[7]/(1+l_X*l_X)/(1+l_X*l_X);\\\n\
((float*)(j))[7]=l_X*l_X*2*((float*)(fp))[5]/((float*)(fp))[7]/(1+l_X*l_X)/(1+l_X*l_X);\\\n\
\\\n\
}\n\
\n\
";

#endif
