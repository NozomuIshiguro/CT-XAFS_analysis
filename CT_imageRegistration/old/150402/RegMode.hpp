//
//  RegMode.hpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/02/25.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#ifndef CT_XANES_analysis_RegMode_hpp
#define CT_XANES_analysis_RegMode_hpp

#if REGMODE==3  //rotation+scale+xy shift

#define transpara_num 4
#define reductpara_num 20

#define OSS_TARGET "    registration shift: dx=0, dy=0, d(phi)=0, scale=1.0\n\n"

#define OSS_SAMPLE(oss,tp,tp_err)\
{\
oss<<"    registration shift: dx=";\
oss.precision(((int*)(p_err))[0]);\
oss<< ((float*)(tp))[0]<<" +/- ";\
oss.precision(1);\
oss<<abs(((float*)(tp_err))[0])<<", dy=";\
oss.precision(((int*)(p_err))[1]);\
oss<<((float*)(tp))[1]<<" +/- ";\
oss.precision(1);\
oss<<abs(((float*)(tp_err))[1])<<", d(phi)=";\
oss.precision(((int*)(p_err))[2]);\
oss<<((float*)(tp))[2]<<" +/- ";\
oss.precision(1);\
oss<<abs(((float*)(tp_err))[2])<<", scale=";\
oss.precision(((int*)(p_err))[3]);\
oss<<((float*)(tp))[3]<<" +/- ";\
oss.precision(1);\
oss<<abs(((float*)(tp_err))[3])<<"\n\n";\
}

#elif REGMODE==2  //scale+xy shift

#define transpara_num 3
#define reductpara_num 14

#define OSS_TARGET "    registration shift: dx=0, dy=0, scale=1.0\n\n"

#define OSS_SAMPLE(oss,tp,tp_err)\
{\
oss<<"    registration shift: dx=";\
oss.precision(((int*)(p_err))[0]);\
oss<<((float*)(tp))[0]<<" +/- ";\
oss.precision(1);\
oss<<abs(((float*)(tp_err))[0])<<", dy=";\
oss.precision(((int*)(p_err))[1]);\
oss<<((float*)(tp))[1]<<" +/- ";\
oss.precision(1);\
oss<<abs(((float*)(tp_err))[1])<<", scale=";\
oss.precision(((int*)(p_err))[2]);\
oss<<((float*)(tp))[2]<<" +/- ";\
oss.precision(1);\
oss<<abs(((float*)(tp_err))[2])<<"\n\n";\
}

#elif REGMODE==1  //rotation+xy shift

#define transpara_num 3
#define reductpara_num 14

#define OSS_TARGET "    registration shift: dx=0, dy=0, d(phi)=0\n\n"

#define OSS_SAMPLE(oss,tp,p_err)\
{\
oss<<"    registration shift: dx=";\
oss.precision(((int*)(p_err))[0]);\
oss<<((float*)(tp))[0]<<" +/- ";\
oss.precision(1);\
oss<<abs(((float*)(tp_err))[0])<<", dy=";\
oss.precision(((int*)(p_err))[1]);\
oss<<((float*)(tp))[1]<<" +/- ";\
oss.precision(1);\
oss<<abs(((float*)(tp_err))[1])<<", d(phi)=";\
oss.precision(((int*)(p_err))[2]);\
oss<<((float*)(tp))[2]<<" +/- ";\
oss.precision(1);\
oss<<abs(((float*)(tp_err))[2])<<"\n\n";\
}


#else //xy shift

#define transpara_num 2
#define reductpara_num 9

#define OSS_TARGET "    registration shift: dx=0, dy=0\n\n"

#define OSS_SAMPLE(oss,tp,tp_err,p_err)\
{\
oss<<"    registration shift: dx=";\
oss.precision(((int*)(p_err))[0]);\
oss<< ((float*)(tp))[0]<<" +/- ";\
oss.precision(1);\
oss<<abs(((float*)(tp_err))[0])<<", dy=";\
oss.precision(((int*)(p_err))[1]);\
oss<<((float*)(tp))[1]<<" +/- ";\
oss.precision(1);\
oss<<abs(((float*)(tp_err))[1])<<"\n\n";\
}

#endif

#endif
