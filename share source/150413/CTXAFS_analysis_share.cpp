//
//  CTXAFS_analysis_share.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/03/06.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"


string kernel_preprocessor_read(string title, string memobjectname,
                                size_t num_pointer, size_t num_buffer,size_t shift){
    ostringstream OSS;
    size_t buffer_pnts=num_pointer/num_buffer;
    if (num_pointer%num_buffer>0) buffer_pnts++;
    
    OSS<<"#define "<<title<<"_READ(pmem,id) {\\"<<endl;
    for (int i=0; i<num_pointer; i++) {
        OSS << "((float*)(pmem))["<<i<<"]=";
        OSS<< memobjectname<<"_N"<<(i/buffer_pnts)<<"[(id)+"<<shift*(i%buffer_pnts)<<"];\\"<<endl;
    }
    OSS<<"}"<<endl<<endl;
    
    return OSS.str();
}

string kernel_preprocessor_write(string title, string memobjectname,
                                size_t num_pointer, size_t num_buffer,size_t shift){
    ostringstream OSS;
    size_t buffer_pnts=num_pointer/num_buffer;
    if (num_pointer%num_buffer>0) buffer_pnts++;
    
    OSS<<"#define "<<title<<"_WRITE(pmem,id) {\\"<<endl;
    for (int i=0; i<num_pointer; i++) {
        OSS <<  memobjectname<<"_N"<<(i/buffer_pnts)<<"[(id)+"<<shift*(i%buffer_pnts)<<"]=";
        OSS <<"((float*)(pmem))["<<i<<"];\\"<<endl;
    }
    OSS<<"}"<<endl<<endl;
    
    return OSS.str();
}

string kernel_preprocessor_def(string title, string pointertype,
                               string memobjectname,string pointername,
                               size_t num_pointer, size_t num_buffer,size_t shift){
    ostringstream OSS;
    
    OSS<<"#define "<<title<<"_DEF ";
    
    for (int i=0; i<num_buffer; i++) {
        OSS << pointertype << memobjectname<<"_N"<<i;
        if(i+1!=num_buffer) OSS<<",\\"<<endl;
    }
    //OSS <<" __local int *"<< pointername;
    OSS  << endl << endl;
    
    /*OSS<<"#define "<<title<<"_P {\\"<<endl;
    size_t buffer_pnts=num_pointer/num_buffer;
    if (num_pointer%num_buffer>0) buffer_pnts++;
    OSS << pointertype <<" __local "<< pointername <<"["<<num_pointer<<"]; ";
    for (int i=0; i<num_pointer; i++) {
        OSS << pointername;
        OSS << "[" << i <<"]=&";
        OSS <<memobjectname<<"_N"<<(i/buffer_pnts);
        OSS<<"["<<shift*(i%buffer_pnts)<<"]";
        OSS<<";\\"<<endl;
    }
    OSS<<"}"<<endl<<endl;*/
    
    return OSS.str();
}

string IntToString(int number)
{
    stringstream ss;
    ss << number;
    return ss.str();
}

string EnumTagString(int EnergyNo){
    
    string EnumTag;
    if (EnergyNo<10) {
        EnumTag="00"+IntToString(EnergyNo);
    } else if(EnergyNo<100){
        EnumTag="0"+IntToString(EnergyNo);
    } else {
        EnumTag=IntToString(EnergyNo);
    }
    return EnumTag;
}

string AnumTagString(int angleNo,string preStr, string postStr){
    
    string angleNumTag;
    if (angleNo<10) {
        angleNumTag = preStr+"000"+IntToString(angleNo)+postStr;
    } else if(angleNo<100){
        angleNumTag = preStr+"00"+IntToString(angleNo)+postStr;
    } else if(angleNo<1000){
        angleNumTag = preStr+"0"+IntToString(angleNo)+postStr;
    } else {
        angleNumTag = preStr+IntToString(angleNo)+postStr;
    }
    return angleNumTag;
}

