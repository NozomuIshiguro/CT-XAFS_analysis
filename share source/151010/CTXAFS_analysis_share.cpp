//
//  CTXAFS_analysis_share.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/03/06.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"

string output_flag(string flag, int argc, const char * argv[]){
    
    string output;
    for (int i=1; i<argc-1; i++) {
        string argv_str=argv[i];
        if(argv_str=="flag"){
            output=argv[i+1];
            break;
        }
    }
    
    return output;
}

string IntToString(int number)
{
    stringstream ss;
    ss << number;
    return ss.str();
}

string LnumTagString(int LoopNo,string preStr, string postStr){
    
    string LnumTag;
    if (LoopNo<10) {
        LnumTag=preStr+"0"+IntToString(LoopNo)+postStr;
    } else {
        LnumTag=preStr+IntToString(LoopNo)+postStr;
    }
    return LnumTag;
}

string EnumTagString(int EnergyNo,string preStr, string postStr){
    
    string EnumTag;
    if (EnergyNo<10) {
        EnumTag=preStr+"00"+IntToString(EnergyNo)+postStr;
    } else if(EnergyNo<100){
        EnumTag=preStr+"0"+IntToString(EnergyNo)+postStr;
    } else {
        EnumTag=preStr+IntToString(EnergyNo)+postStr;
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

