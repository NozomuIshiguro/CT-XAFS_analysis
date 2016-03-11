//
//  OpenCL_analysis.hpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/03/06.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#ifndef CT_XANES_analysis_OpenCL_analysis_hpp
#define CT_XANES_analysis_OpenCL_analysis_hpp
#define __CL_ENABLE_EXCEPTIONS
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <thread>
#include <future>
#include <iomanip>
//#include <basic_ios>
#include <stdlib.h>

#include <time.h>
#include <sys/stat.h>

#if defined (__APPLE__)  //Mac OS X, iOS?
#include <OpenCL/CL.hpp>
#include <dirent.h>
#define WINDOWS 0

#define MKDIR(c) \
mkdir((const char*)(c), 0755)

#define FILEOPEN(fp,c) \
((FILE*)(fp)) = fopen((const char*)(c),"rb")

#elif defined (_M_X64)  //Windows 64 bit
#include <windows.h>
#include <CL/cl.hpp>
#include <direct.h>
#include <dirent.h>
#define WINDOWS 1

#define MKDIR(c) \
_mkdir((const char*)(c))

#define FILEOPEN(fp,c,s) \
fopen_s(&((FILE*)(fp)),((const char*)(c)),((const char*)(s)))


#elif defined (_WIN32)  //Windows 32 bit
#include <windows.h>
#include <CL/cl.hpp>
#include <direct.h>
#include <dirent.h>
#define WINDOWS 1

#define MKDIR(c) \
_mkdir((const char*)(c))

#define FILEOPEN(fp,c,s) \
fopen_s(&((FILE*)(fp)),((const char*)(c)),((const char*)(s)))

#elif defined (__linux__)   //Linux
#include <CL/cl.hpp>
#include <dirent.h>
#define WINDOWS 0

#define MKDIR(c) \
mkdir((const char*)(c), 0755)

#define FILEOPEN(fp,c,s) \
((FILE*)(fp)) = fopen(((const char*)(c)),((const char*)(s)))

#endif
using namespace std;

#define IMAGE_SIZE_X 2048
#define IMAGE_SIZE_Y 2048
#define IMAGE_SIZE_M 4194304 //2048*2048

class OCL_platform_device{
    vector<int> platform_num;
    vector<int> device_num;
    vector<cl::Platform> p_ids;
    vector<cl::Device> d_ids;
    vector<cl::Context> contexts;
    vector<cl::CommandQueue> queues;
public:
    OCL_platform_device(string plat_dev_str);
    int plat_num(int list_num);
    int dev_num(int list_num);
    size_t size();
    cl::Platform plat(int list_num);
    cl::Device dev(int list_num);
    cl::Context context(int list_num);
    cl::CommandQueue queue(int list_num);
};

class input_parameter{
    string ocl_plat_dev_list;
    string data_input_dir;
    string data_output_dir;
    int startEnergyNo;
    int endEnergyNo;
    int startAngleNo;
    int endAngleNo;
    int targetEnergyNo;
    
    string energyFilePath;
    float E0;
    float startEnergy;
    float endEnergy;
    
    vector<float> fitting_para;
    vector<char> free_para;
    vector<string> parameter_name;
    
    vector<float> para_upperLimit;
    vector<float> para_lowerLimit;
public:
    input_parameter(string inputfile_path);
    string getPlatDevList();
    string getInputDir();
    string getOutputDir();
    string getEnergyFilePath();
    float getE0();
    float getStartEnergy();
    float getEndEnergy();
    int getStartEnergyNo();
    int getEndEnergyNo();
    int getStartAngleNo();
    int getEndAngleNo();
    int getTargetEnergyNo();
    vector<float> getFittingPara();
    vector<char> getFreeFixPara();
    vector<string> getFittingParaName();
    void setInputDir(string message);
    void setOutputDir(string message);
    void setEnergyFilePath(string message);
    void setE0(string message);
    void setEnergyRange(string message);
    void setAngleRange(string message);
    void setEnergyNoRange(string message);
    void setTargetEnergyNo(string message);
    void setFittingPara(string message);
    void setFreeFixPara(string message);
    void setFittingParaName(string message);
    void setValidParaLowerLimit(string message);
    void setValidParaUpperLimit(string message);
    vector<float> getParaUpperLimit();
    vector<float> getParaLowerLimit();
};

string IntToString(int number);
string EnumTagString(int EnergyNo);
string AnumTagString(int angleNo,string preStr, string postStr);
string kernel_preprocessor_def(string title, string pointertype,
                        string memobjectname,string pointername,
                        size_t num_pointer, size_t num_buffer,size_t shift);
string kernel_preprocessor_read(string title, string memobjectname,
                                size_t num_pointer, size_t num_buffer,size_t shift);
string kernel_preprocessor_write(string title, string memobjectname,
                                size_t num_pointer, size_t num_buffer,size_t shift);
#endif
