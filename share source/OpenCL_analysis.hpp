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
#include <algorithm>
#include <stdlib.h>
#include <mutex>

#include <time.h>
#include <sys/stat.h>

#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#if defined (__APPLE__)  //Mac OS X, iOS?
#include <OpenCL/cl.hpp>
#include <OpenCL/cl2.hpp>
#include <dirent.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#define GETTOTALSYSTEMMEMORY(ram) {\
int mib[2];\
size_t length;\
mib[0] = CTL_HW;\
mib[1] = HW_MEMSIZE;\
length = sizeof(int64_t);\
sysctl(mib, 2, &(ram), &length, NULL, 0);\
}
#define WINDOWS 0

#define MKDIR(c) \
mkdir((const char*)(c), 0755)

#define FILEOPEN(fp,c,s) \
((FILE*)(fp)) = fopen((const char*)(c),((const char*)(s)))

#elif defined (_M_X64)  //Windows 64 bit
#include <windows.h>
#include <CL/cl.hpp>
#include <CL/cl2.hpp>
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
#include <CL/cl2.hpp>
#include <direct.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/param.h>
#define GETTOTALSYSTEMMEMORY(ram) {\
MEMORYSTATUSEX status;\
status.dwLength = sizeof(status);\
GlobalMemoryStatusEx(&status);\
(ram)=status.ullTotalPhys;\
}
#define WINDOWS 1

#define MKDIR(c) \
_mkdir((const char*)(c))

#define FILEOPEN(fp,c,s) \
fopen_s(&((FILE*)(fp)),((const char*)(c)),((const char*)(s)))

#elif defined (__linux__)   //Linux
#include <CL/cl.hpp>
#include <CL/cl2.hpp>
#include <dirent.h>
#include <unistd.h>
#define GETTOTALSYSTEMMEMORY(ram) {\
(ram)=sysconf(_SC_PHYS_PAGES)*sysconf(_SC_PAGE_SIZE);\
}
#define WINDOWS 0

#define MKDIR(c) \
mkdir((const char*)(c), 0766)

#define FILEOPEN(fp,c,s) \
((FILE*)(fp)) = fopen(((const char*)(c)),((const char*)(s)))

#endif
using namespace std;

#define IMAGE_SIZE_X 2048
#define IMAGE_SIZE_Y 2048
#define IMAGE_SIZE_M 4194304 //2048*2048


class OCL_platform_device{
    vector<int> platform_num;
    vector<vector<int>> device_num;
    vector<cl::Platform> p_ids;
    vector<vector<cl::Device>> d_ids;
    vector<cl::Context> contexts;
    vector<vector<cl::CommandQueue>> queues;
    vector<size_t> queuesizeInContext;
    bool nonshared;
public:
    OCL_platform_device(string plat_dev_str,bool nonshared);
    int plat_num(int list_num);
    int dev_num(int plat_list_num, int dev_list_num);
    size_t devsize(int plat_list_num);
    size_t platsize();
    size_t queuesize(int context_id);
    size_t contextsize();
    cl::Platform plat(int list_num);
    cl::Device dev(int plat_list_num, int dev_list_num);
    cl::Context context(int list_num);
    cl::CommandQueue queue(int context_list_num, int queue_list_num);
};


class input_parameter_mask{
protected:
    int refMask_shape;
    int refMask_x;
    int refMask_y;
    int refMask_width;
    int refMask_height;
    float refMask_angle;
    
    int sampleMask_shape;
    int sampleMask_x;
    int sampleMask_y;
    int sampleMask_width;
    int sampleMask_height;
    float sampleMask_angle;
    
public:
    input_parameter_mask();
    input_parameter_mask(string inputfile_path);
    
    int getRefMask_shape();
    int getRefMask_x();
    int getRefMask_y();
    int getRefMask_width();
    int getRefMask_height();
    float getRefMask_angle();
    
    int getSampleMask_shape();
    int getSampleMask_x();
    int getSampleMask_y();
    int getSampleMask_width();
    int getSampleMask_height();
    float getSampleMask_angle();
    
};

class input_parameter_fitting{
protected:
    string fitting_output_dir;
    string fitting_filebase;
    
    string energyFilePath;
    float E0;
    float startEnergy;
    float endEnergy;
    int fittingStartEnergyNo;
    int fittingEndEnergyNo;
	int num_trial_fit;
	float lamda_t_fit;
    
    vector<float> fitting_para;
    vector<char> free_para;
    vector<string> parameter_name;
    
    vector<float> para_upperLimit;
    vector<float> para_lowerLimit;
    
public:
    input_parameter_fitting();
    input_parameter_fitting(string inputfile_path);

    string getFittingOutputDir();
    string getFittingFileBase();
    string getEnergyFilePath();
    float getE0();
    float getStartEnergy();
    float getEndEnergy();
    int getFittingStartEnergyNo();
    int getFittingEndEnergyNo();

    vector<float> getFittingPara();
    vector<char> getFreeFixPara();
    vector<string> getFittingParaName();
    vector<float> getParaUpperLimit();
    vector<float> getParaLowerLimit();

	int getNumTrial_fit();
	float getLambda_t_fit();

    void setFittingOutputDirFromDialog(string message);
    void setFittingFileBaseFromDialog(string message);
    void setEnergyFilePathFromDialog(string message);
    void setE0FromDialog(string message);
    void setEnergyRangeFromDialog(string message);
    
    void setFittingParaFromDialog(string message);
    void setFreeFixParaFromDialog(string message);
    void setFittingParaNameFromDialog(string message);
    void setValidParaLowerLimitFromDialog(string message);
    void setValidParaUpperLimitFromDialog(string message);
    
    
    void setFittingOutputDir(string outputDir);
    void setFittingFileBase(string outputfilebase);
    void setEnergyFilePath(string energy_path);
    void setE0(string E0_str);
    void setEnergyRange(string E_range);
    void setFittingStartEnergyNo(int energyNo);
    void setFittingEndEnergyNo(int energyNo);
    
    void setFittingPara(string fitpara_input);
    void setFreeFixPara(string freepara_inp);
    void setFittingParaName(string fitparaname_inp);
    void setValidParaLimit(string limit);

	void setLambda_t_fit(float lambda_t_inp);
	void setNumtrial_fit(int numTrial_inp);
};

class input_parameter_reslice{
protected:
    float baseup;
    int startX;
    int endX;
    int startZ;
    int endZ;
    int X_corr;
    int Z_corr;
    int layerN;
    
    float rotCenterShiftStart;
    int rotCenterShiftN;
    float rotCenterShiftStep;
    
public:
    input_parameter_reslice();
    input_parameter_reslice(string inputfile_path);
    
    float getBaseup();
    int getStartX();
    int getEndX();
    int getStartZ();
    int getEndZ();
    bool getZcorr();
    bool getXcorr();
    int getLayerN();
    float getRotCenterShiftStart();
    int getRotCenterShiftN();
    float getRotCenterShiftStep();
    
    void setBaseupFromDialog(string message);
    void setLayerNFromDialog(string message);
    void setRotCenterShiftStartFromDialog(string message);
    void setRotCenterShiftNFromDialog(string message);
    void setRotCenterShiftStepFromDialog(string message);
    
    void setLayerN(string layerN_str);
    void setRotCenterShiftStart(string str);
    void setRotCenterShiftN(string str);
    void setRotCenterShiftStep(string str);
    
};

class input_parameter : public input_parameter_mask,public input_parameter_fitting,
                        public input_parameter_reslice
{
    string ocl_plat_dev_list;
    string data_input_dir;
    string data_output_dir;
    string output_filebase;
    
    int startEnergyNo;
    int endEnergyNo;
    int startAngleNo;
    int endAngleNo;
    int targetEnergyNo;
    int targetAngleNo;
    
    int regMode;
    int cntMode;
    bool imgRegOutput;
    
    int num_trial;
    float lamda_t;
    
    int numParallel;
    vector<float> regCnt_inipara;
    
public:
    input_parameter(string inputfile_path);
    string getPlatDevList();
    void setPlatDevList(string inp_str);
    string getInputDir();
    string getOutputDir();
    string getOutputFileBase();
    int getStartEnergyNo();
    int getEndEnergyNo();
    int getStartAngleNo();
    int getEndAngleNo();
    int getTargetEnergyNo();
    int getTargetAngleNo();
    int getRegMode();
    int getCntMode();
    int getNumTrial();
    float getLambda_t();
    bool getImgRegOutput();
    int getNumParallel();
    vector<float> getRegCnt_inipara();
    
    
    void setInputDirFromDialog(string message);
    void setOutputDirFromDialog(string message);
    void setOutputFileBaseFromDialog(string message);
    void setAngleRangeFromDialog(string message);
    void setEnergyNoRangeFromDialog(string message);
    void setTargetEnergyNoFromDialog(string message);
    void setTargetAngleNoFromDialog(string message);
    void setRegModeFromDialog(string message);
    void setCntModeFromDialog(string message);
    
    void setInputDir(string inputDir);
    void setOutputDir(string outputDir);
    void setOutputFileBase(string outputfilebase);
    void setAngleRange(string ang_range);
    void setEnergyNoRange(string E_N_range);
    void setTargetEnergyNo(string target_E);
    void setTargetAngleNo(string target_A);
    void setRegMode(string regmode);
    void setCntMode(string cntmode);
    void setLambda_t(float lambda_t_inp);
    void setNumtrial(int numTrial_inp);
    void setNumParallel(int numParallel_inp);
    
};

string IntToString(int number);
string LnumTagString(int LoopNo,string preStr, string postStr);
string EnumTagString(int EnergyNo,string preStr, string postStr);
string AnumTagString(int angleNo,string preStr, string postStr);
string output_flag(string flag, int argc, const char * argv[]);

struct CL_objects {

    cl::Buffer dark_buffer;
    cl::Buffer I0_target_buffer;
    vector<cl::Buffer> I0_sample_buffers;
    cl::Buffer mt_target_buffer;
    
    cl::Buffer p_freefix_buffer;
    vector<cl::Kernel> kernels;
    vector<cl::Kernel> kernels_fit;
    
    cl::Buffer energy_buffer;
};

extern mutex m1,m2;
extern vector<thread> input_th, imageReg_th, fitting_th, output_th, output_th_fit;

#endif
