//
//  OpenCL_analysis.hpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/03/06.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#ifndef CT_XANES_analysis_OpenCL_analysis_hpp
#define CT_XANES_analysis_OpenCL_analysis_hpp

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
#include <cfloat>

#include <time.h>
#include <sys/stat.h>

#define __CL_ENABLE_EXCEPTIONS
#if defined (__APPLE__)  //Mac OS X, iOS?
#define OCL120
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define CL_API_SUFFIX__VERSION_2_0 UNAVAILABLE_ATTRIBUTE
#define CL_API_SUFFIX__VERSION_2_1 UNAVAILABLE_ATTRIBUTE
#define CL_EXT_PREFIX__VERSION_2_1_DEPRECATED
#define CL_EXT_PREFIX__VERSION_2_0_DEPRECATED
#define CL_EXT_PREFIX__VERSION_1_2_DEPRECATED CL_EXTENSION_WEAK_LINK
#define CL_EXT_SUFFIX__VERSION_1_2_DEPRECATED
#define CL_EXT_SUFFIX__VERSION_2_0_DEPRECATED
#include <OpenCL/cl.hpp>
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_SIZE_T_COMPATIBILITY
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
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_SIZE_T_COMPATIBILITY
#ifdef OCL120
#include <CL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif
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
    
    void inputFromFile_mask(char* buffer,ifstream *inp_ifs);
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
    bool CSbool;
    vector<float> CSepsilon;
    vector<float> CSalpha;
    int CSit;
    
    vector<float> fitting_para;
    vector<bool> free_para;
    vector<string> parameter_name;
    
    vector<float> para_upperLimit;
    vector<float> para_lowerLimit;
    vector<float> para_attenuator;
    
    
    float kstart;
    float kend;
    float Rstart;
    float Rend;
    float qstart;
    float qend;
    int kw;
    vector<string> feffxxxxdat_path;
    int EXAFSfittingMode;
    int numShell;
    vector<string> shellname;
    float ini_S02;
    bool S02_freefix;
    vector<vector<float>> EXAFS_iniPara;
    vector<vector<bool>> EXAFS_freeFixPara;
    string edgeJ_dir_path;
    bool useEdgeJ;
	string ej_filebase;
    float Rbkg;
    int bkgFittingMode;
    
    int preEdgeStartEnergyNo;
    int preEdgeEndEnergyNo;
    int postEdgeStartEnergyNo;
    int postEdgeEndEnergyNo;
    float preEdgeStartEnergy;
    float preEdgeEndEnergy;
    float postEdgeStartEnergy;
    float postEdgeEndEnergy;
    
public:
    input_parameter_fitting();
    input_parameter_fitting(string inputfile_path);
    
    float get_kstart();
    float get_kend();
    float get_Rstart();
    float get_Rend();
    float get_qstart();
    float get_qend();
    int get_kw();
    void set_kend(float val);
    vector<string> getFeffxxxxdatPath();
    int getEXAFSfittingMode();
    vector<string> getShellName();
    float getIniS02();
    bool getS02freeFix();
    vector<vector<float>> getEXAFSiniPara();
    vector<vector<bool>> getEXAFSfreeFixPara();
    bool getUseEJ();
    string getInputDir_EJ();
	int getShellNum();
    float getRbkg();
    int getPreEdgeStartEnergyNo();
    int getPreEdgeEndEnergyNo();
    int getPostEdgeStartEnergyNo();
    int getPostEdgeEndEnergyNo();
    void setPreEdgeEnergyNoRange(int startEnergyNo,int endEnergyNo);
    void setPostEdgeEnergyNoRange(int startEnergyNo,int endEnergyNo);
    void setFittingEnergyNoRange(int startEnergyNo,int endEnergyNo);
    float getPreEdgeStartEnergy();
    float getPreEdgeEndEnergy();
    float getPostEdgeStartEnergy();
    float getPostEdgeEndEnergy();
    int getPreEdgeFittingMode();
    
    string getFittingOutputDir();
    string getFittingFileBase();
	string getEJFileBase();
    string getEnergyFilePath();
    float getE0();
    float getStartEnergy();
    float getEndEnergy();
    int getFittingStartEnergyNo();
    int getFittingEndEnergyNo();

    vector<float> getFittingPara();
    vector<bool> getFreeFixPara();
    vector<string> getFittingParaName();
    vector<float> getParaUpperLimit();
    vector<float> getParaLowerLimit();
    vector<float> getParaAttenuator();

	int getNumTrial_fit();
	float getLambda_t_fit();
    bool getCSbool();
    vector<float> getCSepsilon();
    vector<float> getCSalpha();
    int getCSit();
    
    vector<string> funcNameList;
    vector<string> LCFstd_paths;
    int numLCF,numGauss, numLor, numAtan, numErf;
    int numParameter;

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
	void setEJFileBase(string outputfilebase);
    void setEnergyFilePath(string energy_path);
    void setE0(string E0_str);
    void setEnergyRange(string E_range);
    void setFittingStartEnergyNo(int energyNo);
    void setFittingEndEnergyNo(int energyNo);
    
    void setFittingPara(string fitpara_input);
    void setFreeFixPara(string freepara_inp);
    void setFittingParaName(string fitparaname_inp);
    void setValidParaLimit(string limit);
    void setParaAttenuator(string para_input);

	void setLambda_t_fit(float lambda_t_inp);
	void setNumtrial_fit(int numTrial_inp);
    
    void setCSbool(bool CSbool_inp);
    void setCSepsilon(string CSepsilon_inp);
    void setCSalpha(string CSalpha_inp);
    void setCSit(int CSit_inp);
    
    void inputFromFile_fitting(char* buffer,ifstream *inp_ifs);
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
    int startLayer;
    int endLayer;
    
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
    int getStartLayer();
    int getEndLayer();
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
    
    void inputFromFile_reslice(char* buffer, ifstream *inp_ifs);
};

class input_parameter : public input_parameter_mask,public input_parameter_fitting,
                        public input_parameter_reslice
{
    string ocl_plat_dev_list;
    string data_input_dir;
    string data_output_dir;
    string output_filebase;
    string iniFilePath;
    
    int startEnergyNo;
    int endEnergyNo;
    int startAngleNo;
    int endAngleNo;
    int targetEnergyNo;
    int targetAngleNo;
	int scanN;
    int mergeN;
    
    int regMode;
    bool imgRegOutput;
    
    int num_trial;
    float lamda_t;
    
    int numParallel;
    vector<float> reg_inipara;
	vector<float> reg_fixpara;
    
    string rawAngleFilePath;
    string XAFSparameterFilePath;
    bool enableSmoothing;
    
    int imageSizeX;
    int imageSizeY;
    
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
    int getNumTrial();
    int getImageSizeX();
    int getImageSizeY();
    int getImageSizeM();
    float getLambda_t();
    bool getImgRegOutput();
    int getNumParallel();
	int getScanN();
    int getMergeN();
    vector<float> getReg_inipara();
	vector<float> getReg_fixpara();
    string getRawAngleFilePath();
    string getSmoothedEnergyFilePath();
    string getXAFSparameterFilePath();
    bool getSmootingEnable();
    vector<int> numPntsInSmoothedPnts;
	vector<float> smoothedEnergyList;
    int smoothingOffset;
    
    
    
    string getIniFilePath();
    void setIniFilePath(string ini_path);
    void setIniFilePathFromDialog(string message);
    
    void setInputDirFromDialog(string message);
    void setOutputDirFromDialog(string message);
    void setOutputFileBaseFromDialog(string message);
    void setAngleRangeFromDialog(string message);
    void setEnergyNoRangeFromDialog(string message);
    void setTargetEnergyNoFromDialog(string message);
    void setTargetAngleNoFromDialog(string message);
    void setRegModeFromDialog(string message);
    void setReg_iniparaFromDialog(string message);
	void setReg_fixparaFromDialog(string message);
    
    void setInputDir(string inputDir);
    void setOutputDir(string outputDir);
    void setOutputFileBase(string outputfilebase);
    void setAngleRange(string ang_range);
    void setEnergyNoRange(string E_N_range);
    void setTargetEnergyNo(string target_E);
    void setTargetAngleNo(string target_A);
    void setRegMode(string regmode);
    void setLambda_t(float lambda_t_inp);
    void setNumtrial(int numTrial_inp);
    void setNumParallel(int numParallel_inp);
    
    void setRawAngleFilePath(string filepath);
    void setXAFSparameterFilePath(string filepath);
    
    void setImageSizeX(int size);
    void setImageSizeY(int size);
    
    void adjustLayerRange();
    
};

string IntToString(int number);
string numTagString(int tagNum, string preStr, string postStr, int degit);
string LnumTagString(int LoopNo,string preStr, string postStr);
string EnumTagString(int EnergyNo,string preStr, string postStr);
string AnumTagString(int angleNo,string preStr, string postStr);
string output_flag(string flag, int argc, const char * argv[]);

#ifdef XANES_FIT
class fitting_eq {
	float* fitting_para;
	char* free_para;
	float* para_upperlimit;
	float* para_lowerlimit;
	size_t param_size;
	size_t free_param_size;
	vector<string> parameter_name;
	float* freefitting_para;
	float* paraAtten;

public:
	fitting_eq(input_parameter inp);
	fitting_eq();

	float* fit_para();
	char* freefix_para();
	size_t freeParaSize();
	size_t ParaSize();
	string param_name(int i);
	float* freefit_para();
	float* freepara_upperlimit;
	float* freepara_lowerlimit;
	float* paraAttenuator();
	float* lowerLimit();
	float* upperLimit();
	size_t constrain_size;
    
    
    void setFittingEquation(vector<string> fittingFuncList);
    void setInitialParameter(vector<float> iniPara);
    void setFreeFixParameter(vector<char> freefixPara);
    
	vector<vector<float>> C_matrix;
	vector<float> D_vector;

	vector<vector<float>> LCFstd_mt;
	vector<vector<float>> LCFstd_E;
	int numLCF;
	vector<int> LCFstd_size;
	vector<int> funcmode;
	int numFunc;
};
#endif

class CL_objects {
    vector<cl::Kernel> kernels;
    
    public:
    CL_objects();
    
    cl::Buffer dark_buffer;
    cl::Buffer I0_target_buffer;
    vector<cl::Buffer> I0_sample_buffers;
    cl::Buffer mt_target_buffer;
    cl::Buffer p_freefix_buffer;
    
    cl::Program program;
    cl::Kernel getKernel(string kernalName);
    void addKernel(cl::Program program, string kernelName);
    
    cl::Buffer energy_buffer;
	cl::Buffer C_matrix_buffer;
	cl::Buffer D_vector_buffer;
	cl::Buffer freeFix_buffer;
    cl::Image1DArray refSpectra;
    cl::Buffer funcMode_buffer;
#ifdef XANES_FIT
	fitting_eq fiteq;
#endif
};

extern mutex m1,m2;
extern vector<thread> input_th, imageReg_th, fitting_th, output_th, output_th_fit;

#endif
