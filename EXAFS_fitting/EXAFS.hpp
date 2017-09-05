//
//  EXAFS.hpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2017/07/28.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#ifndef EXAFS_h
#define EXAFS_h

#include "OpenCL_analysis.hpp"

#define KGRID 0.05
#define RGRID 0.03067961576
#define MAX_KRSIZE 400
#define MAX_KQ 20
#define MAX_R 12.2718463031
#define WIN_DK 0.5
#define WIN_DR 0.0
#define FFT_SIZE 2048

class FEFF_shell {
    vector<float> kw;
    vector<float> real2phc;
    vector<float> mag;
    vector<float> phase;
    vector<float> redFactor;
    vector<float> lambda;
    vector<float> real_p;
    float reff;
    float degen;
    int numPnts;
    
public:
    FEFF_shell(string path);
    
    vector<float> getk();
    vector<float> getReal2phc();
    vector<float> getMag();
    vector<float> getPhase();
    vector<float> getRedFactor();
    vector<float> getLambda();
    vector<float> getReal_p();
    float getReff();
    float getDegen();
    int getNumPnts();
};


class shellObjects{
    //raw image object from feffxxx.dat files
    /*cl::Buffer k_raw;
    cl::Image1D real_2phc_raw;
    cl::Image1D mag_raw;
    cl::Image1D phase_raw;
    cl::Image1D redFactor_raw;
    cl::Image1D lambda_raw;
    cl::Image1D real_p_raw;*/
    
    //redimentioned image object for fitting analysis
    cl::Buffer real_2phc;
    cl::Buffer mag;
    cl::Buffer phase;
    cl::Buffer redFactor;
    cl::Buffer lambda;
    cl::Buffer real_p;
    
    cl::Buffer CN;
    cl::Buffer dR;
    cl::Buffer dE0;
    cl::Buffer ss;
    cl::Buffer E0imag;
    cl::Buffer C3;
    cl::Buffer C4;
    
    cl::Context context;
    cl::CommandQueue queue;
    
    cl::Kernel kernel_feff;
    cl::Kernel kernel_chiout;
    cl::Kernel kernel_jacob_k;
    cl::Kernel kernel_CNw;
    cl::Kernel kernel_update;
    cl::Kernel kernel_UR;
    cl::Kernel kernel_OBD;
    cl::Kernel kernel_contrain1;
    cl::Kernel kernel_contrain2;
    
    
    int imageSizeX;
    int imageSizeY;
    int imageSizeM;
    
    float Reff;
    
    vector<bool> freeFixPara;
public:
    shellObjects(cl::CommandQueue queue, cl::Program program, FEFF_shell shell,
                 int SizeX, int SizeY);
    int outputChiFit(cl::Buffer chi, cl::Buffer S02, int kw, float kstart, float kend);
    int outputJacobiank(cl::Buffer Jacob, cl::Buffer S02, int kw, int paramode,
                        float kstart, float kend, bool useRealPart);
    int inputIniCN(float iniCN, cl::Buffer edgeJ);
    int inputIniR(float inidR);
    int inputInidE0(float iniE0);
    int inputIniss(float iniSS);
    
    void setFreeFixPara(string paraname, bool val);
    bool getFreeFixPara(string paraname);
    bool getFreeFixPara(int paramode);
    int getFreeParaSize();
    
    int copyPara(cl::Buffer dstPara, int offsetN,int paramode);
    int updatePara(cl::Buffer dp, int paramode, int z_id);
    int restorePara(cl::Buffer para_backup, int offsetN, cl::Buffer rho_img, int paramode);
    
    int readParaImage(float* paraData, int paramode);
    
    int constrain1(cl::Buffer eval_img, cl::Buffer C_mat, int cnum, int pnum, int paramode);
    int constrain2(cl::Buffer eval_img, cl::Buffer edgeJ, cl::Buffer C_mat, cl::Buffer D_vec, cl::Buffer C2_vec, int cnum, int pnum, int paramode);
};

int readRawFile(string filepath_input,float *binImgf, int imageSizeM);
int readRawFile_offset(string filepath_input,float *binImgf, int64_t offset, int64_t size);
int outputRawFile_stream(string filename,float*data,size_t pnt_size);

vector<int> GPUmemoryControl(int imageSizeX, int imageSizeY,int ksize,int Rsize,int qsize,
                             int fittingMode, int num_fpara, int shellnum, cl::CommandQueue queue);
int createSpinFactor(cl::Buffer w_buffer, cl::CommandQueue queue, cl::Program program);
int FFT(cl::Buffer chi, cl::Buffer FTchi,
        cl::Buffer w_buffer, cl::CommandQueue queue, cl::Program program,
        int imageSizeX, int imageSizeY, int FFTimageSizeY, int offsetY,
        int koffset, int ksize, int Roffset, int Rsize);
int IFFT(cl::Buffer FTchi, cl::Buffer chiq,
         cl::Buffer w_buffer, cl::CommandQueue queue, cl::Program program,
         int imageSizeX, int imageSizeY, int FFTimageSizeY, int offsetY,
         int Roffset, int Rsize, int qoffset, int qsize);
int EXAFS_kFit(cl::CommandQueue queue, cl::Program program,
               cl::Buffer chidata, cl::Buffer S02, cl::Buffer Rfactor,  vector<shellObjects> shObj,
               int kw, float kstart, float kend, int imageSizeX, int imageSizeY,
               bool freeS02,int numTrial,float lambda, int contrainSize,  cl::Buffer edgeJ,
               cl::Buffer C_matrix_buff, cl::Buffer D_vector_buff, cl::Buffer C2_vector_buff,
               bool CSbool, int CSit, cl::Buffer CSlambda_buff);
int EXAFS_RFit(cl::CommandQueue queue, cl::Program program, cl::Buffer w_factor,
               cl::Buffer FTchidata, cl::Buffer S02, cl::Buffer Rfactor,  vector<shellObjects> shObj,
               int kw, float kstart, float kend, float Rstart, float Rend,
               int imageSizeX, int imageSizeY, int FFTimageSizeY,
               bool freeS02,int numTrial,float lambda, int contrainSize, cl::Buffer edgeJ,
               cl::Buffer C_matrix_buff, cl::Buffer D_vector_buff, cl::Buffer C2_vector_buff,
               bool CSbool, int CSit, cl::Buffer CSlambda_buff);
int EXAFS_qFit(cl::CommandQueue queue, cl::Program program, cl::Buffer w_factor,
               cl::Buffer chiqdata, cl::Buffer S02, cl::Buffer Rfactor,  vector<shellObjects> shObj,
               int kw, float kstart, float kend, float Rstart, float Rend, float qstart, float qend,
               int imageSizeX, int imageSizeY, int FFTimageSizeY,
               bool freeS02,int numTrial,float lambda, int contrainSize, cl::Buffer edgeJ,
               cl::Buffer C_matrix_buff, cl::Buffer D_vector_buff, cl::Buffer C2_vector_buff,
               bool CSbool, int CSit, cl::Buffer CSlambda_buff);
int ChiData_k(cl::CommandQueue queue, cl::Program program,
              cl::Buffer chiData, vector<float*> chiData_pointer,
              int kw, float kstart, float kend, int imageSizeX, int imageSizeY,
              bool imgStckOrChiStck, int offsetM);
int ChiData_R(cl::CommandQueue queue, cl::Program program,
              cl::Buffer FTchiData, vector<float*> chiData_pointer, cl::Buffer w_factor,
              int kw, float kstart, float kend, float Rstart, float Rend,
              int imageSizeX, int imageSizeY, int FFTimageSizeY,bool imgStckOrChiStck, int offsetM);
int ChiData_q(cl::CommandQueue queue, cl::Program program,
              cl::Buffer chiqData, vector<float*> chiData_pointer, cl::Buffer w_factor,
              int kw, float kstart, float kend, float Rstart, float Rend, float qstart, float qend,
              int imageSizeX, int imageSizeY, int FFTimageSizeY, bool imgStckOrChiStck, int offsetM);

int EXAFS_fit_ocl(input_parameter inp, OCL_platform_device plat_dev_list,vector<FEFF_shell> shell);

int createContrainMatrix(vector<string> contrain_eqs, vector<string> fparaName,
                         vector<vector<float>> *C_matrix, vector<float> *D_vector,int cotrainOffset);
int correctBondDistanceContrain(vector<vector<float>> *C_matrix, vector<float> *D_vector,
                                int fpnum, float Reff);

#endif /* EXAFS_h */
