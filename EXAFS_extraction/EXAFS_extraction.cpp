//
//  EXAFS_extraction.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2017/08/09.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#include "EXAFS_extraction.hpp"
#include "XANES_fit_cl.hpp"
#include "EXAFS_fit_cl.hpp"
#include "3D_FFT_cl.hpp"
#include "LevenbergMarquardt_cl.hpp"
#include "EXAFSextraction_cl.hpp"

int extraction_output_thread(int AngleNo,string output_dir,
                            float* chiData, float* bkgData, float* edgeJumpData, int imageSizeM){
    
    ostringstream oss;
    string fileName_output= output_dir+ AnumTagString(AngleNo,"chi0/i", ".raw");
    oss << "output file: " << fileName_output << endl;
    outputRawFile_stream(fileName_output,chiData,imageSizeM*MAX_KRSIZE);
    fileName_output= output_dir+ AnumTagString(AngleNo,"bkg/i", ".raw");
    oss << "output file: " << fileName_output << endl;
    outputRawFile_stream(fileName_output,bkgData,imageSizeM);
    fileName_output= output_dir+ AnumTagString(AngleNo,"edgeJ/i", ".raw");
    oss << "output file: " << fileName_output << endl;
    outputRawFile_stream(fileName_output,edgeJumpData,imageSizeM);
    cout << oss.str();

    delete [] chiData;
    delete [] bkgData;
    delete [] edgeJumpData;
    
    return 0;
}


int fitting(cl::CommandQueue queue, cl::Program program,
            cl::Buffer energy, cl::Buffer mt_img, cl::Buffer fp_img,
            cl::Image1DArray refSpectra, fitting_eq fiteq, float lambda, int numTrial,
            int fittingStartEnergyNo, int fittingEndEnergyNo, int imgSizeX, int imgSizeY){
    
    string errorzone = "0";
    try {
        cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
        string devicename = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>();
        size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        

        int paramsize = (int)fiteq.ParaSize();
        const int imgSizeM = imgSizeX*imgSizeY;
        const cl::NDRange local_item_size(min(imgSizeX,(int)maxWorkGroupSize),1,1);
        const cl::NDRange global_item_size(imgSizeX,imgSizeY,1);
        const cl::NDRange global_item_size2(imgSizeX,imgSizeY,paramsize);
        
        
        cl::Buffer Jacobian_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM*paramsize, 0, NULL);
        cl::Buffer tJdF_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM*paramsize, 0, NULL);
        cl::Buffer tJJ_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM*paramsize*(paramsize+1)/2, 0, NULL);
        cl::Buffer inv_tJJ_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM*paramsize*(paramsize+1)/2, 0, NULL);
        cl::Buffer nyu_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM, 0, NULL);
        cl::Buffer lambda_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM, 0, NULL);
        cl::Buffer chi2_old_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM, 0, NULL);
        cl::Buffer chi2_new_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM, 0, NULL);
        cl::Buffer fp_cnd_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM*paramsize, 0, NULL);
        cl::Buffer dp_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM*paramsize, 0, NULL);
        cl::Buffer dL_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM, 0, NULL);
        cl::Buffer rho_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM, 0, NULL);
        cl::Buffer freeFix_buff(context, CL_MEM_READ_WRITE, sizeof(cl_int)*paramsize, 0, NULL);
        cl::Buffer funcMode_buff(context, CL_MEM_READ_WRITE, sizeof(cl_int)*fiteq.funcmode.size(), 0, NULL);
        
        
        //cout<<ret<<endl;
        cl::Kernel kernel_chi2tJdFtJJ(program,"chi2_tJdF_tJJ_Stack");
        cl::Kernel kernel_chi2new(program,"chi2Stack");
        cl::Kernel kernel_LM(program,"LevenbergMarquardt");
        cl::Kernel kernel_dL(program,"estimate_dL");
        cl::Kernel kernel_cnd(program,"updatePara");
        cl::Kernel kernel_evalUpdate(program,"evaluateUpdateCandidate");
        cl::Kernel kernel_UorH(program,"updateOrHold");
        cl::Kernel kernel_mask(program,"setMask");
        
        
        
        errorzone = "inputing parameters to GPU";
        //input LM parameters
        queue.enqueueFillBuffer(lambda_buff, (cl_float)lambda, 0, sizeof(cl_float)*imgSizeM,NULL,NULL);
        queue.finish();
        queue.enqueueFillBuffer(nyu_buff, (cl_float)2.0f, 0, sizeof(cl_float)*imgSizeM,NULL,NULL);
        queue.finish();
        for (int i=0; i<paramsize; i++) {
            queue.enqueueFillBuffer(fp_img, (cl_float)fiteq.fit_para()[i], sizeof(cl_float)*imgSizeM*i, sizeof(cl_float)*imgSizeM,NULL,NULL);
            queue.finish();
        }
        queue.enqueueWriteBuffer(freeFix_buff, CL_TRUE, 0, sizeof(cl_char)*paramsize, fiteq.freefix_para());
        queue.enqueueWriteBuffer(funcMode_buff, CL_TRUE, 0, sizeof(cl_int)*fiteq.numFunc, &(fiteq.funcmode[0]), NULL, NULL);
        
        //set kernel arguments
        //chi2(old), tJdF, tJJ calculation
        kernel_chi2tJdFtJJ.setArg(0, mt_img);
        kernel_chi2tJdFtJJ.setArg(1, fp_img);
        kernel_chi2tJdFtJJ.setArg(2, refSpectra);
        kernel_chi2tJdFtJJ.setArg(3, chi2_old_buff);
        kernel_chi2tJdFtJJ.setArg(4, tJdF_buff);
        kernel_chi2tJdFtJJ.setArg(5, tJJ_buff);
        kernel_chi2tJdFtJJ.setArg(6, energy);
        kernel_chi2tJdFtJJ.setArg(7, funcMode_buff);
        kernel_chi2tJdFtJJ.setArg(8, freeFix_buff);
        kernel_chi2tJdFtJJ.setArg(9, fittingStartEnergyNo);
        kernel_chi2tJdFtJJ.setArg(10, fittingEndEnergyNo);        
        //chi2(new) calculation
        kernel_chi2new.setArg(0, mt_img);
        kernel_chi2new.setArg(1, fp_cnd_img);
        kernel_chi2new.setArg(2, refSpectra);
        kernel_chi2new.setArg(3, chi2_new_buff);
        kernel_chi2new.setArg(4, energy);
        kernel_chi2new.setArg(5, funcMode_buff);
        kernel_chi2new.setArg(6, fittingStartEnergyNo);
        kernel_chi2new.setArg(7, fittingEndEnergyNo);
        //LM equation
        kernel_LM.setArg(0, tJdF_buff);
        kernel_LM.setArg(1, tJJ_buff);
        kernel_LM.setArg(2, dp_img);
        kernel_LM.setArg(3, lambda_buff);
        kernel_LM.setArg(4, freeFix_buff);
        kernel_LM.setArg(5, inv_tJJ_buff);
        //estimate dL
        kernel_dL.setArg(0, dp_img);
        kernel_dL.setArg(1, tJJ_buff);
        kernel_dL.setArg(2, tJdF_buff);
        kernel_dL.setArg(3, lambda_buff);
        kernel_dL.setArg(4, dL_buff);
        //estimate fp cnd
        kernel_cnd.setArg(0, fp_cnd_img);
        kernel_cnd.setArg(1, dp_img);
        kernel_cnd.setArg(2, (cl_int)0);
        kernel_cnd.setArg(3, (cl_int)0);
        //evaluate update
        kernel_evalUpdate.setArg(0, tJdF_buff);
        kernel_evalUpdate.setArg(1, tJJ_buff);
        kernel_evalUpdate.setArg(2, lambda_buff);
        kernel_evalUpdate.setArg(3, nyu_buff);
        kernel_evalUpdate.setArg(4, chi2_old_buff);
        kernel_evalUpdate.setArg(5, chi2_new_buff);
        kernel_evalUpdate.setArg(6, dL_buff);
        kernel_evalUpdate.setArg(7, rho_buff);
        //update or hold parameter
        kernel_UorH.setArg(0, fp_img);
        kernel_UorH.setArg(1, fp_cnd_img);
        kernel_UorH.setArg(2, rho_buff);
        


        //L-M trial
        for (int trial=0; trial<numTrial; trial++) {
            errorzone = "calculating chi2 old, tJJ, tJdF";
            //chi2(old), tJdF, tJJ calculation
            queue.enqueueNDRangeKernel(kernel_chi2tJdFtJJ, NULL, global_item_size, local_item_size, NULL, NULL);
            queue.finish();
            
            errorzone = "L-M";
            //L-M equation
            queue.enqueueNDRangeKernel(kernel_LM, NULL, global_item_size, local_item_size, NULL, NULL);
            queue.finish();
            
            
            errorzone = "estimating dL";
            //estimate dL
            queue.enqueueNDRangeKernel(kernel_dL, NULL, global_item_size, local_item_size, NULL, NULL);
            queue.finish();
            
            
            errorzone = "estimating fp_cnd";
            //estimate fp candidate
            queue.enqueueCopyBuffer(fp_img, fp_cnd_img, 0, 0, sizeof(cl_float)*imgSizeM*paramsize);
            queue.enqueueNDRangeKernel(kernel_cnd, NULL, global_item_size2, local_item_size, NULL, NULL);
            queue.finish();
            
            
            errorzone = "calclating new chi2";
            //chi2(new) calculation
            queue.enqueueNDRangeKernel(kernel_chi2new, NULL, global_item_size, local_item_size, NULL, NULL);
            queue.finish();
            
            errorzone = "evaluateUpdateCandidate";
            //evaluate rho
            queue.enqueueNDRangeKernel(kernel_evalUpdate, NULL, global_item_size, local_item_size, NULL, NULL);
            queue.finish();
            //update or hold parameter
            queue.enqueueNDRangeKernel(kernel_UorH, NULL, global_item_size2, local_item_size, NULL, NULL);
            queue.finish();
            
        }
    }catch (const cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")"<< errorzone << endl;
    }
    
    return 0;
}



int PreEdgeRemoval(cl::CommandQueue queue, cl::Program program,
                   cl::Buffer mt_img, cl::Buffer energy, cl::Buffer bkg_img,
                   input_parameter inp, int imagesizeX, int imagesizeY,
                   float E0, int startEn, int endEn, int fitmode){
    
    
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
    string devicename = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>();
    size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    int imageSizeM = imagesizeX*imagesizeY;
    const cl::NDRange local_item_size(min(imagesizeX,(int)maxWorkGroupSize),1,1);
    const cl::NDRange global_item_size(imagesizeX,imagesizeY,1);
    
    fitting_eq fiteq;
    vector<string> funcList;
    vector<char> freefixP;
    vector<float> iniP;
    switch (fitmode) {
        case 0: //line
        funcList.push_back("line");
        freefixP.push_back(1);
        freefixP.push_back(1);
        iniP.push_back(0.0f);
        iniP.push_back(-0.001f);
        break;
        
        case 1: //Victreen
        funcList.push_back("Victoreen");
        freefixP.push_back(1);
        freefixP.push_back(1);
        freefixP.push_back(1);
        freefixP.push_back(0);
        iniP.push_back(0.0f);
        iniP.push_back(0.0f);
        iniP.push_back(0.0f);
        iniP.push_back(E0);
        break;
        
        case 2: //McMaster
        funcList.push_back("McMaster");
        freefixP.push_back(1);
        freefixP.push_back(1);
        freefixP.push_back(0);
        iniP.push_back(0.0f);
        iniP.push_back(0.0f);
        iniP.push_back(E0);
        break;
        
        default:
        break;
    }
    fiteq.setFittingEquation(funcList);
    fiteq.setInitialParameter(iniP);
    fiteq.setFreeFixParameter(freefixP);
    
    cl::Buffer fp_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM*fiteq.ParaSize(), 0, NULL);
    cl::ImageFormat format(CL_R, CL_FLOAT);
    cl::Image1DArray refSpectra(context,CL_MEM_READ_WRITE,format,fiteq.numLCF,IMAGE_SIZE_E, 0, NULL);
    
    //fitting
    fitting(queue, program, energy, mt_img, fp_img, refSpectra, fiteq, inp.getLambda_t_fit(), inp.getNumTrial(), startEn, endEn, imagesizeX, imagesizeY);
    
    //estimate bkg image from fp image
    cl::Kernel kernel_bkg(program,"estimateBkg");;
    kernel_bkg.setArg(0, bkg_img);
    kernel_bkg.setArg(1, fp_img);
    kernel_bkg.setArg(2, (cl_int)fitmode);
    kernel_bkg.setArg(3, (cl_float)E0);
    queue.enqueueNDRangeKernel(kernel_bkg, NULL, global_item_size, local_item_size, NULL, NULL);
    queue.finish();
    
    return 0;
}


int PostEdgeEstimation(cl::CommandQueue queue, cl::Program program,
                       cl::Buffer mt_img, cl::Buffer energy,
                       cl::Buffer bkg_img, cl::Buffer edgeJ_img,
                       input_parameter inp, int imagesizeX, int imagesizeY,
                       int startEn, int endEn){
    
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
    string devicename = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>();
    size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    int imageSizeM = imagesizeX*imagesizeY;
    const cl::NDRange local_item_size(min(imagesizeX,(int)maxWorkGroupSize),1,1);
    const cl::NDRange global_item_size(imagesizeX,imagesizeY,1);
    
    fitting_eq fiteq;
    vector<string> funcList;
    vector<char> freefixP;
    vector<float> iniP;
    funcList.push_back("3rdPolynomical");
    fiteq.setFittingEquation(funcList);
    freefixP.push_back(1);
    freefixP.push_back(1);
    freefixP.push_back(1);
    freefixP.push_back(1);
    iniP.push_back(0.0f);
    iniP.push_back(0.0f);
    iniP.push_back(0.0f);
    iniP.push_back(0.0f);
    fiteq.setInitialParameter(iniP);
    fiteq.setFreeFixParameter(freefixP);
    
    cl::Buffer fp_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM*fiteq.ParaSize(), 0, NULL);
    cl::ImageFormat format(CL_R, CL_FLOAT);
    cl::Image1DArray refSpectra(context,CL_MEM_READ_WRITE,format,fiteq.numLCF,IMAGE_SIZE_E, 0, NULL);
    
    //fitting
    fitting(queue, program, energy, mt_img, fp_img, refSpectra, fiteq, inp.getLambda_t_fit(), inp.getNumTrial(), startEn, endEn, imagesizeX, imagesizeY);
    
    //estimate bkg image from fp image
    cl::Kernel kernel_ej(program,"estimateEJ");;
    kernel_ej.setArg(0, edgeJ_img);
    kernel_ej.setArg(1, bkg_img);
    kernel_ej.setArg(2, fp_img);
    queue.enqueueNDRangeKernel(kernel_ej, NULL, global_item_size, local_item_size, NULL, NULL);
    queue.finish();
    
    return 0;
}


int SplineBkgRemoval(cl::CommandQueue queue, cl::Program program,
                     cl::Buffer mt_img, cl::Buffer energy, cl::Buffer w_factor, cl::Buffer chi_img,
                     input_parameter inp, int imagesizeX, int imagesizeY, int num_energy,
                     float kstart, float kend, float Rbkg, int kw){
    
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
    string devicename = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>();
    size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    int imageSizeM = imagesizeX*imagesizeY;
    
    int ksize=ceil((min(float(kend + WIN_DK),(float)MAX_KQ)-max(float(kstart-WIN_DK),0.0f))/KGRID);
    int koffset=floor(max((float)(kstart - WIN_DK), 0.0f) / KGRID);
    int Rsize=ceil((min(float(Rbkg + WIN_DR),(float)MAX_R))/RGRID);
    int Roffset=0;
    int N_ctrlP = (int)floor(2.0*(kend-kstart)*Rbkg/PI)+1;
    float lambda = inp.getLambda_t_fit();
    int numTrial = inp.getNumTrial();
    int FFTimageSizeY = imagesizeY;
    
    //CL objects
    cl::Buffer chiData(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*ksize,0,NULL);
    cl::Buffer FTchiStd(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*Rsize,0,NULL);
    cl::Buffer chiFit(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*ksize,0,NULL);
    cl::Buffer FTchiFit(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*Rsize,0,NULL);
    cl::Buffer dF2_old(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
    cl::Buffer dF2_new(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
    cl::Buffer tJJ(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*N_ctrlP*(N_ctrlP+1)/2,0,NULL);
    cl::Buffer inv_tJJ(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*N_ctrlP*(N_ctrlP+1)/2,0,NULL);
    cl::Buffer tJdF(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*N_ctrlP,0,NULL);
    vector<cl::Buffer> Jacobian;
    vector<cl::Buffer> para_backup;
    for(int i=0;i<N_ctrlP;i++){
        Jacobian.push_back(cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*Rsize,0,NULL));
    }
    cl::Buffer J_dummy(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*ksize,0,NULL);
    cl::Buffer nyu_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
    cl::Buffer lambda_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
    cl::Buffer fp_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM*N_ctrlP, 0, NULL);
    cl::Buffer fp_cnd_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM*N_ctrlP, 0, NULL);
    cl::Buffer dp_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM*N_ctrlP, 0, NULL);
    cl::Buffer dL_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
    cl::Buffer rho_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
    cl::Buffer p_fix(context,CL_MEM_READ_WRITE,sizeof(cl_char)*N_ctrlP,0,NULL);
    queue.enqueueFillBuffer(p_fix, (cl_char)1, 0, sizeof(cl_char)*N_ctrlP);
    queue.enqueueFillBuffer(lambda_buff, (cl_float)lambda, 0, sizeof(float)*imageSizeM);
    queue.enqueueFillBuffer(nyu_buff, (cl_float)2.0f, 0, sizeof(float)*imageSizeM);
    queue.enqueueFillBuffer(FTchiStd, (cl_float)0.0f, 0, sizeof(float)*imageSizeM*Rsize);
    
    
    //work itemsize settings
    const cl::NDRange local_item_size(min((int)maxWorkGroupSize,imagesizeX),1,1);
    const cl::NDRange global_item_size(imagesizeX,imagesizeY,1);
    const cl::NDRange global_item_size_stack(imagesizeX,imagesizeY,ksize);
    const cl::NDRange global_item_size_para(imagesizeX,imagesizeY,N_ctrlP);
    const cl::NDRange global_item_offset1(0,0,koffset);
    
    
    
    //redimension of mt_img stack to k-space chi stack
    queue.enqueueFillBuffer(chi_img, (cl_float)0.0f, 0, sizeof(cl_float)*MAX_KRSIZE);
    cl::Kernel kernel_redim(program,"redimension_mt2chi");;
    kernel_redim.setArg(0, mt_img);
    kernel_redim.setArg(1, chi_img);
    kernel_redim.setArg(2, energy);
    kernel_redim.setArg(3, (cl_int)num_energy);
    kernel_redim.setArg(4, (cl_int)0);
    kernel_redim.setArg(5, (cl_int)MAX_KRSIZE);
    queue.enqueueNDRangeKernel(kernel_redim, NULL, global_item_size, local_item_size, NULL, NULL);
    queue.finish();
    
    
    
    //convert to complex chi
    cl::Kernel kernel_cmplx(program,"chi2cmplxChi_imgStck");
    kernel_cmplx.setArg(0, chi_img);
    kernel_cmplx.setArg(1, chiData);
    kernel_cmplx.setArg(3, (cl_int)koffset);
    kernel_cmplx.setArg(4, (cl_int)0);
    for (int kn=0; kn<ksize; kn++) {
        kernel_cmplx.setArg(2, (cl_int)kn);
        const cl::NDRange global_item_offset2(0,0,koffset+kn);
        queue.enqueueNDRangeKernel(kernel_cmplx, global_item_offset2, global_item_size, local_item_size, NULL, NULL);
        queue.finish();
    }
    
    
    
    //fitting loop
    //kernel settings
    cl::Kernel kernel_kwindow(program,"hanningWindowFuncIMGarray");
    kernel_kwindow.setArg(1, (cl_float)kstart);
    kernel_kwindow.setArg(2, (cl_float)kend);
    kernel_kwindow.setArg(3, (cl_float)WIN_DK);
    kernel_kwindow.setArg(4, (cl_float)KGRID);
    cl::Kernel kernel_tJJ(program,"estimate_tJJ");
    kernel_tJJ.setArg(0, tJJ);
    kernel_tJJ.setArg(4, (cl_int)Rsize);
    cl::Kernel kernel_tJdF(program,"estimate_tJdF");
    kernel_tJdF.setArg(0, tJdF);
    kernel_tJdF.setArg(2, FTchiStd);
    kernel_tJdF.setArg(3, FTchiFit);
    kernel_tJdF.setArg(5, (cl_int)Rsize);
    cl::Kernel kernel_dF2(program,"estimate_dF2");
    kernel_dF2.setArg(1, FTchiStd);
    kernel_dF2.setArg(2, FTchiFit);
    kernel_dF2.setArg(3, (cl_int)Rsize);
    cl::Kernel kernel_LM(program,"LevenbergMarquardt");
    kernel_LM.setArg(0, tJdF);
    kernel_LM.setArg(1, tJJ);
    kernel_LM.setArg(2, dp_img);
    kernel_LM.setArg(3, lambda_buff);
    kernel_LM.setArg(4, p_fix);
    kernel_LM.setArg(5, inv_tJJ);
    cl::Kernel kernel_dL(program,"estimate_dL");
    kernel_dL.setArg(0, dp_img);
    kernel_dL.setArg(1, tJJ);
    kernel_dL.setArg(2, tJdF);
    kernel_dL.setArg(3, lambda_buff);
    kernel_dL.setArg(4, dL_buff);
    cl::Kernel kernel_update(program,"updatePara");
    kernel_update.setArg(0, dp_img);
    kernel_update.setArg(1, fp_cnd_img);
    kernel_update.setArg(3, (cl_int)0);
    kernel_update.setArg(4, (cl_int)0);
    cl::Kernel kernel_eval(program,"evaluateUpdateCandidate");
    kernel_eval.setArg(0, tJdF);
    kernel_eval.setArg(1, tJJ);
    kernel_eval.setArg(2, lambda_buff);
    kernel_eval.setArg(3, nyu_buff);
    kernel_eval.setArg(4, dF2_old);
    kernel_eval.setArg(5, dF2_new);
    kernel_eval.setArg(6, dL_buff);
    kernel_eval.setArg(7, rho_buff);
    cl::Kernel kernel_UorH(program,"updateOrHold");
    kernel_UorH.setArg(0, fp_img);
    kernel_UorH.setArg(1, fp_cnd_img);
    kernel_UorH.setArg(2, rho_buff);
    
    cl_float2 iniChi={0.0f,0.0f};
    for (int trial=0; trial<numTrial; trial++) {
        //reset dF2, tJJ, tJdF
        queue.enqueueFillBuffer(dF2_old, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM);
        queue.enqueueFillBuffer(dF2_new, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM);
        queue.enqueueFillBuffer(tJJ, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM*N_ctrlP*(N_ctrlP+1)/2);
        queue.enqueueFillBuffer(tJdF, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM*N_ctrlP);
        
        
        //chiFit
        //reset chiFit
        queue.enqueueFillBuffer(chiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
        queue.enqueueFillBuffer(FTchiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*Rsize);
        //estimate chiFit
        /*
         code Bspline here (using fp_img, chiData_img to chifit_img)
         */
        //weighting by window function
        kernel_kwindow.setArg(0, chiFit);
        queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset1,global_item_size_stack,local_item_size, NULL, NULL);
        queue.finish();
        //FFT
        for (int offsetY=0; offsetY<imagesizeY; offsetY += FFTimageSizeY) {
            FFT(chiFit, FTchiFit, w_factor, queue, program,
                imagesizeX, imagesizeY, FFTimageSizeY, offsetY,
                koffset,ksize,Roffset,Rsize);
        }
        
        
        //Jacobian
        //reset Jacobian
        for(int i=0;i<N_ctrlP;i++){
            queue.enqueueFillBuffer(Jacobian[i], iniChi, 0, sizeof(cl_float2)*imageSizeM*Rsize);
            queue.enqueueFillBuffer(J_dummy, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
            /*
             code Bspline jacobian here (using fp_img, chiData_img to J_dummy)
             */
            //weighting by window function
            kernel_kwindow.setArg(0, J_dummy);
            queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset1,global_item_size_stack, local_item_size, NULL, NULL);
            queue.finish();
            
            //FFT
            for (int offsetY=0; offsetY<imagesizeY; offsetY += FFTimageSizeY) {
                FFT(J_dummy, Jacobian[0], w_factor, queue, program,
                    imagesizeX, imagesizeY, FFTimageSizeY, offsetY,
                    koffset,ksize,Roffset,Rsize);
            }
        }

        
        
        //estimate tJJ and tJdF
        int pn=0;
        for(int fpn1=0; fpn1<N_ctrlP;fpn1++){
            kernel_tJJ.setArg(1, Jacobian[fpn1]);
            kernel_tJdF.setArg(1, Jacobian[fpn1]);
            kernel_tJdF.setArg(4, (cl_int)fpn1);
            queue.enqueueNDRangeKernel(kernel_tJdF,NULL,global_item_size,local_item_size, NULL, NULL);
            for(int fpn2=fpn1;fpn2<N_ctrlP;fpn2++){
                kernel_tJJ.setArg(2, Jacobian[fpn2]);
                kernel_tJJ.setArg(3, (cl_int)pn);
                queue.enqueueNDRangeKernel(kernel_tJJ,NULL,global_item_size, local_item_size, NULL, NULL);
                pn++;
            }
        }
        queue.finish();
        
        
        //estimate dF2
        kernel_dF2.setArg(0, dF2_old);
        queue.enqueueNDRangeKernel(kernel_dF2,NULL,global_item_size, local_item_size, NULL, NULL);
        queue.finish();
        
        
        //Levenberg-Marquardt
        queue.enqueueNDRangeKernel(kernel_LM,NULL,global_item_size,local_item_size, NULL, NULL);
        queue.finish();
        
        
        //estimate dL
        queue.enqueueNDRangeKernel(kernel_dL,NULL,global_item_size,local_item_size, NULL, NULL);
        queue.finish();
        
        
        //update paramater as candidate
        queue.enqueueCopyBuffer(fp_img, fp_cnd_img, 0, 0, sizeof(cl_float)*imageSizeM*N_ctrlP);
        queue.enqueueNDRangeKernel(kernel_update, NULL, global_item_size_para, local_item_size, NULL, NULL);
        queue.finish();
        
        
        //estimate new dF2
        //reset chiFit
        queue.enqueueFillBuffer(chiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
        queue.enqueueFillBuffer(FTchiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*Rsize);
        //estimate chiFit
        /*
         code Bspline here (using fp_cnd_img, chiData_img to chifit_img)
         */
        //weighting by window function
        kernel_kwindow.setArg(0, chiFit);
        queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset1,global_item_size_stack,local_item_size, NULL, NULL);
        queue.finish();
        //FFT
        for (int offsetY=0; offsetY<imagesizeY; offsetY += FFTimageSizeY) {
            FFT(chiFit, FTchiFit, w_factor, queue, program,
                imagesizeX, imagesizeY, FFTimageSizeY, offsetY,
                koffset,ksize,Roffset,Rsize);
        }
        //estimate dF2
        kernel_dF2.setArg(0, dF2_new);
        queue.enqueueNDRangeKernel(kernel_dF2,NULL,global_item_size_stack, local_item_size, NULL, NULL);
        
        
        //evaluate updated parameter (update to new para or hold at old para )
        //evaluate rho
        queue.enqueueNDRangeKernel(kernel_eval,NULL,global_item_size, local_item_size,NULL,NULL);
        queue.finish();
        //update or hold parameter
        queue.enqueueNDRangeKernel(kernel_UorH,NULL,global_item_size_para,local_item_size,NULL, NULL);
        queue.finish();
    }
    
    
    //estimate final chiFit(spline removed chiData)
    //reset chiFit
    queue.enqueueFillBuffer(chiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
    queue.enqueueFillBuffer(FTchiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*Rsize);
    //estimate chiFit
    /* code Bspline here code Bspline here 
     (using fp_img, chiData_img to chifit_img) */
    
    
    //convert final chiFit(spline removed chiData) to chi_img
    cl::Kernel kernel_c2real(program,"convert2realChi");
    kernel_c2real.setArg(0, chiFit);
    kernel_c2real.setArg(1, chi_img);
    queue.enqueueNDRangeKernel(kernel_c2real,global_item_offset1,global_item_size_stack,local_item_size,NULL, NULL);
    queue.finish();
    
    
    return 0;
}


int extraction_thread(cl::CommandQueue queue, vector<cl::Program> program,
                     int AngleNo, int thread_id,input_parameter inp,
                     cl::Buffer energy, cl::Buffer w_factor, int numEnergy,vector<float*> mt_vec){
    
    
    int preEdgeStartEnNo  = inp.getPreEdgeStartEnergyNo();
    int preEdgeEndEnNo    = inp.getPreEdgeEndEnergyNo();
    int postEdgeStartEnNo = inp.getPostEdgeStartEnergyNo();
    int postEdgeEndEnNo   = inp.getPreEdgeEndEnergyNo();
    int imagesizeX = inp.getImageSizeX();
    int imagesizeY = inp.getImageSizeY();
    float E0 = inp.getE0();
    int bkgfitmode = inp.getBkgFittingMode();
    float kstart = inp.get_kstart();
    float kend = inp.get_kend();
    float Rbkg = inp.getRbkg();
    int kw = inp.get_kw();
    string output_dir=inp.getFittingOutputDir();
    
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
    string devicename = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>();
    size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    int imageSizeM = imagesizeX*imagesizeY;
    const cl::NDRange local_item_size(min(imagesizeX,(int)maxWorkGroupSize),1,1);
    const cl::NDRange global_item_size(imagesizeX,imagesizeY,1);
    const cl::NDRange global_item_size2(imagesizeX,imagesizeY,MAX_KRSIZE);
    
    cl::Buffer mt_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM*numEnergy, 0, NULL);
    cl::Buffer bkg_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
    cl::Buffer edgeJ_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
    cl::Buffer chi_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM*MAX_KRSIZE, 0, NULL);
    cl::Buffer mask_img(context, CL_MEM_READ_WRITE, sizeof(cl_char)*imageSizeM, 0, NULL);
    
    
    //transfer mt data to GPU
    for (int en=0; en<numEnergy; en++) {
        queue.enqueueWriteBuffer(mt_img, CL_TRUE, en*sizeof(cl_float)*imageSizeM, sizeof(cl_float)*imageSizeM, mt_vec[en]);
        queue.finish();
    }
    
    
    //set mask
    cl::Kernel kernel_mask(program[0],"setMask");
    kernel_mask.setArg(0, mt_img);
    kernel_mask.setArg(1, mask_img);
    queue.enqueueNDRangeKernel(kernel_mask, NULL, global_item_size, local_item_size, NULL, NULL);
    queue.finish();
    
    
    PreEdgeRemoval(queue, program[0], mt_img, energy, bkg_img, inp, imagesizeX, imagesizeY,
                   E0, preEdgeStartEnNo, preEdgeEndEnNo, bkgfitmode);
    
    PostEdgeEstimation(queue, program[1], mt_img, energy, bkg_img, edgeJ_img, inp, imagesizeX, imagesizeY, postEdgeStartEnNo, postEdgeEndEnNo);
    
    SplineBkgRemoval(queue, program[2], mt_img, energy, w_factor, chi_img, inp,
                     imagesizeX, imagesizeY, numEnergy, kstart, kend, Rbkg, kw);
    
    
    //apply mask
    cl::Kernel kernel_threshold(program[0],"applyMask");
    kernel_threshold.setArg(0, bkg_img);
    kernel_threshold.setArg(1, mask_img);
    queue.enqueueNDRangeKernel(kernel_threshold, NULL, global_item_size, local_item_size, NULL, NULL);
    queue.finish();
    kernel_threshold.setArg(0, edgeJ_img);
    queue.enqueueNDRangeKernel(kernel_threshold, NULL, global_item_size, local_item_size, NULL, NULL);
    queue.finish();
    kernel_threshold.setArg(0, chi_img);
    queue.enqueueNDRangeKernel(kernel_threshold, NULL, global_item_size2, local_item_size, NULL, NULL);
    queue.finish();
    
    
    for (int en = 0; en<numEnergy; en++) {
        delete[] mt_vec[en];
    }
    
    
    //transfer chi, bkg, edgeJ data to memory
    float* chiData;
    float* bkgData;
    float* edgeJData;
    chiData = new float[imageSizeM*MAX_KRSIZE];
    bkgData = new float[imageSizeM];
    edgeJData = new float[imageSizeM];
    queue.enqueueReadBuffer(chi_img, CL_TRUE, 0, sizeof(cl_float)*imageSizeM*MAX_KRSIZE, chiData);
    queue.finish();
    queue.enqueueReadBuffer(bkg_img, CL_TRUE, 0, sizeof(cl_float)*imageSizeM, bkgData);
    queue.finish();
    queue.enqueueReadBuffer(edgeJ_img, CL_TRUE, 0, sizeof(cl_float)*imageSizeM, edgeJData);
    queue.finish();
    
    
    //output thread
    output_th_fit[thread_id].join();
    output_th_fit[thread_id]=thread(extraction_output_thread,AngleNo,output_dir,
                                    move(chiData),move(bkgData), move(edgeJData),imageSizeM);
    
    return 0;
}



int extraction_data_input_thread(cl::CommandQueue queue, vector<cl::Program> program,
                                 int AngleNo, int thread_id,input_parameter inp,
                                 cl::Buffer energy, cl::Buffer w_factor){
    
    int startEnergyNo =inp.getFittingStartEnergyNo();
    int endEnergyNo = inp.getFittingEndEnergyNo();
    int imageSizeM = inp.getImageSizeM();
    
    string fileName_base = inp.getFittingFileBase();
    string input_dir=inp.getInputDir();
    vector<float*> mt_vec;
    vector<string> filepath_input;
    const int imgSizeM = inp.getImageSizeM();
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
        mt_vec.push_back(new float[imgSizeM]);
        filepath_input.push_back(input_dir + EnumTagString(i,"/",fileName_base) + AnumTagString(AngleNo,"",".raw"));
    }
    
    
    //input mt data
    //m1.lock();
    int num_energy = endEnergyNo -startEnergyNo +1;
    for (int i=0; i<num_energy; i++) {
        readRawFile(filepath_input[i],mt_vec[i],imageSizeM);
    }
    //m1.unlock();
    
    fitting_th[thread_id].join();
    fitting_th[thread_id] = thread(extraction_thread,
                                   queue, program, AngleNo, thread_id, inp,
                                   energy, w_factor, num_energy,move(mt_vec));
    
    return 0;
}

//dummy thread
static int dummy(){
    return 0;
}

int EXAFS_extraction_ocl(input_parameter inp, OCL_platform_device plat_dev_list){
    
    cl_int ret;
    float preEdgeStartE  = inp.getPreEdgeStartEnergy();
    float preEdgeEndE    = inp.getPreEdgeEndEnergy();
    float postEdgeStartE = inp.getPostEdgeStartEnergy();
    float postEdgeEndE   = inp.getPostEdgeEndEnergy();
    float E0=inp.getE0();
    float kstart = inp.get_kstart();
    float kend   = inp.get_kend();
    int startAngleNo=inp.getStartAngleNo();
    int endAngleNo=inp.getEndAngleNo();
    float splineStartE = kstart/EFF;
    float splineEndE = kend/EFF;
    float Rbkg = inp.getRbkg();
    int N_ctrlP = (int)floor(2.0*(kend-kstart)*Rbkg/PI)+1;
    
    
    //create output dir
    string fileName_output = inp.getFittingOutputDir() + "/bkg";
    MKDIR(fileName_output.c_str());
    fileName_output = inp.getFittingOutputDir() + "/edgeJ";
    MKDIR(fileName_output.c_str());
    fileName_output = inp.getFittingOutputDir() + "/chi0";
    MKDIR(fileName_output.c_str());
    
    
    //energy file input
    ifstream energy_ifs(inp.getEnergyFilePath(),ios::in);
    vector<float> energy;
    int i=0;
    do {
        string a;
        energy_ifs>>a;
        if (energy_ifs.eof()) break;
        float aa;
        try {
            aa = stof(a);
        }catch(invalid_argument ret){ //ヘッダータグが存在する場合に入力エラーになる際への対応
            continue;
        }
        cout<<i<<": "<<aa;
        energy.push_back(aa-E0);
        cout<<endl;
        i++;
    } while (!energy_ifs.eof());
    
    
    //energy No searching
    int preEdgeStartEnNo=0, preEdgeEndEnNo=0, postEdgeStartEnNo=0, postEdgeEndEnNo=0;
    int splineStartEnNo=0, splineEndEnNo=0;
    for (int en=0; en<energy.size(); en++) {
        preEdgeStartEnNo = (preEdgeStartE>=energy[i]) ? preEdgeStartEnNo:i;
        preEdgeEndEnNo = (preEdgeEndE>=energy[i]) ? preEdgeEndEnNo:i;
        postEdgeStartEnNo = (postEdgeStartE>=energy[i]) ? postEdgeStartEnNo:i;
        postEdgeEndEnNo = (postEdgeEndE>=energy[i]) ? postEdgeEndEnNo:i;
        splineStartEnNo = (splineStartE>=energy[i]) ? splineStartEnNo:i;
        splineEndEnNo = (splineEndE>=energy[i]) ? splineEndEnNo:i;
    }
    cout<< "pre-edge start energy No.: "<<preEdgeStartEnNo<<endl;
    cout<< "pre-edge end energy No.: "<<preEdgeEndEnNo<<endl;
    cout<< "post-edge start energy No.: "<<postEdgeStartEnNo<<endl;
    cout<< "post-edge end energy No.: "<<postEdgeEndEnNo<<endl;
    cout<< "spline start energy No.: "<<splineStartEnNo<<endl;
    cout<< "spline end energy No.: "<<splineEndEnNo<<endl;
    
    int startEnergyNo = min(min(preEdgeStartEnNo,postEdgeStartEnNo),splineStartEnNo);
    int endEnergyNo = max(max(preEdgeEndEnNo,postEdgeEndEnNo),splineEndEnNo);
    int num_energy=endEnergyNo-startEnergyNo+1;
    inp.setFittingEnergyNoRange(startEnergyNo, endEnergyNo);
    inp.setPreEdgeEnergyNoRange(preEdgeStartEnNo-startEnergyNo, preEdgeEndEnNo-startEnergyNo);
    inp.setPostEdgeEnergyNoRange(postEdgeStartEnNo-startEnergyNo, postEdgeEndEnNo-startEnergyNo);
    cout << "energy num for EXAFS extracting: "<<num_energy<<endl<<endl;
    
    //OpenCL Program
    vector<vector<cl::Program>> programs;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        vector<cl::Program> programs_atPlat;
        cl::Program::Sources source;
#if defined (OCL120)
        source.push_back(make_pair(kernel_fit_src.c_str(),kernel_fit_src.length()));
        source.push_back(make_pair(kernel_EXAFSfit_src.c_str(),kernel_EXAFSfit_src.length()));
        source.push_back(make_pair(kernel_3D_FFT_src.c_str(),kernel_3D_FFT_src.length()));
        source.push_back(make_pair(kernel_LM_src.c_str(),kernel_LM_src.length()));
        source.push_back(make_pair(kernel_ext_src.c_str(),kernel_ext_src.length()));
#else
        source.push_back(kernel_fit_src);
        source.push_back(kernel_EXAFSfit_src);
        source.push_back(kernel_3D_FFT_src);
        source.push_back(kernel_LM_src);
        source.push_back(kernel_ext_src);
#endif
        programs_atPlat.push_back(cl::Program(plat_dev_list.context(i), source,&ret));//pre-edge用
        programs_atPlat.push_back(cl::Program(plat_dev_list.context(i), source,&ret));//postedge用
        programs_atPlat.push_back(cl::Program(plat_dev_list.context(i), source,&ret));//spline用
        //kernel build
        string option ="";
#ifdef DEBUG
        option += "-D DEBUG -Werror";
#endif
        string GPUvendor =  plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0].getInfo<CL_DEVICE_VENDOR>();
        if(GPUvendor == "nvidia"){
            option += " -cl-nv-maxrregcount=64 -cl-nv-verbose";
        }
        else if (GPUvendor.find("NVIDIA Corporation")==0) {
            option += " -cl-nv-maxrregcount=64";
        }
        ostringstream oss0;
        oss0 << "-D FFT_SIZE=" << FFT_SIZE << " ";
        oss0 << "-D PARA_NUM=" << 3 << " ";
        oss0 << "-D PARA_NUM_SQ=" << 9 << " ";
        string option0 = oss0.str()+option;
        ostringstream oss1;
        oss1 << "-D FFT_SIZE=" << FFT_SIZE << " ";
        oss1 << "-D PARA_NUM=" << 4 << " ";
        oss1 << "-D PARA_NUM_SQ=" << 16 << " ";
        string option1 = oss1.str()+option;
        ostringstream oss2;
        oss2 << "-D FFT_SIZE=" << FFT_SIZE << " ";
        oss2 << "-D PARA_NUM=" << N_ctrlP << " ";
        oss2 << "-D PARA_NUM_SQ=" << N_ctrlP*N_ctrlP << " ";
        string option2 = oss2.str()+option;
        ret = programs_atPlat[0].build(option0.c_str());
        ret = programs_atPlat[1].build(option1.c_str());
        ret = programs_atPlat[2].build(option2.c_str());
#ifdef DEBUG
        string logstr=programs_atPlat[0].getBuildInfo<CL_PROGRAM_BUILD_LOG>(plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0]);
        cout << logstr << endl;
#endif
        programs.push_back(programs_atPlat);
    }
    
    
    //display OCL device
    vector<int> dA;
    int t=0;
    for (int i=0; i<plat_dev_list.platsize(); i++) {
        for (int j=0; j<plat_dev_list.devsize(i); j++) {
            cout<<"Device No. "<<t+1<<endl;
            string platform_param;
            plat_dev_list.plat(i).getInfo(CL_PLATFORM_NAME, &platform_param);
            cout << "CL PLATFORM NAME: "<< platform_param<<endl;
            plat_dev_list.plat(i).getInfo(CL_PLATFORM_VERSION, &platform_param);
            cout << "   "<<platform_param<<endl;
            string device_pram;
            plat_dev_list.dev(i,j).getInfo(CL_DEVICE_NAME, &device_pram);
            cout << "CL DEVICE NAME: " << device_pram << endl << endl;
            t++;
        }
    }
    
    
    //energy_buffers create (per context)
    vector<cl::Buffer> energy_buffers;
    vector<cl::Buffer> w_factors;
    for (int i=0; i<plat_dev_list.contextsize(); i++){
        //energy buffers
        energy_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*num_energy, 0, NULL));
        plat_dev_list.queue(i,0).enqueueWriteBuffer(energy_buffers[i], CL_TRUE, 0, sizeof(cl_float)*num_energy, &energy[startEnergyNo], NULL, NULL);
        
        //spinfactors
        w_factors.push_back(cl::Buffer(plat_dev_list.context(i),CL_MEM_READ_WRITE,sizeof(cl_float2)*FFT_SIZE/2, 0, NULL));
        createSpinFactor(w_factors[i], plat_dev_list.queue(i,0), programs[i][2]);
    }
    
    
    //start thread
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        input_th.push_back(thread(dummy));
        fitting_th.push_back(thread(dummy));
        output_th_fit.push_back(thread(dummy));
    }
    for (int An=startAngleNo; An<=endAngleNo;) {
        for (int i=0; i<plat_dev_list.contextsize(); i++) {
            if (input_th[i].joinable()) {
                input_th[i].join();
                input_th[i] = thread(extraction_data_input_thread,
                                     plat_dev_list.queue(i,0), programs[i],
                                     An, i, inp, energy_buffers[i], w_factors[i]);
                An++;
                if (An > endAngleNo) break;
            } else{
                this_thread::sleep_for(chrono::milliseconds(100));
            }
        }
        if (i > endAngleNo) break;
    }
    
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        input_th[j].join();
    }
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        fitting_th[j].join();
    }
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        output_th_fit[j].join();
    }
   
    return 0;
}
