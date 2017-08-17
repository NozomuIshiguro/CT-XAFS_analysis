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

vector<int> extractionGPUmemoryControl(int imageSizeX, int imageSizeY,int ksize,int Rsize,
                                       int num_energy, int preEdgeFittingMode, cl::CommandQueue queue){
    
    vector<int> processImageSizeY;
    processImageSizeY.push_back(imageSizeY); //for other process
    processImageSizeY.push_back(imageSizeY); //for FFT/IFFT size
    
    size_t GPUmemorySize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    
    size_t usingMemorySize;
    int preedge_paraN=1;
    switch (preEdgeFittingMode) {
        case 0: //line
            preedge_paraN=2;
            break;
        case 1: //Victoreen
            preedge_paraN=4;
            break;
        case 2: //McMaster
            preedge_paraN=3;
            break;
        default:
            break;
    }
    int postedge_paraN=4;
    int N_knots = (int)floor(2.0*ksize*Rsize/FFT_SIZE)+1;
    int paraN = (int)max(preedge_paraN,max(postedge_paraN,N_knots));
    do {
        usingMemorySize =0;
        //energy
        usingMemorySize += num_energy*sizeof(float);
        //spin factor
        usingMemorySize += FFT_SIZE/2*sizeof(cl_float2);
        //mt_img
        usingMemorySize += imageSizeX*processImageSizeY[0]*num_energy*sizeof(float);
        //bkg_energy
        usingMemorySize += imageSizeX*processImageSizeY[0]*sizeof(float);
        //edgeJ_img
        usingMemorySize += imageSizeX*processImageSizeY[0]*sizeof(float);
        //chi_img
        usingMemorySize += imageSizeX*processImageSizeY[0]*MAX_KRSIZE*sizeof(float);
        
        //tJJ
        usingMemorySize += imageSizeX*processImageSizeY[0]*paraN*(paraN+1)/2*sizeof(float);
        //tJdF, dp, fp, fp_cnd
        usingMemorySize += 4*imageSizeX*processImageSizeY[0]*paraN*sizeof(float);
        //dF2_old, dF2_new, lambda, nyu, dL, rho
        usingMemorySize += 6*imageSizeX*processImageSizeY[0]*sizeof(float);
        
        //chidata, chifit, Jacdummy_k
        usingMemorySize += 3*imageSizeX*processImageSizeY[0]*ksize*sizeof(cl_float2);
        //FTchifit, FTchistd
        usingMemorySize += 2*imageSizeX*processImageSizeY[0]*Rsize*sizeof(cl_float2);
        //Jacobian
        usingMemorySize += imageSizeX*processImageSizeY[0]*Rsize*paraN*sizeof(cl_float2);
        
        
        //FFT/IFFT objects
        size_t usingMemorySizeFFT;
        do {
            usingMemorySizeFFT=usingMemorySize;
            //FTchi dummy
            usingMemorySizeFFT += imageSizeX*processImageSizeY[1]*FFT_SIZE*sizeof(cl_float2);
                
            if(GPUmemorySize<usingMemorySizeFFT){
                processImageSizeY[1] /=2;
            }else{
                break;
            }
        } while (processImageSizeY[1]>processImageSizeY[0]/8);
        usingMemorySize = usingMemorySizeFFT;
        
        if(GPUmemorySize<usingMemorySize){
            processImageSizeY[0] /=2;
            processImageSizeY[1] = processImageSizeY[0];
        }else{
            break;
        }
    } while (processImageSizeY[0]>1);
    
#ifdef DEBUG
    cout << "image size X: "<<imageSizeX<<endl;
    cout << "processing image size Y: " << processImageSizeY[0] <<endl;
    cout << "FFT processing image size Y: " << processImageSizeY[1] <<endl;
    cout << "GPU mem size "<<GPUmemorySize <<" bytes >= using mem size " << usingMemorySize<<" bytes"<<endl<<endl;
#endif
    
    return processImageSizeY;
}

int extraction_output_thread(int AngleNo,string output_dir,
                            float* chiData, float* bkgData, float* edgeJumpData, int imageSizeM, int numk){
    
    ostringstream oss;
    string fileName_output= output_dir+ AnumTagString(AngleNo,"/chi0/i", ".raw");
    oss << "output file: " << fileName_output << endl;
    outputRawFile_stream(fileName_output,chiData,imageSizeM*numk);
    fileName_output= output_dir+ AnumTagString(AngleNo,"/bkg/i", ".raw");
    oss << "output file: " << fileName_output << endl;
    outputRawFile_stream(fileName_output,bkgData,imageSizeM);
    fileName_output= output_dir+ AnumTagString(AngleNo,"/edgeJ/i", ".raw");
    oss << "output file: " << fileName_output << endl;
    outputRawFile_stream(fileName_output,edgeJumpData,imageSizeM);
    cout << oss.str()<<endl;

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
        cl::Buffer nyu_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM, 0, NULL);
        cl::Buffer lambda_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM, 0, NULL);
        cl::Buffer chi2_old_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM, 0, NULL);
        cl::Buffer chi2_new_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM, 0, NULL);
        cl::Buffer fp_cnd_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM*paramsize, 0, NULL);
        cl::Buffer dp_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM*paramsize, 0, NULL);
        cl::Buffer dL_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM, 0, NULL);
        cl::Buffer rho_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imgSizeM, 0, NULL);
        cl::Buffer freeFix_buff(context, CL_MEM_READ_WRITE, sizeof(cl_char)*paramsize, 0, NULL);
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
        //estimate dL
        kernel_dL.setArg(0, dp_img);
        kernel_dL.setArg(1, tJJ_buff);
        kernel_dL.setArg(2, tJdF_buff);
        kernel_dL.setArg(3, lambda_buff);
        kernel_dL.setArg(4, dL_buff);
        //estimate fp cnd
        kernel_cnd.setArg(0, dp_img);
        kernel_cnd.setArg(1, fp_cnd_img);
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
            /*float* tJJ_data;
            tJJ_data = new float[imgSizeM*paramsize*(paramsize+1)/2];
            queue.enqueueReadBuffer(tJJ_buff, CL_TRUE, 0, sizeof(float)*imgSizeM*paramsize*(paramsize+1)/2, tJJ_data);
            cout<<"tJJ"<<endl;
            for (int i=0; i<paramsize*(paramsize+1)/2; i++) {
                cout <<"tJJ["<<i<<"]: "<<tJJ_data[i*imgSizeM]<<endl;
            }
            cout<<endl;
            cout<<"tJdF"<<endl;
            float* tJdF_data;
            tJdF_data = new float[imgSizeM*paramsize];
            queue.enqueueReadBuffer(tJdF_buff, CL_TRUE, 0, sizeof(float)*imgSizeM*paramsize, tJdF_data);
            for (int i=0; i<paramsize; i++) {
                cout <<"tJdF["<<i<<"]: "<<tJdF_data[i*imgSizeM]<<endl;
            }
            cout<<endl;*/
            
            errorzone = "L-M";
            //L-M equation
            queue.enqueueNDRangeKernel(kernel_LM, NULL, global_item_size, local_item_size, NULL, NULL);
            queue.finish();
            /*cout<<"dp"<<endl;
            float* dp_data;
            dp_data = new float[imgSizeM*paramsize];
            queue.enqueueReadBuffer(dp_img, CL_TRUE, 0, sizeof(float)*imgSizeM*paramsize, dp_data);
            for (int i=0; i<paramsize; i++) {
                cout <<"dp["<<i<<"]: "<<dp_data[i*imgSizeM]<<endl;
            }
            cout<<endl;*/
            
            
            errorzone = "estimating dL";
            //estimate dL
            queue.enqueueNDRangeKernel(kernel_dL, NULL, global_item_size, local_item_size, NULL, NULL);
            queue.finish();
            
            
            errorzone = "estimating fp_cnd";
            //estimate fp candidate
            queue.enqueueCopyBuffer(fp_img, fp_cnd_img, 0, 0, sizeof(cl_float)*imgSizeM*paramsize);
            queue.enqueueNDRangeKernel(kernel_cnd, NULL, global_item_size2, local_item_size, NULL, NULL);
            queue.finish();
            /*cout<<"fp_cnd"<<endl;
            float* fp_cnd_data;
            fp_cnd_data = new float[imgSizeM*paramsize];
            queue.enqueueReadBuffer(fp_cnd_img, CL_TRUE, 0, sizeof(float)*imgSizeM*paramsize, fp_cnd_data);
            for (int i=0; i<paramsize; i++) {
                cout <<"fp_cnd["<<i<<"]: "<<fp_cnd_data[i*imgSizeM]<<endl;
            }
            cout<<endl;*/
            
            
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
            /*float* rho_data;
            rho_data = new float[imgSizeM];
            queue.enqueueReadBuffer(rho_buff, CL_TRUE, 0, sizeof(float)*imgSizeM, rho_data);
            cout <<"rho: "<<rho_data[0]<<endl;
            cout<<endl;
            cout<<"fp"<<endl;
            float* fp_data;
            fp_data = new float[imgSizeM*paramsize];
            queue.enqueueReadBuffer(fp_img, CL_TRUE, 0, sizeof(float)*imgSizeM*paramsize, fp_data);
            for (int i=0; i<paramsize; i++) {
                cout <<"fp["<<i<<"]: "<<fp_data[i*imgSizeM]<<endl;
            }
            cout<<endl;*/
            
        }
    }catch (const cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")"<< errorzone << endl;
    }
    
    return 0;
}



int PreEdgeRemoval(cl::CommandQueue queue, cl::Program program,
                   cl::Buffer mt_img, cl::Buffer energy, cl::Buffer bkg_img,
                   int imagesizeX, int imagesizeY, float E0, int startEn, int endEn,
                   int fitmode, float lambda, int numTrial){
    
    
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
        freefixP.push_back(49);
        freefixP.push_back(49);
        iniP.push_back(0.0f);
        iniP.push_back(-0.001f);
        break;
        
        case 1: //Victreen
        funcList.push_back("Victoreen");
        freefixP.push_back(49);
        freefixP.push_back(49);
        freefixP.push_back(49);
        freefixP.push_back(48);
        iniP.push_back(0.0f);
        iniP.push_back(0.001f);
        iniP.push_back(0.0001f);
        iniP.push_back(E0);
        break;
        
        case 2: //McMaster
        funcList.push_back("McMaster");
        freefixP.push_back(49);
        freefixP.push_back(49);
        freefixP.push_back(48);
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
    cl::Image1DArray refSpectra(context,CL_MEM_READ_WRITE,format,1,IMAGE_SIZE_E, 0, NULL);
    
    //fitting
    fitting(queue, program, energy, mt_img, fp_img, refSpectra, fiteq, lambda, numTrial, startEn, endEn, imagesizeX, imagesizeY);
    
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
                       int imagesizeX, int imagesizeY, int startEn, int endEn,
                       float lambda, int numTrial){
    
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
    freefixP.push_back(49);
    freefixP.push_back(49);
    freefixP.push_back(49);
    freefixP.push_back(49);
    iniP.push_back(0.0f);
    iniP.push_back(0.0f);
    iniP.push_back(0.0f);
    iniP.push_back(0.0f);
    fiteq.setInitialParameter(iniP);
    fiteq.setFreeFixParameter(freefixP);
    
    cl::Buffer fp_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM*fiteq.ParaSize(), 0, NULL);
    cl::ImageFormat format(CL_R, CL_FLOAT);
    cl::Image1DArray refSpectra(context,CL_MEM_READ_WRITE,format,1,IMAGE_SIZE_E, 0, NULL);
    
    //fitting
    fitting(queue, program, energy, mt_img, fp_img, refSpectra, fiteq, lambda, numTrial, startEn, endEn, imagesizeX, imagesizeY);
    
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
                     int imagesizeX, int imagesizeY, int FFTimageSizeY, int num_energy,
                     float kstart, float kend, float Rbkg, int kw, float lambda, int numTrial, bool kendClamp){
    
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
    string devicename = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>();
    vector<size_t> maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    int imageSizeM = imagesizeX*imagesizeY;
    
    int ksize=(int)ceil((min(float(kend + WIN_DK),(float)MAX_KQ)-max(float(kstart-WIN_DK),0.0f))/KGRID)+1;
    int koffset=(int)floor(max((float)(kstart - WIN_DK), 0.0f) / KGRID);
    int Rsize=(int)ceil((min(float(Rbkg + WIN_DR),(float)MAX_R))/RGRID)+1;
    int Roffset=0;
    int N_ctrlP = (int)floor(2.0*(kend-kstart)*Rbkg/PI)+1;
    float knotStart = max(float(kstart-WIN_DK),0.0f);
    float knotPitch = (min(float(kend + WIN_DK),(float)MAX_KQ)-max(float(kstart-WIN_DK),0.0f))/(N_ctrlP-1);
    //cout <<"N_ctrlP: "<<N_ctrlP<<endl;
    //cout <<"knot pitch: "<<knotPitch<<endl;
    char* p_freefix;
    p_freefix = new char[N_ctrlP];
    for (int i=0; i<N_ctrlP-1; i++) {
        p_freefix[i]=49;
    }
    p_freefix[N_ctrlP-1]=(kendClamp) ? 48:49;
    
    //CL objects
    cl_float2 iniChi={0.0f,0.0f};
    cl::Buffer chiData(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*ksize,0,NULL);
    cl::Buffer FTchiStd(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*Rsize,0,NULL);
    cl::Buffer chiFit(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*ksize,0,NULL);
    cl::Buffer FTchiFit(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*Rsize,0,NULL);
    cl::Buffer dF2_old(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
    cl::Buffer dF2_new(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
    cl::Buffer tJJ(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*N_ctrlP*(N_ctrlP+1)/2,0,NULL);
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
    cl::Buffer basis(context,CL_MEM_READ_WRITE,sizeof(cl_float)*(N_ctrlP+4)*ksize,0,NULL);
    cl::Buffer basis_dummy(context,CL_MEM_READ_WRITE,sizeof(cl_float)*(N_ctrlP+4)*ksize,0,NULL);
    queue.enqueueWriteBuffer(p_fix, CL_TRUE, 0, sizeof(cl_char)*N_ctrlP, p_freefix);
    queue.enqueueFillBuffer(lambda_buff, (cl_float)lambda, 0, sizeof(float)*imageSizeM);
    queue.enqueueFillBuffer(nyu_buff, (cl_float)2.0f, 0, sizeof(float)*imageSizeM);
    queue.enqueueFillBuffer(FTchiStd, (cl_float2)iniChi, 0, sizeof(cl_float2)*imageSizeM*Rsize);
    
    
    //work itemsize settings
    const cl::NDRange local_item_size1(min((int)maxWorkGroupSize[0],imagesizeX),1,1);
    const cl::NDRange local_item_size2(1,min((int)maxWorkGroupSize[1],ksize),1);
    const cl::NDRange global_item_size(imagesizeX,imagesizeY,1);
    const cl::NDRange global_item_size_stack(imagesizeX,imagesizeY,ksize);
    const cl::NDRange global_item_size_para(imagesizeX,imagesizeY,N_ctrlP);
    const cl::NDRange global_item_offset1(0,0,koffset);
    const cl::NDRange global_item_offset2(0,koffset,0);
    const cl::NDRange global_item_size_basis(N_ctrlP+4,ksize,1);
    const cl::NDRange global_item_size_ctrlP(imagesizeX,imagesizeY,N_ctrlP);
    
    
    //redimension of mt_img stack to k-space chi stack
    queue.enqueueFillBuffer(chi_img, (cl_float)0.0f, 0, sizeof(cl_float)*MAX_KRSIZE);
    cl::Kernel kernel_redim(program,"redimension_mt2chi");
    kernel_redim.setArg(0, mt_img);
    kernel_redim.setArg(1, chi_img);
    kernel_redim.setArg(2, energy);
    kernel_redim.setArg(3, (cl_int)num_energy);
    kernel_redim.setArg(4, (cl_int)0);
    kernel_redim.setArg(5, (cl_int)MAX_KRSIZE);
    queue.enqueueNDRangeKernel(kernel_redim, NULL, global_item_size, local_item_size1, NULL, NULL);
    queue.finish();
    /*cl_float* chi_data;
    chi_data = new cl_float[imageSizeM*MAX_KRSIZE];
    queue.enqueueReadBuffer(chi_img, CL_TRUE, 0, sizeof(cl_float)*imageSizeM*MAX_KRSIZE, chi_data);
    cout<<"chidata"<<endl;
    for (int i=0; i<MAX_KRSIZE; i++) {
        cout <<i*0.05f<<"\t"<<chi_data[i*imageSizeM]<<endl;
    }
    cout<<endl;*/
    
    
    //convert to complex chi
    cl::Kernel kernel_cmplx(program,"chi2cmplxChi_imgStck");
    kernel_cmplx.setArg(0, chi_img);
    kernel_cmplx.setArg(1, chiData);
    kernel_cmplx.setArg(3, (cl_int)koffset);
    kernel_cmplx.setArg(4, (cl_int)0);
    for (int kn=0; kn<ksize; kn++) {
        kernel_cmplx.setArg(2, (cl_int)kn);
        const cl::NDRange global_item_offset3(0,0,koffset+kn);
        queue.enqueueNDRangeKernel(kernel_cmplx, global_item_offset3, global_item_size, local_item_size1, NULL, NULL);
        queue.finish();
    }
    /*cl_float2* chi_data;
    chi_data = new cl_float2[imageSizeM*ksize];
    queue.enqueueReadBuffer(chiData, CL_TRUE, 0, sizeof(cl_float2)*imageSizeM*ksize, chi_data);
    cout<<"chidata"<<endl;
    for (int i=0; i<ksize; i++) {
        cout <<i*0.05f<<"\t"<<chi_data[i*imageSizeM].y<<endl;
    }
    cout<<endl;*/
    
    
    //initialize ctrlP
    cl::Kernel kernel_iniCtrlP(program,"initialCtrlP");
    kernel_iniCtrlP.setArg(0, chiData);
    kernel_iniCtrlP.setArg(1, fp_img);
    kernel_iniCtrlP.setArg(2, (cl_float)knotPitch);
    kernel_iniCtrlP.setArg(3, (cl_int)ksize);
    queue.enqueueNDRangeKernel(kernel_iniCtrlP, NULL, global_item_size_ctrlP, local_item_size1, NULL, NULL);
    queue.finish();
    /*float* ctrlP_data;
    ctrlP_data = new float[N_ctrlP];
    queue.enqueueReadBuffer(fp_img, CL_TRUE, 0, sizeof(cl_float)*N_ctrlP, ctrlP_data);
    cout<<"ctrlP"<<endl;
    for (int i=0; i<N_ctrlP; i++) {
        cout <<i*knotPitch<<"\t"<<ctrlP_data[i]<<endl;
    }
    cout<<endl;*/
    
    //create B-spline basis0
    cl::Kernel kernel_basis0(program,"Bspline_basis_zero");
    kernel_basis0.setArg(0, basis);
    kernel_basis0.setArg(1, (cl_int)N_ctrlP);
    kernel_basis0.setArg(2, (cl_float)knotStart);
    kernel_basis0.setArg(3, (cl_float)knotPitch);
    kernel_basis0.setArg(4, (cl_int)4);
    queue.enqueueNDRangeKernel(kernel_basis0, global_item_offset2, global_item_size_basis, local_item_size2, NULL, NULL);
    queue.finish();
    /*float* basisData;
    basisData = new float[(N_ctrlP+4)*ksize];
    queue.enqueueReadBuffer(basis, CL_TRUE, 0, sizeof(cl_float)*(N_ctrlP+4)*ksize, basisData);
    cout<<"basis"<<endl;
    for (int i=0; i<ksize; i++) {
        cout <<(koffset+i)*0.05f<<"\t"<<basisData[13+i*(N_ctrlP+4)]<<endl;
    }
    cout<<endl;*/
    
    
    //update order of basis
    cl::Kernel kernel_updatebasis(program,"Bspline_basis_updateOrder");
    kernel_updatebasis.setArg(0, basis);
    kernel_updatebasis.setArg(1, basis_dummy);
    kernel_updatebasis.setArg(2, (cl_int)N_ctrlP);
    kernel_updatebasis.setArg(3, (cl_float)knotStart);
    kernel_updatebasis.setArg(4, (cl_float)knotPitch);
    kernel_updatebasis.setArg(5, (cl_int)4);
    for (int i=1; i<=4; i++) {
        kernel_updatebasis.setArg(6, (cl_int)i);
        queue.enqueueNDRangeKernel(kernel_updatebasis, global_item_offset2, global_item_size_basis, local_item_size2, NULL, NULL);
        queue.finish();
        queue.enqueueCopyBuffer(basis_dummy, basis, 0, 0, sizeof(cl_float)*(N_ctrlP+4)*ksize);
        queue.finish();
    }
    /*float* basisData;
    basisData = new float[(N_ctrlP+4)*ksize];
    queue.enqueueReadBuffer(basis, CL_TRUE, 0, sizeof(cl_float)*(N_ctrlP+4)*ksize, basisData);
    cout<<"basis"<<endl;
    for (int i=0; i<ksize; i++) {
        cout <<(koffset+i)*0.05f<<"\t"<<basisData[i*(N_ctrlP+4)]<<endl;
    }
    cout<<endl;*/
    
    
    
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
    cl::Kernel kernel_dL(program,"estimate_dL");
    kernel_dL.setArg(0, dp_img);
    kernel_dL.setArg(1, tJJ);
    kernel_dL.setArg(2, tJdF);
    kernel_dL.setArg(3, lambda_buff);
    kernel_dL.setArg(4, dL_buff);
    cl::Kernel kernel_update(program,"updatePara");
    kernel_update.setArg(0, dp_img);
    kernel_update.setArg(1, fp_cnd_img);
    kernel_update.setArg(2, (cl_int)0);
    kernel_update.setArg(3, (cl_int)0);
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
    cl::Kernel kernel_spline(program,"BsplineRemoval");
    kernel_spline.setArg(0, chiData);
    kernel_spline.setArg(1, chiFit);
    kernel_spline.setArg(2, basis);
    kernel_spline.setArg(4, (cl_int)N_ctrlP);
    kernel_spline.setArg(5, (cl_int)4);
    kernel_spline.setArg(6, (cl_int)4);
    cl::Kernel kernel_kw(program,"kweight");
    kernel_kw.setArg(0, J_dummy);
    kernel_kw.setArg(1, (cl_int)kw);
    
    //estimate Jacobian (put outside of loop because J are independent from fp)
    cl::Kernel kernel_jacob(program,"Jacobian_BsplineRemoval");
    kernel_jacob.setArg(0, J_dummy);
    kernel_jacob.setArg(2, basis);
    kernel_jacob.setArg(3, (cl_int)N_ctrlP);
    kernel_jacob.setArg(4, (cl_int)4);
    kernel_jacob.setArg(5, (cl_int)4);
    //Jacobian
    //reset Jacobian
    for(int i=0;i<N_ctrlP;i++){
        queue.enqueueFillBuffer(Jacobian[i], iniChi, 0, sizeof(cl_float2)*imageSizeM*Rsize);
        queue.enqueueFillBuffer(J_dummy, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
        kernel_jacob.setArg(1, (cl_int)i);
        queue.enqueueNDRangeKernel(kernel_jacob,NULL,global_item_size_stack, local_item_size1, NULL, NULL);
        queue.finish();
        //kweight
        queue.enqueueNDRangeKernel(kernel_kw,global_item_offset1,global_item_size_stack, local_item_size1, NULL, NULL);
        queue.finish();
        //weighting by window function
        kernel_kwindow.setArg(0, J_dummy);
        queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset1,global_item_size_stack, local_item_size1, NULL, NULL);
        queue.finish();
        
        //FFT
        for (int offsetY=0; offsetY<imagesizeY; offsetY += FFTimageSizeY) {
            FFT(J_dummy, Jacobian[i], w_factor, queue, program,
                imagesizeX, imagesizeY, FFTimageSizeY, offsetY,
                koffset,ksize,Roffset,Rsize);
        }
    }
    
    
    //estimate tJJ (put outside of loop because J are independent from fp)
    int pn=0;
    queue.enqueueFillBuffer(tJJ,(cl_float)0.0f,0,sizeof(cl_float)*imageSizeM*N_ctrlP*(N_ctrlP+1)/2);
    for(int fpn1=0; fpn1<N_ctrlP;fpn1++){
        kernel_tJJ.setArg(1, Jacobian[fpn1]);
        for(int fpn2=fpn1;fpn2<N_ctrlP;fpn2++){
            kernel_tJJ.setArg(2, Jacobian[fpn2]);
            kernel_tJJ.setArg(3, (cl_int)pn);
            if(p_freefix[fpn1]==49&&p_freefix[fpn2]==49){
                queue.enqueueNDRangeKernel(kernel_tJJ,NULL,global_item_size,local_item_size1,NULL,NULL);
                queue.finish();
            }
            pn++;
        }
    }
    /*float* tJJ_data;
    tJJ_data = new float[imageSizeM*N_ctrlP*(N_ctrlP+1)/2];
    queue.enqueueReadBuffer(tJJ, CL_TRUE, 0, sizeof(float)*imageSizeM*N_ctrlP*(N_ctrlP+1)/2, tJJ_data);
    cout<<"tJJ"<<endl;
    for (int i=0; i<N_ctrlP*(N_ctrlP+1)/2; i++) {
        cout <<"tJJ["<<i<<"]: "<<tJJ_data[i*imageSizeM]<<endl;
    }
    cout<<endl;*/
    
    kernel_kw.setArg(0, chiFit);
    kernel_kwindow.setArg(0, chiFit);
    for (int trial=0; trial<numTrial; trial++) {
        //reset dF2, tJdF
        queue.enqueueFillBuffer(dF2_old, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM);
        queue.enqueueFillBuffer(dF2_new, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM);
        queue.enqueueFillBuffer(tJdF, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM*N_ctrlP);
        /*float* fp_data;
        fp_data = new float[imageSizeM*N_ctrlP];
        queue.enqueueReadBuffer(fp_img, CL_TRUE, 0, sizeof(float)*imageSizeM*N_ctrlP, fp_data);
        cout<<"fp"<<endl;
        for (int i=0; i<N_ctrlP; i++) {
            cout <<"fp["<<i<<"]: "<<fp_data[i*imageSizeM]<<endl;
        }
        cout<<endl;*/
        
        
        //chiFit
        //reset chiFit
        queue.enqueueFillBuffer(chiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
        queue.enqueueFillBuffer(FTchiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*Rsize);
        //estimate chiFit
        kernel_spline.setArg(3, fp_img);
        queue.enqueueNDRangeKernel(kernel_spline,NULL,global_item_size_stack,local_item_size1, NULL, NULL);
        queue.finish();
        //kweight
        queue.enqueueNDRangeKernel(kernel_kw,global_item_offset1,global_item_size_stack, local_item_size1, NULL, NULL);
        queue.finish();
        //weighting by window function
        queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset1,global_item_size_stack,local_item_size1, NULL, NULL);
        queue.finish();
        //FFT
        for (int offsetY=0; offsetY<imagesizeY; offsetY += FFTimageSizeY) {
            FFT(chiFit, FTchiFit, w_factor, queue, program,
                imagesizeX, imagesizeY, FFTimageSizeY, offsetY,
                koffset,ksize,Roffset,Rsize);
        }

        
        //estimate tJJ and tJdF
        for(int fpn1=0; fpn1<N_ctrlP;fpn1++){
            kernel_tJdF.setArg(1, Jacobian[fpn1]);
            kernel_tJdF.setArg(4, (cl_int)fpn1);
            if (p_freefix[fpn1]==49) {
                queue.enqueueNDRangeKernel(kernel_tJdF,NULL,global_item_size,local_item_size1, NULL, NULL);
                queue.finish();
            }
        }
        /*cout<<"tJdF"<<endl;
        float* tJdF_data;
        tJdF_data = new float[imageSizeM*N_ctrlP];
        queue.enqueueReadBuffer(tJdF, CL_TRUE, 0, sizeof(float)*imageSizeM*N_ctrlP, tJdF_data);
        for (int i=0; i<N_ctrlP; i++) {
            cout <<"tJdF["<<i<<"]: "<<tJdF_data[i*imageSizeM]<<endl;
        }
        cout<<endl;*/
        
        
        //estimate dF2
        kernel_dF2.setArg(0, dF2_old);
        queue.enqueueNDRangeKernel(kernel_dF2,NULL,global_item_size, local_item_size1, NULL, NULL);
        queue.finish();
        
        
        //Levenberg-Marquardt
        queue.enqueueNDRangeKernel(kernel_LM,NULL,global_item_size,local_item_size1, NULL, NULL);
        queue.finish();
        /*float* dp_data;
        dp_data = new float[imageSizeM*N_ctrlP];
        queue.enqueueReadBuffer(dp_img, CL_TRUE, 0, sizeof(float)*imageSizeM*N_ctrlP, dp_data);
        cout<<"dp"<<endl;
        for (int i=0; i<N_ctrlP; i++) {
            cout <<"dp["<<i<<"]: "<<dp_data[i*imageSizeM]<<endl;
        }*/
        
        
        //estimate dL
        queue.enqueueNDRangeKernel(kernel_dL,NULL,global_item_size,local_item_size1, NULL, NULL);
        queue.finish();
        
        
        //update paramater as candidate
        queue.enqueueCopyBuffer(fp_img, fp_cnd_img, 0, 0, sizeof(cl_float)*imageSizeM*N_ctrlP);
        queue.enqueueNDRangeKernel(kernel_update, NULL, global_item_size_para, local_item_size1, NULL, NULL);
        queue.finish();
        
        
        //estimate new dF2
        //reset chiFit
        queue.enqueueFillBuffer(chiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
        queue.enqueueFillBuffer(FTchiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*Rsize);
        //estimate chiFit
        kernel_spline.setArg(3, fp_cnd_img);
        queue.enqueueNDRangeKernel(kernel_spline,NULL,global_item_size_stack,local_item_size1, NULL, NULL);
        queue.finish();
        //kweight
        queue.enqueueNDRangeKernel(kernel_kw,global_item_offset1,global_item_size_stack, local_item_size1, NULL, NULL);
        queue.finish();
        //weighting by window function
        queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset1,global_item_size_stack,local_item_size1, NULL, NULL);
        queue.finish();
        //FFT
        for (int offsetY=0; offsetY<imagesizeY; offsetY += FFTimageSizeY) {
            FFT(chiFit, FTchiFit, w_factor, queue, program,
                imagesizeX, imagesizeY, FFTimageSizeY, offsetY,
                koffset,ksize,Roffset,Rsize);
        }
        //estimate dF2
        kernel_dF2.setArg(0, dF2_new);
        queue.enqueueNDRangeKernel(kernel_dF2,NULL,global_item_size_stack, local_item_size1, NULL, NULL);
        
        
        //evaluate updated parameter (update to new para or hold at old para )
        //evaluate rho
        queue.enqueueNDRangeKernel(kernel_eval,NULL,global_item_size, local_item_size1,NULL,NULL);
        queue.finish();
        //update or hold parameter
        queue.enqueueNDRangeKernel(kernel_UorH,NULL,global_item_size_para,local_item_size1,NULL, NULL);
        queue.finish();
        /*float* fp_data;
        fp_data = new float[imageSizeM*N_ctrlP];
        queue.enqueueReadBuffer(fp_img, CL_TRUE, 0, sizeof(float)*imageSizeM*N_ctrlP, fp_data);
        cout<<"fp"<<endl;
        for (int i=0; i<N_ctrlP; i++) {
            cout <<"fp["<<i<<"]: "<<fp_data[i*imageSizeM]<<endl;
        }
        cout<<endl;*/
    }
    
    
    //estimate final chiFit(spline removed chiData)
    //reset chiFit
    //queue.enqueueFillBuffer(chiData, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
    queue.enqueueFillBuffer(chiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
    queue.enqueueFillBuffer(FTchiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*Rsize);
    //estimate chiFit
    kernel_spline.setArg(3, fp_img);
    queue.enqueueNDRangeKernel(kernel_spline,NULL,global_item_size_stack,local_item_size1, NULL, NULL);
    queue.finish();
    
    
    //convert final chiFit(spline removed chiData) to chi_img
    cl::Kernel kernel_c2real(program,"convert2realChi");
    kernel_c2real.setArg(0, chiFit);
    kernel_c2real.setArg(1, chi_img);
    queue.enqueueNDRangeKernel(kernel_c2real,global_item_offset1,global_item_size_stack,local_item_size1,NULL, NULL);
    queue.finish();

	//fill 0 before kstart and after kend
	queue.enqueueFillBuffer(chi_img, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM*koffset);
	queue.enqueueFillBuffer(chi_img, (cl_float)0.0f, sizeof(cl_float)*imageSizeM*(koffset+ksize), sizeof(cl_float)*imageSizeM*(MAX_KRSIZE-koffset-ksize));
	queue.finish();

    
    return 0;
}


int extraction_thread(cl::CommandQueue queue, vector<cl::Program> program,
                     int AngleNo, int thread_id,input_parameter inp,
                     cl::Buffer energy, cl::Buffer w_factor, int numEnergy,vector<float*> mt_vec,
                      vector<int> processImageSizeY){
    
    
    int preEdgeStartEnNo  = inp.getPreEdgeStartEnergyNo();
    int preEdgeEndEnNo    = inp.getPreEdgeEndEnergyNo();
    int postEdgeStartEnNo = inp.getPostEdgeStartEnergyNo();
    int postEdgeEndEnNo   = inp.getPostEdgeEndEnergyNo();
    int imagesizeX = inp.getImageSizeX();
    int imagesizeY = inp.getImageSizeY();
    float E0 = inp.getE0();
    int bkgfitmode = inp.getPreEdgeFittingMode();
    float kstart = inp.get_kstart();
    float kend = inp.get_kend();
	int ksize = (int)ceil((min(float(kend + WIN_DK), (float)MAX_KQ) - max(float(kstart - WIN_DK), 0.0f)) / KGRID) + 1;
	int koffset = (int)floor(max((float)(kstart - WIN_DK), 0.0f) / KGRID);
    float Rbkg = inp.getRbkg();
    int kw = inp.get_kw();
    string output_dir=inp.getFittingOutputDir();
    
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
    string devicename = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>();
    size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    int imageSizeM = imagesizeX*imagesizeY;
    int processImageSizeM = imagesizeX*processImageSizeY[0];
    const cl::NDRange local_item_size(min(imagesizeX,(int)maxWorkGroupSize),1,1);
    const cl::NDRange global_item_size(imagesizeX,processImageSizeY[0],1);
    const cl::NDRange global_item_size2(imagesizeX,processImageSizeY[0],MAX_KRSIZE);
    
    cl::Buffer mt_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImageSizeM*numEnergy, 0, NULL);
    cl::Buffer bkg_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImageSizeM, 0, NULL);
    cl::Buffer edgeJ_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImageSizeM, 0, NULL);
    cl::Buffer chi_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImageSizeM*MAX_KRSIZE, 0, NULL);
    cl::Buffer mask_img(context, CL_MEM_READ_WRITE, sizeof(cl_char)*processImageSizeM, 0, NULL);
    
    
    float* chiData;
    float* bkgData;
    float* edgeJData;
    chiData = new float[imageSizeM*MAX_KRSIZE];
    bkgData = new float[imageSizeM];
    edgeJData = new float[imageSizeM];
    for (int offset=0; offset<imageSizeM; offset+=processImageSizeM) {
		//transfer mt data to GPU
        for (int en=0; en<numEnergy; en++) {
            queue.enqueueWriteBuffer(mt_img, CL_TRUE, en*sizeof(cl_float)*processImageSizeM, sizeof(cl_float)*processImageSizeM, &mt_vec[en][offset]);
            queue.finish();
        }
        
        //set mask
        cl::Kernel kernel_mask(program[0],"setMask");
        kernel_mask.setArg(0, mt_img);
        kernel_mask.setArg(1, mask_img);
        queue.enqueueNDRangeKernel(kernel_mask, NULL, global_item_size, local_item_size, NULL, NULL);
        queue.finish();
        
        float lambda = inp.getLambda_t_fit();
        int num_trial = inp.getNumTrial_fit();
        PreEdgeRemoval(queue, program[0], mt_img, energy, bkg_img, imagesizeX, processImageSizeY[0],
                       E0, preEdgeStartEnNo, preEdgeEndEnNo, bkgfitmode, lambda, num_trial);
        
        PostEdgeEstimation(queue, program[1], mt_img, energy, bkg_img, edgeJ_img, imagesizeX, processImageSizeY[0], postEdgeStartEnNo, postEdgeEndEnNo, lambda, num_trial);
        
        SplineBkgRemoval(queue, program[2], mt_img, energy, w_factor, chi_img, imagesizeX, processImageSizeY[0], processImageSizeY[1], numEnergy, kstart, kend, Rbkg, kw, lambda, num_trial, true);
        
        
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
        
        
        //transfer chi, bkg, edgeJ data to memory
        for (int kn=0; kn<koffset+ksize; kn++) {
            queue.enqueueReadBuffer(chi_img, CL_TRUE, sizeof(cl_float)*processImageSizeM*kn, sizeof(cl_float)*processImageSizeM, &chiData[offset+kn*imageSizeM]);
            queue.finish();
        }
        queue.enqueueReadBuffer(bkg_img, CL_TRUE, 0, sizeof(cl_float)*processImageSizeM, &bkgData[offset]);
        queue.finish();
        queue.enqueueReadBuffer(edgeJ_img, CL_TRUE, 0, sizeof(cl_float)*processImageSizeM, &edgeJData[offset]);
        queue.finish();
    }
    
	for (int en = 0; en<numEnergy; en++) {
		delete[] mt_vec[en];
	}

    //output thread
    output_th_fit[thread_id].join();
    output_th_fit[thread_id]=thread(extraction_output_thread,AngleNo,output_dir,
                                    move(chiData),move(bkgData), move(edgeJData),imageSizeM,koffset+ksize);
    
    return 0;
}



int extraction_data_input_thread(cl::CommandQueue queue, vector<cl::Program> program,
                                 int AngleNo, int thread_id,input_parameter inp,
                                 cl::Buffer energy, cl::Buffer w_factor,
                                 vector<int> processImageSizeY){
    
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
                                   energy, w_factor, num_energy,move(mt_vec),processImageSizeY);
    
    return 0;
}

//dummy thread
static int dummy(){
    return 0;
}

int EXAFS_extraction_ocl(input_parameter inp, OCL_platform_device plat_dev_list){
    
    float preEdgeStartE  = inp.getPreEdgeStartEnergy();
    float preEdgeEndE    = inp.getPreEdgeEndEnergy();
    float postEdgeStartE = inp.getPostEdgeStartEnergy();
    float postEdgeEndE   = inp.getPostEdgeEndEnergy();
    float E0=inp.getE0();
    float kstart = inp.get_kstart();
    float kend   = inp.get_kend();
    int startAngleNo=inp.getStartAngleNo();
    int endAngleNo=inp.getEndAngleNo();
    int preEdgeFittingMode = inp.getPreEdgeFittingMode();
    int num_param_preEdge=1;
    int num_paramsq_preEdge=1;
    switch (preEdgeFittingMode) {
        case 0: //line
        num_param_preEdge=2;
        num_paramsq_preEdge=4;
        break;
        
        case 1: //victoreen
        num_param_preEdge=4;
        num_paramsq_preEdge=16;
        break;
        
        case 2: //victoreen
        num_param_preEdge=3;
        num_paramsq_preEdge=9;
        break;
        
        default:
        break;
    }
    int imageSizeX=inp.getImageSizeX();
    int imageSizeY=inp.getImageSizeY();
    
    
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
    int num_evec = (int)energy.size();
    float kmax = (float)floor((float)sqrt(energy[num_evec-1]*EFF)/KGRID-1.0f)*KGRID;
    kend = (float)fmin(kend,kmax-WIN_DK);
    inp.set_kend(kend);
    cout<< "spline start k: "<<kstart<<endl;
    cout<< "spline end k: "<<kend<<endl;
    float splineStartE = (float)fmax(0.0f,kstart-WIN_DK)*(float)fmax(0.0f,kstart-WIN_DK)/EFF;
    float splineEndE = (float)fmin(20.0f,kend+WIN_DK)*(float)fmin(20.0f,kend+WIN_DK)/EFF;
    float Rbkg = inp.getRbkg();
    int N_ctrlP = (int)floor(2.0*(kend-kstart)*Rbkg/PI)+1;
    cout <<"Number of knots "<<N_ctrlP<<endl;
    int preEdgeStartEnNo=0, preEdgeEndEnNo=0, postEdgeStartEnNo=0, postEdgeEndEnNo=0;
    int splineStartEnNo=0, splineEndEnNo=0;
    for (int en=0; en<energy.size(); en++) {
        preEdgeStartEnNo = (preEdgeStartE<=energy[en]) ? preEdgeStartEnNo:en;
        preEdgeEndEnNo = (preEdgeEndE<=energy[en]) ? preEdgeEndEnNo:en;
        postEdgeStartEnNo = (postEdgeStartE<=energy[en]) ? postEdgeStartEnNo:en;
        postEdgeEndEnNo = (postEdgeEndE<=energy[en]) ? postEdgeEndEnNo:en;
        splineStartEnNo = (splineStartE<=energy[en]) ? splineStartEnNo:en;
        splineEndEnNo = (splineEndE<=energy[en]) ? splineEndEnNo:en;
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
    
    //process imageSize control
    int ksize=(int)ceil((min(float(kend + WIN_DK),(float)MAX_KQ)-max(float(kstart-WIN_DK),0.0f))/KGRID)+1;
    int Rsize=(int)ceil((min(float(Rbkg + WIN_DR),(float)MAX_R))/RGRID)+1;
    
    
    
    //OpenCL Program
    vector<vector<cl::Program>> programs;
    vector<vector<int>> processImageSizeYs;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        processImageSizeYs.push_back(extractionGPUmemoryControl(imageSizeX, imageSizeY, ksize, Rsize, num_energy, preEdgeFittingMode, plat_dev_list.queue(i,0)));
        int processImageSizeM = imageSizeX*processImageSizeYs[i][0];
        
        
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
        programs_atPlat.push_back(cl::Program(plat_dev_list.context(i), source));//pre-edge用
        programs_atPlat.push_back(cl::Program(plat_dev_list.context(i), source));//postedge用
        programs_atPlat.push_back(cl::Program(plat_dev_list.context(i), source));//spline用
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
        oss0 << " -D IMAGESIZE_X=" << imageSizeX;
        oss0 << " -D IMAGESIZE_Y=" << processImageSizeYs[i][0];
        oss0 << " -D IMAGESIZE_M=" << processImageSizeM;
        oss0 << " -D ENERGY_NUM=" << num_energy;
        oss0 << " -D FFT_SIZE=" << FFT_SIZE;
        oss0 << " -D PARA_NUM=" << num_param_preEdge;
        oss0 << " -D PARA_NUM_SQ=" << num_paramsq_preEdge;
        string option0 = option+oss0.str();
        ostringstream oss1;
        oss1 << " -D IMAGESIZE_X=" << imageSizeX;
        oss1 << " -D IMAGESIZE_Y=" << processImageSizeYs[i][0];
        oss1 << " -D IMAGESIZE_M=" << processImageSizeM;
        oss1 << " -D ENERGY_NUM=" << num_energy;
        oss1 << " -D FFT_SIZE=" << FFT_SIZE;
        oss1 << " -D PARA_NUM=" << 4;
        oss1 << " -D PARA_NUM_SQ=" << 16;
        string option1 = option+oss1.str();
        ostringstream oss2;
        oss1 << " -D IMAGESIZE_X=" << imageSizeX;
        oss1 << " -D IMAGESIZE_Y=" << processImageSizeYs[i][0];
        oss1 << " -D IMAGESIZE_M=" << processImageSizeM;
        oss1 << " -D ENERGY_NUM=" << num_energy;
        oss2 << " -D FFT_SIZE=" << FFT_SIZE;
        oss2 << " -D PARA_NUM=" << N_ctrlP;
        oss2 << " -D PARA_NUM_SQ=" << N_ctrlP*N_ctrlP;
        string option2 = option+oss2.str();
        programs_atPlat[0].build(option0.c_str());
        programs_atPlat[1].build(option1.c_str());
        programs_atPlat[2].build(option2.c_str());
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
                                     An, i, inp, energy_buffers[i], w_factors[i],processImageSizeYs[i]);
                An++;
                if (An > endAngleNo) break;
            } else{
                this_thread::sleep_for(chrono::milliseconds(100));
            }
        }
        if (An > endAngleNo) break;
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
