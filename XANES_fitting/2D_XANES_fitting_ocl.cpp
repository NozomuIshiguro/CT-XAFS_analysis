//
//  2D_XANES_fitting_ocl.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2017/04/28.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#include "XANES_fitting.hpp"
#include "XANES_fit_cl.hpp"
#include "LevenbergMarquardt_cl.hpp"

int XANESGPUmemoryControl(int imageSizeX, int imageSizeY,
                                  int num_energy, fitting_eq fiteq, cl::CommandQueue queue){
    
    int processImageSizeY = imageSizeY;
    
    size_t GPUmemorySize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    
    size_t usingMemorySize;
    size_t paraN = fiteq.ParaSize();
    int numFunc = fiteq.numFunc;
    int numLCF = fiteq.numLCF;
    size_t numContrain = fiteq.contrain_size;
    do {
        usingMemorySize =0;
        //energy
        usingMemorySize += num_energy*sizeof(float);
        // Cmatrix
        usingMemorySize += paraN*numContrain*sizeof(float);
        //Dvector
        usingMemorySize += numContrain*sizeof(float);
        //Funcmode
        usingMemorySize += numFunc*sizeof(float);
        //refspectra
        usingMemorySize += numLCF*IMAGE_SIZE_E*sizeof(float);
        
        //mt_data,weight
        usingMemorySize += 2*imageSizeX*processImageSizeY*num_energy*sizeof(float);
        //mt fit
        usingMemorySize += imageSizeX*processImageSizeY*sizeof(float);
        //Jacobian
        usingMemorySize += imageSizeX*processImageSizeY*sizeof(float);
        //tJJ, inv_tJJ
        usingMemorySize += imageSizeX*processImageSizeY*paraN*(paraN+1)/2*sizeof(float);
        //tJdF, dp, fp, fp_cnd
        usingMemorySize += 4*imageSizeX*processImageSizeY*paraN*sizeof(float);
        //dF2_old, dF2_new, lambda, nyu, dL, rho,weight_thleshold
        usingMemorySize += 7*imageSizeX*processImageSizeY*sizeof(float);
        
        if(GPUmemorySize<usingMemorySize){
            processImageSizeY /=2;
        }else{
            break;
        }
        
    } while (processImageSizeY>1);
    
#ifdef DEBUG
    cout << "image size X: "<<imageSizeX<<endl;
    cout << "processing image size Y: " << processImageSizeY <<endl;
    cout << "GPU mem size "<<GPUmemorySize <<" bytes >= using mem size " << usingMemorySize<<" bytes"<<endl<<endl;
#endif
    
    return processImageSizeY;
}

string kernel_preprocessor_nums(fitting_eq fiteq,input_parameter inp, int processImageSizeY){
    ostringstream OSS;
    
    OSS<<"-D E0="<<inp.getE0();
    
    OSS<<" -D ENERGY_NUM="<<inp.getFittingEndEnergyNo()-inp.getFittingStartEnergyNo()+1;
    
    OSS<<" -D PARA_NUM="<< fiteq.ParaSize();
    
    OSS<<" -D PARA_NUM_SQ="<< fiteq.ParaSize()*fiteq.ParaSize();
    
    OSS<<" -D CONTRAIN_NUM="<< fiteq.contrain_size;
    
    OSS<<" -D START_E="<< inp.getStartEnergy() - inp.getE0();
    
    OSS<<" -D END_E="<< inp.getEndEnergy() - inp.getE0();
    
    OSS<<" -D FUNC_NUM="<< fiteq.numFunc;
    
    OSS<<" -D IMAGESIZE_X="<< inp.getImageSizeX();
    OSS<<" -D IMAGESIZE_Y="<< processImageSizeY;
    OSS<<" -D IMAGESIZE_M="<< inp.getImageSizeX()*processImageSizeY;
    
    
    if (inp.getCSbool()) OSS << " -D EPSILON="<<0.01f*(inp.getFittingEndEnergyNo() - inp.getFittingStartEnergyNo() + 1);
    else OSS << " -D EPSILON="<<0.0f;
    
    return OSS.str();
}

int fitresult_output_thread(fitting_eq fit_eq, int AngleNo,string output_dir,
                            vector<float*> result_outputs, int imageSizeM){
    
    ostringstream oss;
    for (int i=0; i<fit_eq.ParaSize(); i++) {
        char buffer;
        buffer=fit_eq.freefix_para()[i];
        if (atoi(&buffer)==1) {
            string fileName_output= output_dir+ "/"+fit_eq.param_name(i)+ AnumTagString(AngleNo,"/i", ".raw");
            oss << "output file: " << fileName_output << endl;
            outputRawFile_stream(fileName_output,result_outputs[i],imageSizeM);
        }
    }
    oss << endl;
    cout << oss.str();
    for (int i=0; i<fit_eq.ParaSize(); i++) {
        delete [] result_outputs[i];
    }
    return 0;
}

int data_input_thread(cl::CommandQueue command_queue, cl::Program program,
                      fitting_eq fiteq, int AngleNo, int thread_id,input_parameter inp,
                      cl::Buffer energy, cl::Buffer C_matrix_buff, cl::Buffer D_vector_buff,
                      cl::Buffer C2_vector_buff,cl::Buffer freeFix_buff, cl::Image1DArray refSpectra,
                      cl::Buffer funcMode_buff, int processImageSizeY){
    
    
    int startEnergyNo = inp.getFittingStartEnergyNo();
    int endEnergyNo = inp.getFittingEndEnergyNo();
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
        readRawFile(filepath_input[i],mt_vec[i],imgSizeM);
    }
    //m1.unlock();
    
    fitting_th[thread_id].join();
    fitting_th[thread_id] = thread(XANES_fit_thread,command_queue, program,
                                   fiteq,AngleNo,thread_id,inp,
                                   energy, C_matrix_buff, D_vector_buff, C2_vector_buff,
                                   freeFix_buff,refSpectra,funcMode_buff,
                                   move(mt_vec),0,processImageSizeY);
    
    
    return 0;
}

//dummy thread
static int dummy(){
    return 0;
}

int XANES_fit_thread(cl::CommandQueue command_queue, cl::Program program,
                     fitting_eq fiteq,int AngleNo, int thread_id,input_parameter inp,
                     cl::Buffer energy, cl::Buffer C_matrix_buff, cl::Buffer D_vector_buff,
                     cl::Buffer C2_vector_buff, cl::Buffer freeFix_buff, cl::Image1DArray refSpectra,
                     cl::Buffer funcMode_buff, vector<float*> mt_vec, int64_t offset,
                     int processImgSizeY)
{
	string errorzone = "0";
	try {
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        string devicename = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>();
        size_t maxWorkGroupSize = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        
        int fittingStartEnergyNo = inp.getFittingStartEnergyNo();
        int fittingEndEnergyNo = inp.getFittingEndEnergyNo();
        int startEnergyNo=inp.getStartEnergyNo();
        int num_energy = fittingEndEnergyNo - fittingStartEnergyNo + 1;
        int paramsize = (int)fiteq.ParaSize();
        const int imgSizeX = inp.getImageSizeX();
        const int imgSizeY = inp.getImageSizeY();
        const int imgSizeM = inp.getImageSizeM();
        const int processImgSizeM = imgSizeX*processImgSizeY;
        string output_dir=inp.getFittingOutputDir();
        float CI=10.0f;
        int contrainSize = (int)fiteq.contrain_size;
        
        const cl::NDRange local_item_size(min(imgSizeX,(int)maxWorkGroupSize),1,1);
        const cl::NDRange global_item_size(imgSizeX,processImgSizeY,1);
        const cl::NDRange global_item_size2(imgSizeX,processImgSizeY,paramsize);
        cout << "Processing angle No. " << AngleNo << "..." << endl << endl;
        //cout << "     global worksize :" << IMAGE_SIZE_X << ", " << IMAGE_SIZE_Y << ", 1" << endl;
        //cout << "     local worksize :" << maxWorkGroupSize << ", 1, 1" << endl << endl;
        
        
        cl::Buffer mt_fit_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImgSizeM, 0, NULL);
        cl::Buffer Jacobian_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImgSizeM*paramsize, 0, NULL);
        cl::Buffer tJdF_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImgSizeM*paramsize, 0, NULL);
        cl::Buffer tJJ_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImgSizeM*paramsize*(paramsize+1)/2, 0, NULL);
        cl::Buffer nyu_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImgSizeM, 0, NULL);
        cl::Buffer lambda_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImgSizeM, 0, NULL);
        cl::Buffer chi2_old_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImgSizeM,0,NULL);
        cl::Buffer chi2_new_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImgSizeM,0,NULL);
        cl::Buffer mask_buff(context, CL_MEM_READ_WRITE, sizeof(cl_char)*processImgSizeM, 0, NULL);
        cl::Buffer results_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImgSizeM*paramsize, 0, NULL);
        cl::Buffer results_cnd_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImgSizeM*paramsize, 0, NULL);
        cl::Buffer dp_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImgSizeM*paramsize, 0, NULL);
        cl::Buffer mt_img(context, CL_MEM_READ_ONLY, sizeof(cl_float)*processImgSizeM*num_energy, 0, NULL);
        cl::Buffer weight_img(context, CL_MEM_READ_ONLY, sizeof(cl_float)*processImgSizeM*num_energy,0,NULL);
		cl::Buffer dL_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImgSizeM, 0, NULL);
        cl::Buffer rho_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImgSizeM, 0, NULL);
        cl::Buffer weight_thleshold_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImgSizeM,0,NULL);
        cl::Buffer eval_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImgSizeM,0,NULL);
        cl::Buffer contrainWgt_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*processImgSizeM,0,NULL);
        command_queue.enqueueFillBuffer(contrainWgt_img, (cl_float)1.0f, 0, sizeof(cl_float)*processImgSizeM);
        cl::Buffer Lambda_fista(context, CL_MEM_READ_WRITE, sizeof(cl_float)*paramsize, 0, NULL);

        
        //cout<<ret<<endl;
        cl::Kernel kernel_chi2tJdFtJJ(program,"chi2_tJdF_tJJ_Stack");
        cl::Kernel kernel_chi2new(program,"chi2Stack");
        cl::Kernel kernel_LM(program,"LevenbergMarquardt");
        cl::Kernel kernel_dL(program,"estimate_dL");
        cl::Kernel kernel_cnd(program,"updatePara");
        cl::Kernel kernel_contrain1(program,"contrain_1");
        cl::Kernel kernel_contrain2(program,"contrain_2");
        cl::Kernel kernel_evalUpdate(program,"evaluateUpdateCandidate");
        cl::Kernel kernel_UorH(program,"updateOrHold");
        cl::Kernel kernel_mask(program,"setMask");
        cl::Kernel kernel_threshold(program,"applyMask");
        cl::Kernel kernel_FISTA(program,"FISTA");
        
        
        //set kernel arguments
        //chi2(old), tJdF, tJJ calculation
        kernel_chi2tJdFtJJ.setArg(0, mt_img);
        kernel_chi2tJdFtJJ.setArg(1, results_img);
        kernel_chi2tJdFtJJ.setArg(2, refSpectra);
        kernel_chi2tJdFtJJ.setArg(3, chi2_old_buff);
        kernel_chi2tJdFtJJ.setArg(4, tJdF_buff);
        kernel_chi2tJdFtJJ.setArg(5, tJJ_buff);
        kernel_chi2tJdFtJJ.setArg(6, energy);
        kernel_chi2tJdFtJJ.setArg(7, funcMode_buff);
        kernel_chi2tJdFtJJ.setArg(8, freeFix_buff);
        kernel_chi2tJdFtJJ.setArg(9, (cl_int)0);
        kernel_chi2tJdFtJJ.setArg(10, (cl_int)(num_energy-1));
        kernel_chi2tJdFtJJ.setArg(11, weight_img);
        kernel_chi2tJdFtJJ.setArg(12, weight_thleshold_img);
        //chi2(new) calculation
        kernel_chi2new.setArg(0, mt_img);
        kernel_chi2new.setArg(1, results_cnd_img);
        kernel_chi2new.setArg(2, refSpectra);
        kernel_chi2new.setArg(3, chi2_new_buff);
        kernel_chi2new.setArg(4, energy);
        kernel_chi2new.setArg(5, funcMode_buff);
        kernel_chi2new.setArg(6, (cl_int)0);
        kernel_chi2new.setArg(7, (cl_int)(num_energy-1));
        kernel_chi2new.setArg(8, weight_img);
        kernel_chi2new.setArg(9, weight_thleshold_img);
        kernel_chi2new.setArg(10, (cl_float)CI);
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
        kernel_cnd.setArg(1, results_cnd_img);
        kernel_cnd.setArg(2, (cl_int)0);
        kernel_cnd.setArg(3, (cl_int)0);
        //contrain
        kernel_contrain1.setArg(0, results_cnd_img);
        kernel_contrain1.setArg(1, eval_img);
        kernel_contrain1.setArg(2, C_matrix_buff);
        kernel_contrain2.setArg(0, results_cnd_img);
        kernel_contrain2.setArg(1, contrainWgt_img);
        kernel_contrain2.setArg(2, eval_img);
        kernel_contrain2.setArg(3, C_matrix_buff);
        kernel_contrain2.setArg(4, D_vector_buff);
        kernel_contrain2.setArg(5, C2_vector_buff);
        kernel_contrain2.setArg(8, (cl_char)48);
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
        kernel_UorH.setArg(0, results_img);
        kernel_UorH.setArg(1, results_cnd_img);
        kernel_UorH.setArg(2, rho_buff);
        
        
        //allocate memory for results
        vector<float*> results_vec;
        for (int i=0; i<paramsize; i++) {
            results_vec.push_back(new float[imgSizeM]);
        }
        
        
        errorzone = "setting parameters for CS to GPU";
        //cout << paramsize << endl;
        if (inp.getCSbool()) {
            for (int i=0; i<paramsize; i++) {
                float val = (fiteq.freefix_para()[i] == 49) ? inp.getCSlambda()[i] : 0.0f;
                command_queue.enqueueFillBuffer(Lambda_fista, (cl_float)val, sizeof(cl_float)*i, sizeof(cl_float), NULL, NULL);
                command_queue.finish();
            }
            //FISTA
            kernel_FISTA.setArg(0, results_img);
            kernel_FISTA.setArg(1, dp_img);
            kernel_FISTA.setArg(2, tJJ_buff);
            kernel_FISTA.setArg(3, tJdF_buff);
            kernel_FISTA.setArg(4, freeFix_buff);
            kernel_FISTA.setArg(5, lambda_buff);
            kernel_FISTA.setArg(6, Lambda_fista);
        }
        
        
        for (int offsetY=0; offsetY<imgSizeY; offsetY+=processImgSizeY) {
            errorzone = "inputing parameters to GPU";
            //input LM parameters
            command_queue.enqueueFillBuffer(lambda_buff, (cl_float)inp.getLambda_t_fit(), 0, sizeof(cl_float)*processImgSizeM,NULL,NULL);
            command_queue.finish();
            command_queue.enqueueFillBuffer(nyu_buff, (cl_float)2.0f, 0, sizeof(cl_float)*processImgSizeM,NULL,NULL);
            command_queue.finish();
            for (int i=0; i<paramsize; i++) {
                command_queue.enqueueFillBuffer(results_img, (cl_float)fiteq.fit_para()[i], sizeof(cl_float)*processImgSizeM*i, sizeof(cl_float)*processImgSizeM,NULL,NULL);
                command_queue.finish();
            }
            
            
            errorzone = "writing 2D-XAFS images to GPU";
            //write buffer
            for (int i = fittingStartEnergyNo; i<=fittingEndEnergyNo; i++) {
                command_queue.enqueueWriteBuffer(mt_img, CL_TRUE, sizeof(float)*processImgSizeM*(i-fittingStartEnergyNo), sizeof(float)*processImgSizeM, &mt_vec[i-startEnergyNo][offset+offsetY]);
                command_queue.finish();
            }
            
            
            errorzone = "initial weighting";
            //chi2(new) calculation
            command_queue.enqueueFillBuffer(weight_img, (cl_float)1.0f, 0, sizeof(float)*processImgSizeM*num_energy);
            command_queue.enqueueNDRangeKernel(kernel_chi2new, NULL, global_item_size, local_item_size, NULL, NULL);
            command_queue.finish();
            
            
            //L-M trial
            for (int trial=0; trial<inp.getNumTrial_fit(); trial++) {
                errorzone = "calculating chi2 old, tJJ, tJdF";
                //chi2(old), tJdF, tJJ calculation
                command_queue.enqueueNDRangeKernel(kernel_chi2tJdFtJJ, NULL, global_item_size, local_item_size, NULL, NULL);
                command_queue.finish();
                
                
                errorzone = "L-M";
                //L-M equation
                command_queue.enqueueNDRangeKernel(kernel_LM, NULL, global_item_size, local_item_size, NULL, NULL);
                command_queue.finish();
                
                
                //FISTA (Soft Thresholding Function)
                if(inp.getCSbool()){
                    errorzone = "FISTA";
                    for (int k = 0; k < inp.getCSit(); k++) {
                        command_queue.enqueueNDRangeKernel(kernel_FISTA, NULL, global_item_size, local_item_size, NULL, NULL);
                        command_queue.finish();
                    }
                }
                
                
                errorzone = "estimating dL";
                //estimate dL
                command_queue.enqueueNDRangeKernel(kernel_dL, NULL, global_item_size, local_item_size, NULL, NULL);
                command_queue.finish();
                
                
                errorzone = "estimating fp_cnd";
                //estimate fp candidate
                command_queue.enqueueCopyBuffer(results_img, results_cnd_img, 0, 0, sizeof(cl_float)*processImgSizeM*paramsize);
                command_queue.enqueueNDRangeKernel(kernel_cnd, NULL, global_item_size2, local_item_size, NULL, NULL);
                command_queue.finish();
                
                
                //contrain
                errorzone = "setting contrain";
                for (int i=0; i<contrainSize; i++) {
                    kernel_contrain1.setArg(3, (cl_int)i);
                    kernel_contrain2.setArg(6, (cl_int)i);
                    command_queue.enqueueFillBuffer(eval_img, (cl_float)0.0f, 0, sizeof(cl_float)*processImgSizeM);
                    for (int j=0; j<paramsize; j++) {
                        const cl::NDRange global_item_offset(0,0,j);
                        kernel_contrain1.setArg(4, (cl_int)j);
                        command_queue.enqueueNDRangeKernel(kernel_contrain1, global_item_offset, global_item_size, local_item_size, NULL, NULL);
                        command_queue.finish();
                    }
                    for (int j=0; j<paramsize; j++) {
                        const cl::NDRange global_item_offset(0,0,j);
                        kernel_contrain2.setArg(7, (cl_int)j);
                        command_queue.enqueueNDRangeKernel(kernel_contrain2, global_item_offset, global_item_size, local_item_size, NULL, NULL);
                        command_queue.finish();
                    }
                }
                
                
                errorzone = "calclating new chi2";
                //chi2(new) calculation
                command_queue.enqueueNDRangeKernel(kernel_chi2new, NULL, global_item_size, local_item_size, NULL, NULL);
                command_queue.finish();
                
                
                errorzone = "evaluateUpdateCandidate";
                //evaluate rho
                command_queue.enqueueNDRangeKernel(kernel_evalUpdate, NULL, global_item_size, local_item_size, NULL, NULL);
                command_queue.finish();
                //update or hold parameter
                command_queue.enqueueNDRangeKernel(kernel_UorH, NULL, global_item_size2, local_item_size, NULL, NULL);
                command_queue.finish();
            }
            
            
            errorzone = "setting mask";
            //set mask
            kernel_mask.setArg(0, mt_img);
            kernel_mask.setArg(1, mask_buff);
            command_queue.enqueueNDRangeKernel(kernel_mask, NULL, global_item_size, local_item_size, NULL, NULL);
            command_queue.finish();
            
            
            errorzone = "applying threshold";
            //apply threshold
            kernel_threshold.setArg(0, results_img);
            kernel_threshold.setArg(1, mask_buff);
            command_queue.enqueueNDRangeKernel(kernel_threshold, NULL, global_item_size2, local_item_size, NULL, NULL);
            command_queue.finish();
            
            
            errorzone = "read results from GPU";
            //read results image from buffer
            for (int i = 0; i<paramsize; i++) {
                command_queue.enqueueReadBuffer(results_img, CL_TRUE, sizeof(cl_float)*processImgSizeM*i, sizeof(cl_float)*processImgSizeM, &results_vec[i][offsetY]);
                command_queue.finish();
            }
        }
        
        
        //delete mt data
#ifdef NOT_IMGREG
        for (int i = 0; i<num_energy; i++) {
            delete[] mt_vec[i];
        }
#endif
        
        //output thread
        output_th_fit[thread_id].join();
        output_th_fit[thread_id]=thread(fitresult_output_thread, fiteq, AngleNo,
                                        output_dir,move(results_vec),imgSizeM);
        
    } catch (const cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")"<< errorzone << endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(ret.err());
    }
    
    return 0;

}

int XANES_fit_ocl(fitting_eq fiteq, input_parameter inp,
                  OCL_platform_device plat_dev_list)
{
    float startEnergy=inp.getStartEnergy();
    float endEnergy=inp.getEndEnergy();
    float E0=inp.getE0();
    int startAngleNo=inp.getStartAngleNo();
    int endAngleNo=inp.getEndAngleNo();
    
    
    //create output dir
    for (int i=0; i<fiteq.ParaSize(); i++) {
        char buffer=fiteq.freefix_para()[i];
        if (atoi(&buffer)==1) {
            string fileName_output = inp.getFittingOutputDir() + "/"+fiteq.param_name(i);
            MKDIR(fileName_output.c_str());
        }
    }
    
    
    //energy file input & processing
    ifstream energy_ifs(inp.getEnergyFilePath(),ios::in);
    if (!energy_ifs.is_open()) {
        cout<<"energy file not found."<<endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(-1);
    }
    vector<float> energy;
    int i=0, startEnergyNo=0, endEnergyNo=0;
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
        if ((aa>=startEnergy)&(aa<=endEnergy)) {
            energy.push_back(aa-E0);
            cout<<" <- fitting range";
            endEnergyNo = i;
        } else if(aa<startEnergy) {
            startEnergyNo = i+1;
        }
        cout<<endl;
        i++;
    } while (!energy_ifs.eof());
    int num_energy=endEnergyNo-startEnergyNo+1;
    inp.setFittingStartEnergyNo(startEnergyNo);
    inp.setFittingEndEnergyNo(endEnergyNo);
    ostringstream oss;
    oss << startEnergyNo<<"-"<<endEnergyNo;
    inp.setEnergyNoRange(oss.str());
    oss.flush();
    cout << "energy num for fitting: "<<num_energy<<endl<<endl;
    
    
    //kernel program source
    /*ifstream ifs("E:/Dropbox/CTprogram/XANES_fitting/XANES_fit.cl", ios::in);
     if(!ifs) {
     cerr << "   Failed to load kernel \n" << endl;
     return -1;
     }
     istreambuf_iterator<char> it(ifs);
     istreambuf_iterator<char> last;
     string kernel_code(it,last);
     ifs.close();*/
    
    
    //OpenCL Program
	vector<cl::Program> programs;
    vector<int> processImageSizeY;
    int imageSizeX = inp.getImageSizeX();
    int imageSizeY = inp.getImageSizeY();
	try {
		for (int i = 0; i<plat_dev_list.contextsize(); i++) {
			processImageSizeY.push_back(XANESGPUmemoryControl(imageSizeX,imageSizeY,num_energy,fiteq, plat_dev_list.queue(i,0)));
            cl::Program::Sources source;
#if defined (OCL120)
			source.push_back(std::make_pair(kernel_fit_src.c_str(), kernel_fit_src.length()));
			source.push_back(std::make_pair(kernel_LM_src.c_str(), kernel_LM_src.length()));
#else
			source.push_back(kernel_fit_src);
			source.push_back(kernel_LM_src);
#endif
			programs.push_back(cl::Program(plat_dev_list.context(i), source));
			//kernel build
			string option = "";
#ifdef DEBUG
			option += "-D DEBUG -Werror ";
#endif
			option += kernel_preprocessor_nums(fiteq, inp, processImageSizeY[i]);
			string GPUvendor = plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0].getInfo<CL_DEVICE_VENDOR>();
			if (GPUvendor == "nvidia") {
				option += " -cl-nv-maxrregcount=64 -cl-nv-verbose";
			}
			else if (GPUvendor.find("NVIDIA Corporation") == 0) {
				option += " -cl-nv-maxrregcount=64";
			}
			//cout << option << endl;
			programs[i].build(option.c_str());
#ifdef DEBUG
			string logstr = programs[i].getBuildInfo<CL_PROGRAM_BUILD_LOG>(plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0]);
			cout << logstr << endl;
#endif
		}
	}
	catch (const cl::Error ret) {
		cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
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
    vector<cl::Buffer> C_matrix_buffers;
    vector<cl::Buffer> D_vector_buffers;
    vector<cl::Buffer> C2_vector_buffers;
    vector<cl::Buffer> freeFix_buffers;
    vector<cl::Buffer> funcMode_buffers;
    cl::ImageFormat format(CL_R, CL_FLOAT);
    vector<cl::Image1DArray> refSpectra;
    int paramsize = (int)fiteq.ParaSize();
    int contrainsize = (int)fiteq.contrain_size;
    const cl::NDRange local_item_size(1,1,1);
    const cl::NDRange global_item_size(1,1,1);
    const cl::NDRange global_item_size2(contrainsize,1,1);
    for (int i=0; i<plat_dev_list.contextsize(); i++){
        energy_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*num_energy, 0, NULL));
        C_matrix_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*paramsize*contrainsize, 0, NULL));
        D_vector_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*contrainsize, 0, NULL));
        C2_vector_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float)*contrainsize, 0, NULL));
        freeFix_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_char)*paramsize, 0, NULL));
        funcMode_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_int)*fiteq.numFunc, 0, NULL));
        refSpectra.push_back(cl::Image1DArray(plat_dev_list.context(i), CL_MEM_READ_WRITE,format,fiteq.numLCF,IMAGE_SIZE_E, 0, NULL));
        
        
        //write buffers
        plat_dev_list.queue(i,0).enqueueWriteBuffer(energy_buffers[i], CL_TRUE, 0, sizeof(cl_float)*num_energy, &energy[0], NULL, NULL);
        for (int j = 0; j < contrainsize; j++) {
            plat_dev_list.queue(i, 0).enqueueWriteBuffer(C_matrix_buffers[i], CL_TRUE, sizeof(cl_float)*j*paramsize, sizeof(cl_float)*paramsize, &(fiteq.C_matrix[j][0]), NULL, NULL);
        }
        plat_dev_list.queue(i,0).enqueueWriteBuffer(D_vector_buffers[i], CL_TRUE, 0, sizeof(cl_float)*contrainsize, &fiteq.D_vector[0], NULL, NULL);
        plat_dev_list.queue(i,0).enqueueWriteBuffer(freeFix_buffers[i], CL_TRUE, 0, sizeof(cl_char)*paramsize, fiteq.freefix_para(), NULL, NULL);
        plat_dev_list.queue(i,0).enqueueWriteBuffer(funcMode_buffers[i], CL_TRUE, 0, sizeof(cl_int)*fiteq.numFunc, &(fiteq.funcmode[0]), NULL, NULL);
        
        //write buffers for LCF
        cl::Kernel kernel_LCFstd(programs[i],"redimension_refSpecta");
        cl::size_t<3> origin;
        cl::size_t<3> region;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        region[1] = 1;
        region[2] = 1;
        for (int j=0; j<fiteq.numLCF; j++) {
            cl::Image1D refSpectraRaw(plat_dev_list.context(i), CL_MEM_READ_ONLY,format, fiteq.LCFstd_size[j], 0, NULL);
            cl::Buffer refSpectraRawE(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*fiteq.LCFstd_size[j], 0, NULL);
            
            region[0] = fiteq.LCFstd_size[j];
            plat_dev_list.queue(i,0).enqueueWriteImage(refSpectraRaw,CL_TRUE,origin,region,0,0,&(fiteq.LCFstd_mt[j][0]));
            plat_dev_list.queue(i,0).enqueueWriteBuffer(refSpectraRawE, CL_TRUE,0,sizeof(cl_float)*fiteq.LCFstd_size[j], &(fiteq.LCFstd_E[j][0]));
            
            kernel_LCFstd.setArg(0, refSpectra[i]);
            kernel_LCFstd.setArg(1, refSpectraRaw);
            kernel_LCFstd.setArg(2, refSpectraRawE);
            kernel_LCFstd.setArg(3, (cl_int)fiteq.LCFstd_size[j]);
            kernel_LCFstd.setArg(4, (cl_int)j);
            
            plat_dev_list.queue(i,0).enqueueNDRangeKernel(kernel_LCFstd, NULL, global_item_size, local_item_size, NULL, NULL);
            plat_dev_list.queue(i,0).finish();
        }
        
        
        //estimate C2
        cl::Kernel kernel_contrain0(programs[i],"contrain_0");
        kernel_contrain0.setArg(0, C_matrix_buffers[i]);
        kernel_contrain0.setArg(1, C2_vector_buffers[i]);
        plat_dev_list.queue(i,0).enqueueNDRangeKernel(kernel_contrain0, NULL, global_item_size2, local_item_size, NULL, NULL);
        plat_dev_list.queue(i,0).finish();
    }

	/*float *spectra;
	spectra = new float[IMAGE_SIZE_E*2];
	cl::size_t<3> origin;
	cl::size_t<3> region;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;
	region[0] = IMAGE_SIZE_E;
	region[1] = 2;
	region[2] = 1;
	plat_dev_list.queue(0, 0).enqueueReadImage(refSpectra[0], CL_TRUE, origin, region, 0, 0, spectra);
	string output_dir = inp.getFittingOutputDir();
	string fileName_output = output_dir + AnumTagString(1, "/sp", ".raw");
	outputRawFile_stream(fileName_output, spectra, IMAGE_SIZE_E*2);*/

    
    
    //start thread
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        input_th.push_back(thread(dummy));
        fitting_th.push_back(thread(dummy));
        output_th_fit.push_back(thread(dummy));
    }
    for (int i=startAngleNo; i<=endAngleNo;) {
        for (int j=0; j<plat_dev_list.contextsize(); j++) {
            if (input_th[j].joinable()) {
                input_th[j].join();
                input_th[j] = thread(data_input_thread,
                                     plat_dev_list.queue(j,0),programs[j],
                                     fiteq,i,j,inp,energy_buffers[j], C_matrix_buffers[j],
                                     D_vector_buffers[j],C2_vector_buffers[j],
                                     freeFix_buffers[j],refSpectra[j],funcMode_buffers[j],
                                     processImageSizeY[j]);
                i++;
                if (i > endAngleNo) break;
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
