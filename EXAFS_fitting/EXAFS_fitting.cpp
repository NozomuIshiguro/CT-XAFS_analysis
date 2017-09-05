//
//  2D_EXAFS_fitting_ocl.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2016/01/04.
//  Copyright © 2016年 Nozomu Ishiguro. All rights reserved.
//

#include "EXAFS.hpp"

vector<int> GPUmemoryControl(int imageSizeX, int imageSizeY,int ksize,int Rsize,int qsize,
                             int fittingMode, int num_fpara, int shellnum, cl::CommandQueue queue){
    
    vector<int> processImageSizeY;
    processImageSizeY.push_back(imageSizeY); //for other process
    processImageSizeY.push_back(imageSizeY); //for FFT/IFFT size
    
    size_t GPUmemorySize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    
    size_t usingMemorySize;
    //size_t shellnum = shObjs.size();
    do {
        usingMemorySize =0;
        //S0 image
        usingMemorySize += imageSizeX*processImageSizeY[0]*sizeof(float);
        //feff data in shObs
        usingMemorySize += shellnum*MAX_KRSIZE*6*sizeof(float);
        //parameter images in shObs
        usingMemorySize += shellnum*imageSizeX*processImageSizeY[0]*7*sizeof(float);
        //dF_old, dF_new, nyu, lambda, dL, rho
        usingMemorySize += 6*imageSizeX*processImageSizeY[0]*sizeof(float);
        //tJJ
        usingMemorySize += imageSizeX*processImageSizeY[0]*num_fpara*(num_fpara+1)/2*sizeof(float);
        //tJdF, dp, para_backup
         usingMemorySize += 3*imageSizeX*processImageSizeY[0]*num_fpara*sizeof(float);
        
        switch (fittingMode) {
            case 0: //k-fit
                //chi data & chi fit
                usingMemorySize += 2*imageSizeX*processImageSizeY[0]*ksize*sizeof(cl_float2);
                //Jacobian
                usingMemorySize += imageSizeX*processImageSizeY[0]*ksize*num_fpara*sizeof(cl_float2);
                break;
                
            case 1: //R-fit
                //spin factor
                usingMemorySize += FFT_SIZE/2*sizeof(cl_float2);
                //chi fit, Jacobian_dummy_k
                usingMemorySize += 2*imageSizeX*processImageSizeY[0]*ksize*sizeof(cl_float2);
                //FTchi data, FTchi fit
                usingMemorySize += 2*imageSizeX*processImageSizeY[0]*Rsize*sizeof(cl_float2);
                //Jacobian
                usingMemorySize += imageSizeX*processImageSizeY[0]*Rsize*num_fpara*sizeof(cl_float2);
                break;
                
            case 2: //q-fit
                //spin factor
                usingMemorySize += FFT_SIZE/2*sizeof(cl_float2);
                //chi fit, Jacobian_dummy_k
                usingMemorySize += 2*imageSizeX*processImageSizeY[0]*ksize*sizeof(cl_float2);
                //FTchi fit, Jacobian_dummy_R
                usingMemorySize += 2*imageSizeX*processImageSizeY[0]*Rsize*sizeof(cl_float2);
                //chiq data, chiq fit
                usingMemorySize += 2*imageSizeX*processImageSizeY[0]*qsize*sizeof(cl_float2);
                //Jacobian
                usingMemorySize += imageSizeX*processImageSizeY[0]*qsize*num_fpara*sizeof(cl_float2);
                break;
        }
        
        //FFT/IFFT objects
        if(fittingMode>0){
            size_t usingMemorySizeFFT;
            do {
                usingMemorySizeFFT=usingMemorySize;
                
                
                usingMemorySizeFFT += imageSizeX*processImageSizeY[1]*FFT_SIZE*sizeof(cl_float2);
                
                if(GPUmemorySize<usingMemorySizeFFT){
                    processImageSizeY[1] /=2;
                }else{
                    break;
                }
            } while (processImageSizeY[1]>processImageSizeY[0]/8);
            usingMemorySize = usingMemorySizeFFT;
        }
    
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


int createSpinFactor(cl::Buffer w_buffer, cl::CommandQueue queue, cl::Program program){
    
    size_t maxWorkGroupSize = min(FFT_SIZE/2,(int)queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>());
    const cl::NDRange local_item_size(maxWorkGroupSize,1,1);
    const cl::NDRange global_item_size(FFT_SIZE/2,1,1);
    
    //set kernel (spinFact)
    cl::Kernel kernel_SF(program,"spinFact");
    kernel_SF.setArg(0, w_buffer);
    //spinFact
    queue.enqueueNDRangeKernel(kernel_SF, NULL, global_item_size, local_item_size, NULL, NULL);
    queue.finish();
    
    return 0;
}


int FFT(cl::Buffer chi, cl::Buffer FTchi,
        cl::Buffer w_buffer, cl::CommandQueue queue, cl::Program program,
        int imageSizeX, int imageSizeY, int FFTimageSizeY, int offsetY,
        int koffset, int ksize, int Roffset, int Rsize){
    
	try {
		int imageSizeM = imageSizeX*FFTimageSizeY;
		vector<size_t> maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
		cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
		cl::Buffer FTchi_dum(context, CL_MEM_READ_WRITE, sizeof(cl_float2)*imageSizeM*FFT_SIZE, 0, NULL);
		cl_float2 ini = {0.0f,0.0f};
		queue.enqueueFillBuffer(FTchi_dum, ini, 0, sizeof(cl_float2)*imageSizeM*FFT_SIZE);
		//cout << imageSizeX <<" "<<imageSizeY<<" "<<ksize<<endl;
		//cout << Roffset <<" " << Rsize<<endl;


		int log_FFTsize = (int)log2(FFT_SIZE);
		//cout << log_FFTsize<<endl;


		//set kernel (bitReverse and XZ transpose)
		cl::Kernel kernel_BRXZtrans(program, "bitReverseAndXZ_transpose");
		kernel_BRXZtrans.setArg(0, chi);
		kernel_BRXZtrans.setArg(1, FTchi_dum);
		kernel_BRXZtrans.setArg(2, (cl_int)log_FFTsize);
		kernel_BRXZtrans.setArg(3, (cl_int)0);
		kernel_BRXZtrans.setArg(4, (cl_int)imageSizeX);
		kernel_BRXZtrans.setArg(5, (cl_int)koffset);
		kernel_BRXZtrans.setArg(6, (cl_int)FFT_SIZE);
		kernel_BRXZtrans.setArg(7, (cl_int)offsetY);
		kernel_BRXZtrans.setArg(8, (cl_int)imageSizeY);
		kernel_BRXZtrans.setArg(9, (cl_int)0);
		kernel_BRXZtrans.setArg(10, (cl_int)FFTimageSizeY);
		const cl::NDRange global_item_size1(imageSizeX, FFTimageSizeY, ksize);
		const cl::NDRange local_item_size1(min(imageSizeX, (int)maxWorkGroupSize[0]), 1, 1);
		queue.enqueueNDRangeKernel(kernel_BRXZtrans, NULL, global_item_size1, local_item_size1, NULL, NULL);
		queue.finish();


		//set kernel (butterflyIMGarray)
		cl::Kernel kernel_butterfly(program, "butterfly");
		kernel_butterfly.setArg(0, FTchi_dum);
		kernel_butterfly.setArg(1, w_buffer);
		kernel_butterfly.setArg(3, (cl_uint)0x0); //flag
		const cl::NDRange global_item_size2(FFT_SIZE / 2, FFTimageSizeY, imageSizeX);
		const cl::NDRange local_item_size2(min(FFT_SIZE / 2, (int)maxWorkGroupSize[0]), 1, 1);
		for (int iter = 1; iter <= log_FFTsize; iter++) {
			kernel_butterfly.setArg(2, (cl_uint)iter);
			queue.enqueueNDRangeKernel(kernel_butterfly, NULL, global_item_size2, local_item_size2, NULL, NULL);
			queue.finish();
		}


		//set kernel (FFTnormMGarray)
		cl::Kernel kernel_norm(program, "FFTnorm");
		kernel_norm.setArg(0, FTchi_dum);
		kernel_norm.setArg(1, (cl_float)KGRID);
		const cl::NDRange global_item_size3(FFT_SIZE, FFTimageSizeY, imageSizeX);
		const cl::NDRange local_item_size3(min(FFT_SIZE, (int)maxWorkGroupSize[0]), 1, 1);
		queue.enqueueNDRangeKernel(kernel_norm, NULL, global_item_size3, local_item_size3, NULL, NULL);
		queue.finish();

		
		//set kernel (XZ transpose)
		cl::Kernel kernel_XZtrans(program, "XZ_transpose");
		kernel_XZtrans.setArg(0, FTchi_dum);
		kernel_XZtrans.setArg(1, FTchi);
		kernel_XZtrans.setArg(2, (cl_int)Roffset);
		kernel_XZtrans.setArg(3, (cl_int)FFT_SIZE);
		kernel_XZtrans.setArg(4, (cl_int)0);
		kernel_XZtrans.setArg(5, (cl_int)imageSizeX);
		kernel_XZtrans.setArg(6, (cl_int)0);
		kernel_XZtrans.setArg(7, (cl_int)FFTimageSizeY);
		kernel_XZtrans.setArg(8, (cl_int)offsetY);
		kernel_XZtrans.setArg(9, (cl_int)imageSizeY);
		const cl::NDRange global_item_size4(Rsize, FFTimageSizeY, imageSizeX);
		const cl::NDRange local_item_size4(1, 1, min(imageSizeX, (int)maxWorkGroupSize[2]));
		queue.enqueueNDRangeKernel(kernel_XZtrans, NULL, global_item_size4, local_item_size4, NULL, NULL);
		queue.finish();
	}
	catch (cl::Error ret) {
		cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(ret.err());
	}

    
    
    
    return 0;
}


int IFFT(cl::Buffer FTchi, cl::Buffer chiq,
        cl::Buffer w_buffer, cl::CommandQueue queue, cl::Program program,
        int imageSizeX, int imageSizeY, int FFTimageSizeY, int offsetY,
        int Roffset, int Rsize, int qoffset, int qsize){
    
    
	try {
		int imageSizeM = imageSizeX*FFTimageSizeY;
		vector<size_t> maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
		cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();


		cl::Buffer chiq_dum(context, CL_MEM_READ_WRITE, sizeof(cl_float2)*imageSizeM*FFT_SIZE, 0, NULL);
		cl_float2 ini = {0.0f,0.0f};
		queue.enqueueFillBuffer(chiq_dum, ini, 0, sizeof(cl_float2)*imageSizeM*FFT_SIZE);
		//cout << imageSizeX <<" "<<imageSizeY<<" "<<ksize<<endl;
		//cout << Roffset <<" " << Rsize<<endl;


		int log_FFTsize = (int)log2(FFT_SIZE);
		//cout << log_FFTsize<<endl;


		//set kernel (bitReverse and XZ transpose)
		cl::Kernel kernel_BRXZtrans(program, "bitReverseAndXZ_transpose");
		kernel_BRXZtrans.setArg(0, FTchi);
		kernel_BRXZtrans.setArg(1, chiq_dum);
		kernel_BRXZtrans.setArg(2, (cl_int)log_FFTsize);
		kernel_BRXZtrans.setArg(3, (cl_int)0);
		kernel_BRXZtrans.setArg(4, (cl_int)imageSizeX);
		kernel_BRXZtrans.setArg(5, (cl_int)Roffset);
		kernel_BRXZtrans.setArg(6, (cl_int)FFT_SIZE);
		kernel_BRXZtrans.setArg(7, (cl_int)offsetY);
		kernel_BRXZtrans.setArg(8, (cl_int)imageSizeY);
		kernel_BRXZtrans.setArg(9, (cl_int)0);
		kernel_BRXZtrans.setArg(10, (cl_int)FFTimageSizeY);
		const cl::NDRange global_item_size1(imageSizeX, FFTimageSizeY, Rsize);
		const cl::NDRange local_item_size1(min(imageSizeX, (int)maxWorkGroupSize[0]), 1, 1);
		queue.enqueueNDRangeKernel(kernel_BRXZtrans, NULL, global_item_size1, local_item_size1, NULL, NULL);
		queue.finish();


		//set kernel (butterflyIMGarray)
		cl::Kernel kernel_butterfly(program, "butterfly");
		kernel_butterfly.setArg(0, chiq_dum);
		kernel_butterfly.setArg(1, w_buffer);
		kernel_butterfly.setArg(3, (cl_uint)0x80000000); //flag
		const cl::NDRange global_item_size2(FFT_SIZE / 2, FFTimageSizeY, imageSizeX);
		const cl::NDRange local_item_size2(min(FFT_SIZE / 2, (int)maxWorkGroupSize[0]), 1, 1);
		for (int iter = 1; iter <= log_FFTsize; iter++) {
			kernel_butterfly.setArg(2, iter);
			queue.enqueueNDRangeKernel(kernel_butterfly, NULL, global_item_size2, local_item_size2, NULL, NULL);
			queue.finish();
		}


		//set kernel (FFTnormMGarray)
		cl::Kernel kernel_norm(program, "IFFTnorm");
		kernel_norm.setArg(0, chiq_dum);
		kernel_norm.setArg(1, (cl_float)RGRID);
		const cl::NDRange global_item_size3(FFT_SIZE, FFTimageSizeY, imageSizeX);
		const cl::NDRange local_item_size3(min(FFT_SIZE, (int)maxWorkGroupSize[0]), 1, 1);
		queue.enqueueNDRangeKernel(kernel_norm, NULL, global_item_size3, local_item_size3, NULL, NULL);
		queue.finish();


		//set kernel (XZ transpose)
		cl::Kernel kernel_XZtrans(program, "XZ_transpose");
		kernel_XZtrans.setArg(0, chiq_dum);
		kernel_XZtrans.setArg(1, chiq);
		kernel_XZtrans.setArg(2, (cl_int)qoffset);
		kernel_XZtrans.setArg(3, (cl_int)FFT_SIZE);
		kernel_XZtrans.setArg(4, (cl_int)0);
		kernel_XZtrans.setArg(5, (cl_int)imageSizeX);
		kernel_XZtrans.setArg(6, (cl_int)0);
		kernel_XZtrans.setArg(7, (cl_int)FFTimageSizeY);
		kernel_XZtrans.setArg(8, (cl_int)offsetY);
		kernel_XZtrans.setArg(9, (cl_int)imageSizeY);
		const cl::NDRange global_item_size4(qsize, FFTimageSizeY, imageSizeX);
		const cl::NDRange local_item_size4(1, 1, min(imageSizeX, (int)maxWorkGroupSize[2]));
		queue.enqueueNDRangeKernel(kernel_XZtrans, NULL, global_item_size4, local_item_size4, NULL, NULL);
		queue.finish();
	
	}catch (cl::Error ret) {
		cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(ret.err());
	}

    return 0;
}



int EXAFS_kFit(cl::CommandQueue queue, cl::Program program,
               cl::Buffer chidata, cl::Buffer S02, cl::Buffer Rfactor, vector<shellObjects> shObj,
               int kw, float kstart, float kend, int imageSizeX, int imageSizeY,
               bool freeS02,int numTrial,float lambda, int contrainSize, cl::Buffer edgeJ,
               cl::Buffer C_matrix_buff, cl::Buffer D_vector_buff, cl::Buffer C2_vector_buff,
               bool CSbool, int CSit, cl::Buffer CSlambda_buff){
    try {
        
        int FreeParaSize = (freeS02) ? 1:0;
        //cout << shObj.size()<<endl;
        for (int i=0; i<shObj.size(); i++) {
            FreeParaSize += shObj[i].getFreeParaSize();
        }
        
        int imageSizeM = imageSizeX*imageSizeY;
        int ksize = (int)ceil((min(float(kend+WIN_DK),(float)MAX_KQ)-max(float(kstart-WIN_DK),0.0f))/KGRID)+1;
        int koffset = (int)floor(max((float)(kstart-WIN_DK),0.0f)/KGRID);
        
        //OCL buffers
        cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Buffer chiFit(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*ksize,0,NULL);
        cl::Buffer dF2_old(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
        cl::Buffer dF2_new(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
        cl::Buffer tJJ(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*FreeParaSize*(FreeParaSize+1)/2,0,NULL);
        cl::Buffer tJdF(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*FreeParaSize,0,NULL);
        vector<cl::Buffer> Jacobian;
        cl::Buffer para_backup(cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*FreeParaSize,0,NULL));
        for(int i=0;i<FreeParaSize;i++){
            Jacobian.push_back(cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*ksize,0,NULL));
            
        }
        cl::Buffer nyu_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
        cl::Buffer lambda_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
        cl::Buffer dp_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM*FreeParaSize, 0, NULL);
        cl::Buffer dL_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
        cl::Buffer rho_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
        cl::Buffer p_fix(context,CL_MEM_READ_WRITE,sizeof(cl_char)*FreeParaSize,0,NULL);
        queue.enqueueFillBuffer(lambda_buff, (cl_float)lambda, 0, sizeof(float)*imageSizeM);
        queue.enqueueFillBuffer(nyu_buff, (cl_float)2.0f, 0, sizeof(float)*imageSizeM);
        queue.enqueueFillBuffer(p_fix, (cl_char)1, 0, sizeof(cl_char)*FreeParaSize);
        cl::Buffer eval_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM,0,NULL);
        cl::Buffer freefix_fista(context, CL_MEM_READ_WRITE, sizeof(cl_char)*FreeParaSize, 0, NULL);
        queue.enqueueFillBuffer(freefix_fista, (cl_char)49, 0, sizeof(cl_char)*FreeParaSize);
        
        
        
        //kernel settings
        cl::Kernel kernel_kwindow(program,"hanningWindowFuncIMGarray");
        kernel_kwindow.setArg(1, (cl_float)kstart);
        kernel_kwindow.setArg(2, (cl_float)kend);
        kernel_kwindow.setArg(3, (cl_float)WIN_DK);
        kernel_kwindow.setArg(4, (cl_float)KGRID);
        cl::Kernel kernel_tJJ(program,"estimate_tJJ");
        kernel_tJJ.setArg(0, tJJ);
        kernel_tJJ.setArg(4, (cl_int)ksize);
        cl::Kernel kernel_tJdF(program,"estimate_tJdF");
        kernel_tJdF.setArg(0, tJdF);
        kernel_tJdF.setArg(2, chidata);
        kernel_tJdF.setArg(3, chiFit);
        kernel_tJdF.setArg(5, (cl_int)ksize);
        cl::Kernel kernel_dF2(program,"estimate_dF2");
        kernel_dF2.setArg(1, chidata);
        kernel_dF2.setArg(2, chiFit);
        kernel_dF2.setArg(3, (cl_int)ksize);
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
        cl::Kernel kernel_UR(program,"updateOrRestore");
        kernel_UR.setArg(0, S02);
        kernel_UR.setArg(1, para_backup);
        kernel_UR.setArg(2, rho_buff);
        kernel_UR.setArg(3, (cl_int)0);
        kernel_UR.setArg(4, (cl_int)0);
        cl::Kernel kernel_contrain1(program,"contrain_1");
        cl::Kernel kernel_contrain2(program,"contrain_2");
        kernel_contrain1.setArg(0, S02);
        kernel_contrain1.setArg(1, eval_img);
        kernel_contrain1.setArg(2, C_matrix_buff);
        kernel_contrain2.setArg(0, S02);
        kernel_contrain2.setArg(1, edgeJ);
        kernel_contrain2.setArg(2, eval_img);
        kernel_contrain2.setArg(3, C_matrix_buff);
        kernel_contrain2.setArg(4, D_vector_buff);
        kernel_contrain2.setArg(5, C2_vector_buff);
        kernel_contrain2.setArg(8, (cl_char)48);
        cl::Kernel kernel_Rfactor(program,"estimate_Rfactor");
        kernel_Rfactor.setArg(0, Rfactor);
        kernel_Rfactor.setArg(1, chidata);
        kernel_Rfactor.setArg(2, chiFit);
        kernel_Rfactor.setArg(3, (cl_int)ksize);
        //FISTA
        cl::Kernel kernel_FISTA(program,"FISTA");
        if (CSbool) {
            kernel_FISTA.setArg(0, para_backup);
            kernel_FISTA.setArg(1, dp_img);
            kernel_FISTA.setArg(2, tJJ);
            kernel_FISTA.setArg(3, tJdF);
            kernel_FISTA.setArg(4, freefix_fista);
            kernel_FISTA.setArg(5, lambda_buff);
            kernel_FISTA.setArg(6, CSlambda_buff);
        }
        
        
        
        size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        const cl::NDRange local_item_size(min((int)maxWorkGroupSize,imageSizeX),1,1);
        const cl::NDRange global_item_size(imageSizeX,imageSizeY,1);
        const cl::NDRange global_item_size_stack(imageSizeX,imageSizeY,ksize);
        const cl::NDRange global_item_offset(0,0,koffset);
        
        //estimate chifit, jacobian , dF2
        cl_float2 iniChi={0.0f,0.0f};
        for (int trial=0; trial<numTrial; trial++) {
            //cout<<"trial "<< trial+1<<endl;
            //rest dF2, tJJ, tJdF
            queue.enqueueFillBuffer(dF2_old, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM);
            queue.enqueueFillBuffer(dF2_new, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM);
            queue.enqueueFillBuffer(tJJ, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM*FreeParaSize*(FreeParaSize+1)/2);
            queue.enqueueFillBuffer(tJdF, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM*FreeParaSize);
            
            
            //chiFit
            //reset chiFit
            queue.enqueueFillBuffer(chiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
            //estimate chiFit
            for (int sh=0; sh<shObj.size(); sh++) {
                shObj[sh].outputChiFit(chiFit,S02,kw,kstart,kend);
            }
            //weighting by window function
            kernel_kwindow.setArg(0, chiFit);
            queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset,global_item_size_stack,local_item_size, NULL, NULL);
            queue.finish();
            /*cl_float2* chi_data;
            chi_data = new cl_float2[MAX_KRSIZE];
            for (int j=0; j<MAX_KRSIZE; j++) {
                chi_data[j].x = 0.0f;
                chi_data[j].y = 0.0f;
            }
            queue.enqueueReadBuffer(chiFit, CL_TRUE, 0, sizeof(cl_float2)*ksize, &chi_data[koffset]);
            for (int j=0; j<MAX_KRSIZE; j++) {
                cout << chi_data[j].y <<endl;
            }
            delete [] chi_data;*/
            
            
            //Jacobian
            //reset Jacobian
            for(int i=0;i<FreeParaSize;i++){
                queue.enqueueFillBuffer(Jacobian[i], iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
            }
            //estimate jacobian
            int fpn = (freeS02) ? 1:0;
            for (int sh=0; sh<shObj.size(); sh++) {
                //jacobian for S02
                if(freeS02){
                    shObj[sh].outputJacobiank(Jacobian[0], S02, kw, 0, kstart, kend, false);
                }
                
                //Jacobian
                for (int paraN=0; paraN<7; paraN++) {
                    if(shObj[sh].getFreeFixPara(paraN+1)){
                        shObj[sh].outputJacobiank(Jacobian[fpn], S02, kw, paraN+1, kstart, kend,false);
                        fpn++;
                    }
                }
            }
            //weighting by window function
            for(int fpn=0; fpn<FreeParaSize; fpn++){
                kernel_kwindow.setArg(0, Jacobian[fpn]);
                queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset,global_item_size_stack, local_item_size, NULL, NULL);
                queue.finish();
            }
            /*cl_float2* J_data;
            J_data = new cl_float2[MAX_KRSIZE];
            for (int i=0; i<MAX_KRSIZE  ; i++) {
                J_data[i].x =(cl_float)0.0f;
                J_data[i].y =(cl_float)0.0f;
            }
            queue.enqueueReadBuffer(Jacobian[3], CL_TRUE, 0, sizeof(cl_float2)*ksize, &J_data[koffset]);
            for (int i=0; i<MAX_KRSIZE  ; i++) {
                cout << J_data[i].x <<"\t" << J_data[i].y <<endl;
            }*/
            
            
            
            //estimate tJJ and tJdF
            int pn=0;
            for(int fpn1=0; fpn1<FreeParaSize;fpn1++){
                kernel_tJJ.setArg(1, Jacobian[fpn1]);
                kernel_tJdF.setArg(1, Jacobian[fpn1]);
                kernel_tJdF.setArg(4, (cl_int)fpn1);
                queue.enqueueNDRangeKernel(kernel_tJdF,NULL,global_item_size,local_item_size, NULL, NULL);
                queue.finish();
                for(int fpn2=fpn1;fpn2<FreeParaSize;fpn2++){
                    kernel_tJJ.setArg(2, Jacobian[fpn2]);
                    kernel_tJJ.setArg(3, (cl_int)pn);
                    queue.enqueueNDRangeKernel(kernel_tJJ,NULL,global_item_size, local_item_size, NULL, NULL);
                    queue.finish();
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
            /*float* dp_data;
            dp_data = new float[imageSizeM*FreeParaSize];
            queue.enqueueReadBuffer(dp_img, CL_TRUE, 0, sizeof(float)*imageSizeM*FreeParaSize, dp_data);
            for (int i=0; i<FreeParaSize; i++) {
                cout <<"dp["<<i<<"]: "<<dp_data[i*imageSizeM]<<endl;
            }
            cout<<endl;*/
            
            
            //FISTA (Soft Thresholding Function)
            if(CSbool){
                for (int k = 0; k < CSit; k++) {
                    queue.enqueueNDRangeKernel(kernel_FISTA, NULL, global_item_size, local_item_size, NULL, NULL);
                    queue.finish();
                }
            }
            
            
            //estimate dL
            queue.enqueueNDRangeKernel(kernel_dL,NULL,global_item_size,local_item_size, NULL, NULL);
            queue.finish();
            
            
            //update paramater as candidate
            //S02
            fpn = 0;
            if(freeS02){
                queue.enqueueCopyBuffer(S02, para_backup, 0, sizeof(float)*imageSizeM*fpn, sizeof(float)*imageSizeM);
                
                kernel_update.setArg(1, S02);
                kernel_update.setArg(2, (cl_int)0);
                queue.enqueueNDRangeKernel(kernel_update,NULL,global_item_size,local_item_size,NULL,NULL);
                queue.finish();
                fpn++;
            }
            //others
            for (int sh=0; sh<shObj.size(); sh++) {
                for (int paraN=0; paraN<7; paraN++) {
                    if(shObj[sh].getFreeFixPara(paraN+1)){
                        shObj[sh].copyPara(para_backup, fpn, paraN+1);
                        shObj[sh].updatePara(dp_img, paraN+1, fpn);
                        fpn++;
                    }
                }
            }
            
            
            
            //contrain
            for (int cn=0; cn<contrainSize; cn++) {
                kernel_contrain1.setArg(3, (cl_int)cn);
                kernel_contrain2.setArg(6, (cl_int)cn);
                queue.enqueueFillBuffer(eval_img, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM);
                //contrain 1
                fpn = 0;
                //S02
                if(freeS02){
                    kernel_contrain1.setArg(4, (cl_int)fpn);
                    queue.enqueueNDRangeKernel(kernel_contrain1,NULL,global_item_size,local_item_size,NULL,NULL);
                    queue.finish();
                    fpn++;
                }
                //others
                for (int sh=0; sh<shObj.size(); sh++) {
                    for (int paraN=0; paraN<7; paraN++) {
                        if(shObj[sh].getFreeFixPara(paraN+1)){
                            shObj[sh].constrain1(eval_img, C_matrix_buff, cn, fpn, paraN+1);
                            fpn++;
                        }
                    }
                }
                //contrain 2
                fpn = 0;
                //S02
                if(freeS02){
                    kernel_contrain2.setArg(7, (cl_int)fpn);
                    queue.enqueueNDRangeKernel(kernel_contrain2,NULL,global_item_size,local_item_size,NULL,NULL);
                    queue.finish();
                    fpn++;
                }
                //others
                for (int sh=0; sh<shObj.size(); sh++) {
                    for (int paraN=0; paraN<7; paraN++) {
                        if(shObj[sh].getFreeFixPara(paraN+1)){
                            shObj[sh].constrain2(eval_img, edgeJ, C_matrix_buff, D_vector_buff, C2_vector_buff, cn, fpn, paraN+1);
                            fpn++;
                        }
                    }
                }
            }
            
            
            
            
            //estimate new dF2
            //reset chiFit
            queue.enqueueFillBuffer(chiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
            //estimate chifit
            for (int sh=0; sh<shObj.size(); sh++) {
                shObj[sh].outputChiFit(chiFit,S02,kw,kstart,kend);
            }
            //weighting by window function
            kernel_kwindow.setArg(0, chiFit);
            queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset,global_item_size_stack, local_item_size, NULL, NULL);
            queue.finish();
            //estimate dF2
            kernel_dF2.setArg(0, dF2_new);
            queue.enqueueNDRangeKernel(kernel_dF2,NULL,global_item_size_stack, local_item_size, NULL, NULL);
            
            
            
            //evaluate updated parameter (hold new para candidate or restore to old para )
            //S02
            fpn = 0;
            if(freeS02){
                queue.enqueueNDRangeKernel(kernel_UR,NULL,global_item_size, local_item_size, NULL, NULL);
                queue.finish();
                fpn++;
            }
            //others
            for (int sh=0; sh<shObj.size(); sh++) {
                for (int paraN=0; paraN<7; paraN++) {
                    if(shObj[sh].getFreeFixPara(paraN+1)){
                        shObj[sh].restorePara(para_backup, fpn, rho_buff, paraN+1);
                        fpn++;
                    }
                }
            }
        }
        
        //estimate Rfactor
        queue.enqueueNDRangeKernel(kernel_Rfactor,NULL,global_item_size, local_item_size, NULL, NULL);
        queue.finish();
        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(ret.err());
    }
    
    return 0;
}


int EXAFS_RFit(cl::CommandQueue queue, cl::Program program, cl::Buffer w_factor,
               cl::Buffer FTchidata, cl::Buffer S02,  cl::Buffer Rfactor, vector<shellObjects> shObj,
               int kw, float kstart, float kend, float Rstart, float Rend,
               int imageSizeX, int imageSizeY, int FFTimageSizeY,
               bool freeS02,int numTrial,float lambda, int contrainSize, cl::Buffer edgeJ,
               cl::Buffer C_matrix_buff, cl::Buffer D_vector_buff, cl::Buffer C2_vector_buff,
               bool CSbool, int CSit, cl::Buffer CSlambda_buff){
    try {
        
        int FreeParaSize = (freeS02) ? 1:0;
        for (int i=0; i<shObj.size(); i++) {
            FreeParaSize += shObj[i].getFreeParaSize();
        }
        
        int imageSizeM = imageSizeX*imageSizeY;
        int ksize = (int)ceil((min(float(kend+WIN_DK),(float)MAX_KQ)-max(float(kstart-WIN_DK),0.0f))/KGRID)+1;
        int koffset = (int)floor(max((float)(kstart-WIN_DK),0.0f)/KGRID);
        int Rsize = (int)ceil((min(float(Rend+WIN_DR),(float)MAX_R)-max(float(Rstart-WIN_DR),0.0f))/RGRID)+1;
        int Roffset = (int)floor(max((float)(Rstart-WIN_DR),0.0f)/RGRID);
        
        
        //OCL buffers
        cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Buffer chiFit(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*ksize,0,NULL);
        cl::Buffer FTchiFit(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*Rsize,0,NULL);
        cl::Buffer dF2_old(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
        cl::Buffer dF2_new(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
        cl::Buffer tJJ(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*FreeParaSize*(FreeParaSize+1)/2,0,NULL);
        cl::Buffer tJdF(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*FreeParaSize,0,NULL);
        vector<cl::Buffer> Jacobian;
        cl::Buffer para_backup(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*FreeParaSize,0,NULL);
        for(int i=0;i<FreeParaSize;i++){
            Jacobian.push_back(cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*Rsize,0,NULL));
        }
        cl::Buffer J_dummy(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*ksize,0,NULL);
        cl::Buffer nyu_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
        cl::Buffer lambda_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
        cl::Buffer dp_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM*FreeParaSize, 0, NULL);
        cl::Buffer dL_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
        cl::Buffer rho_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
        cl::Buffer p_fix(context,CL_MEM_READ_WRITE,sizeof(cl_char)*FreeParaSize,0,NULL);
        queue.enqueueFillBuffer(p_fix, (cl_char)1, 0, sizeof(cl_char)*FreeParaSize);
        queue.enqueueFillBuffer(lambda_buff, (cl_float)lambda, 0, sizeof(float)*imageSizeM);
        queue.enqueueFillBuffer(nyu_buff, (cl_float)2.0f, 0, sizeof(float)*imageSizeM);
        cl::Buffer eval_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM,0,NULL);
        cl::Buffer freefix_fista(context, CL_MEM_READ_WRITE, sizeof(cl_char)*FreeParaSize, 0, NULL);
        queue.enqueueFillBuffer(freefix_fista, (cl_char)49, 0, sizeof(cl_char)*FreeParaSize);
        
        
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
        kernel_tJdF.setArg(2, FTchidata);
        kernel_tJdF.setArg(3, FTchiFit);
        kernel_tJdF.setArg(5, (cl_int)Rsize);
        cl::Kernel kernel_dF2(program,"estimate_dF2");
        kernel_dF2.setArg(1, FTchidata);
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
        cl::Kernel kernel_UR(program,"updateOrRestore");
        kernel_UR.setArg(0, S02);
        kernel_UR.setArg(1, para_backup);
        kernel_UR.setArg(2, rho_buff);
        kernel_UR.setArg(3, (cl_int)0);
        kernel_UR.setArg(4, (cl_int)0);
        cl::Kernel kernel_contrain1(program,"contrain_1");
        cl::Kernel kernel_contrain2(program,"contrain_2");
        kernel_contrain1.setArg(0, S02);
        kernel_contrain1.setArg(1, eval_img);
        kernel_contrain1.setArg(2, C_matrix_buff);
        kernel_contrain2.setArg(0, S02);
        kernel_contrain2.setArg(1, edgeJ);
        kernel_contrain2.setArg(2, eval_img);
        kernel_contrain2.setArg(3, C_matrix_buff);
        kernel_contrain2.setArg(4, D_vector_buff);
        kernel_contrain2.setArg(5, C2_vector_buff);
        kernel_contrain2.setArg(8, (cl_char)48);
        cl::Kernel kernel_Rfactor(program,"estimate_Rfactor");
        kernel_Rfactor.setArg(0, Rfactor);
        kernel_Rfactor.setArg(1, FTchidata);
        kernel_Rfactor.setArg(2, FTchiFit);
        kernel_Rfactor.setArg(3, (cl_int)Rsize);
        //FISTA
        cl::Kernel kernel_FISTA(program,"FISTA");
        if (CSbool) {
            kernel_FISTA.setArg(0, para_backup);
            kernel_FISTA.setArg(1, dp_img);
            kernel_FISTA.setArg(2, tJJ);
            kernel_FISTA.setArg(3, tJdF);
            kernel_FISTA.setArg(4, freefix_fista);
            kernel_FISTA.setArg(5, lambda_buff);
            kernel_FISTA.setArg(6, CSlambda_buff);
        }
        
        
        size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        const cl::NDRange local_item_size(min((int)maxWorkGroupSize,imageSizeX),1,1);
        const cl::NDRange global_item_size(imageSizeX,imageSizeY,1);
        const cl::NDRange global_item_size_stack(imageSizeX,imageSizeY,ksize);
        const cl::NDRange global_item_offset(0,0,koffset);
        
        //estimate chifit, jacobian , dF2
        cl_float2 iniChi={0.0f,0.0f};
        for (int trial=0; trial<numTrial; trial++) {
            //reset dF2, tJJ, tJdF
            queue.enqueueFillBuffer(dF2_old, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM);
            queue.enqueueFillBuffer(dF2_new, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM);
            queue.enqueueFillBuffer(tJJ, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM*FreeParaSize*(FreeParaSize+1)/2);
            queue.enqueueFillBuffer(tJdF, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM*FreeParaSize);
            
            
            //chiFit
            //reset chiFit
            queue.enqueueFillBuffer(chiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
            queue.enqueueFillBuffer(FTchiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*Rsize);
            //estimate chiFit
            for (int sh=0; sh<shObj.size(); sh++) {
                shObj[sh].outputChiFit(chiFit,S02,kw,kstart,kend);
            }
            //weighting by window function
            kernel_kwindow.setArg(0, chiFit);
            queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset,global_item_size_stack,local_item_size, NULL, NULL);
            queue.finish();
            //FFT
            for (int offsetY=0; offsetY<imageSizeY; offsetY += FFTimageSizeY) {
                FFT(chiFit, FTchiFit, w_factor, queue, program,
                    imageSizeX, imageSizeY, FFTimageSizeY, offsetY,
                    koffset,ksize,Roffset,Rsize);
            }
            
            
            //Jacobian
            //reset Jacobian
            for(int i=0;i<FreeParaSize;i++){
                queue.enqueueFillBuffer(Jacobian[i], iniChi, 0, sizeof(cl_float2)*imageSizeM*Rsize);
            }
            //estimate jacobian
            int fpn = (freeS02) ? 1:0;
            //jacobian for S02
            if(freeS02){
                //estimate Jacobian in k-space
                queue.enqueueFillBuffer(J_dummy, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
                for (int sh=0; sh<shObj.size(); sh++) {
                    shObj[sh].outputJacobiank(J_dummy, S02, kw, 0, kstart, kend, false);
                }
                
                //weighting by window function
                kernel_kwindow.setArg(0, J_dummy);
                queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset,global_item_size_stack, local_item_size, NULL, NULL);
                queue.finish();
                
                //FFT
                for (int offsetY=0; offsetY<imageSizeY; offsetY += FFTimageSizeY) {
                    FFT(J_dummy, Jacobian[0], w_factor, queue, program,
                        imageSizeX, imageSizeY, FFTimageSizeY, offsetY,
                        koffset,ksize,Roffset,Rsize);
                }
            }
            for (int sh=0; sh<shObj.size(); sh++) {
                //Jacobian of other parameters
                for (int paraN=0; paraN<7; paraN++) {
                    if(shObj[sh].getFreeFixPara(paraN+1)){
                        //estimate Jacobian in k-space
                        queue.enqueueFillBuffer(J_dummy, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
                        shObj[sh].outputJacobiank(J_dummy, S02, kw, paraN+1, kstart, kend, false);
                        
                        //weighting by window function
                        kernel_kwindow.setArg(0, J_dummy);
                        queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset,global_item_size_stack, local_item_size, NULL, NULL);
                        queue.finish();
                        
                        //FFT
                        for (int offsetY=0; offsetY<imageSizeY; offsetY += FFTimageSizeY) {
                            FFT(J_dummy, Jacobian[fpn], w_factor, queue, program,
                                imageSizeX, imageSizeY, FFTimageSizeY, offsetY,
                                koffset,ksize,Roffset,Rsize);
                        }
                        fpn++;
                    }
                }
            }
            
            
            
            //estimate tJJ and tJdF
            int pn=0;
            for(int fpn1=0; fpn1<FreeParaSize;fpn1++){
                kernel_tJJ.setArg(1, Jacobian[fpn1]);
                kernel_tJdF.setArg(1, Jacobian[fpn1]);
                kernel_tJdF.setArg(4, (cl_int)fpn1);
                queue.enqueueNDRangeKernel(kernel_tJdF,NULL,global_item_size,local_item_size, NULL, NULL);
                for(int fpn2=fpn1;fpn2<FreeParaSize;fpn2++){
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
            
            
            //copy para backup
            //S02
            fpn = 0;
            if(freeS02){
                queue.enqueueCopyBuffer(S02, para_backup, 0, sizeof(cl_float)*imageSizeM*fpn, sizeof(float)*imageSizeM);
                fpn++;
            }
            //others
            for (int sh=0; sh<shObj.size(); sh++) {
                for (int paraN=0; paraN<7; paraN++) {
                    if(shObj[sh].getFreeFixPara(paraN+1)){
                        shObj[sh].copyPara(para_backup,fpn,paraN+1);
                        fpn++;
                    }
                }
            }
            
            
            //FISTA (Soft Thresholding Function)
            if(CSbool){
                for (int k = 0; k < CSit; k++) {
                    queue.enqueueNDRangeKernel(kernel_FISTA, NULL, global_item_size, local_item_size, NULL, NULL);
                    queue.finish();
                }
            }
            
            
            //estimate dL
            queue.enqueueNDRangeKernel(kernel_dL,NULL,global_item_size,local_item_size, NULL, NULL);
            queue.finish();
            
            
            //update paramater as candidate
            //S02
            fpn = 0;
            if(freeS02){
                //queue.enqueueCopyBuffer(S02, para_backup, 0, sizeof(cl_float)*imageSizeM*fpn, sizeof(float)*imageSizeM);
                
                kernel_update.setArg(1, S02);
                kernel_update.setArg(2, (cl_int)0);
                queue.enqueueNDRangeKernel(kernel_update,NULL,global_item_size,local_item_size,NULL,NULL);
                queue.finish();
                fpn++;
            }
            //others
            for (int sh=0; sh<shObj.size(); sh++) {
                for (int paraN=0; paraN<7; paraN++) {
                    if(shObj[sh].getFreeFixPara(paraN+1)){
                        //shObj[sh].copyPara(para_backup,fpn,paraN+1);
                        shObj[sh].updatePara(dp_img, paraN+1, fpn);
                        fpn++;
                    }
                }
            }
            
            
            //contrain
            for (int cn=0; cn<contrainSize; cn++) {
                kernel_contrain1.setArg(3, (cl_int)cn);
                kernel_contrain2.setArg(6, (cl_int)cn);
                queue.enqueueFillBuffer(eval_img, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM);
                //contrain 1
                fpn = 0;
                //S02
                if(freeS02){
                    kernel_contrain1.setArg(4, (cl_int)fpn);
                    queue.enqueueNDRangeKernel(kernel_contrain1,NULL,global_item_size,local_item_size,NULL,NULL);
                    queue.finish();
                    fpn++;
                }
                //others
                for (int sh=0; sh<shObj.size(); sh++) {
                    for (int paraN=0; paraN<7; paraN++) {
                        if(shObj[sh].getFreeFixPara(paraN+1)){
                            shObj[sh].constrain1(eval_img, C_matrix_buff, cn, fpn, paraN+1);
                            fpn++;
                        }
                    }
                }
                //contrain 2
                fpn = 0;
                //S02
                if(freeS02){
                    kernel_contrain2.setArg(7, (cl_int)fpn);
                    queue.enqueueNDRangeKernel(kernel_contrain2,NULL,global_item_size,local_item_size,NULL,NULL);
                    queue.finish();
                    fpn++;
                }
                //others
                for (int sh=0; sh<shObj.size(); sh++) {
                    for (int paraN=0; paraN<7; paraN++) {
                        if(shObj[sh].getFreeFixPara(paraN+1)){
                            shObj[sh].constrain2(eval_img, edgeJ, C_matrix_buff, D_vector_buff, C2_vector_buff, cn, fpn, paraN+1);
                            fpn++;
                        }
                    }
                }
            }
            
            
            //estimate new dF2
            //reset chiFit
            queue.enqueueFillBuffer(chiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
            queue.enqueueFillBuffer(FTchiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*Rsize);
            //estimate chiFit
            for (int sh=0; sh<shObj.size(); sh++) {
                shObj[sh].outputChiFit(chiFit,S02,kw,kstart,kend);
            }
            //weighting by window function
            kernel_kwindow.setArg(0, chiFit);
            queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset,global_item_size_stack,local_item_size, NULL, NULL);
            queue.finish();
            //FFT
            for (int offsetY=0; offsetY<imageSizeY; offsetY += FFTimageSizeY) {
                FFT(chiFit, FTchiFit, w_factor, queue, program,
                    imageSizeX, imageSizeY, FFTimageSizeY, offsetY,
                    koffset,ksize,Roffset,Rsize);
            }
            //estimate dF2
            kernel_dF2.setArg(0, dF2_new);
            queue.enqueueNDRangeKernel(kernel_dF2,NULL,global_item_size_stack, local_item_size, NULL, NULL);
            
            
            //evaluate updated parameter (hold new para candidate or restore to old para )
            //S02
            fpn = 0;
            if(freeS02){
                queue.enqueueNDRangeKernel(kernel_UR,NULL,global_item_size, local_item_size, NULL, NULL);
                queue.finish();
                fpn++;
            }
            //others
            for (int sh=0; sh<shObj.size(); sh++) {
                for (int paraN=0; paraN<7; paraN++) {
                    if(shObj[sh].getFreeFixPara(paraN+1)){
                        shObj[sh].restorePara(para_backup,fpn, rho_buff, paraN+1);
                        fpn++;
                    }
                }
            }
        }
        
        //estimate Rfactor
        queue.enqueueNDRangeKernel(kernel_Rfactor,NULL,global_item_size, local_item_size, NULL, NULL);
        queue.finish();
        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(ret.err());
    }
    
    return 0;
}


int EXAFS_qFit(cl::CommandQueue queue, cl::Program program, cl::Buffer w_factor,
               cl::Buffer chiqdata, cl::Buffer S02, cl::Buffer Rfactor, vector<shellObjects> shObj,
               int kw, float kstart, float kend, float Rstart, float Rend, float qstart, float qend,
               int imageSizeX, int imageSizeY, int FFTimageSizeY,
               bool freeS02,int numTrial,float lambda, int contrainSize, cl::Buffer edgeJ,
               cl::Buffer C_matrix_buff, cl::Buffer D_vector_buff, cl::Buffer C2_vector_buff,
               bool CSbool, int CSit, cl::Buffer CSlambda_buff){
    try {
        
        int FreeParaSize = (freeS02) ? 1:0;
        for (int i=0; i<shObj.size(); i++) {
            FreeParaSize += shObj[i].getFreeParaSize();
        }
        
        int imageSizeM = imageSizeX*imageSizeY;
        int ksize = (int)ceil((min(float(kend+WIN_DK),(float)MAX_KQ)-max(float(kstart-WIN_DK),0.0f))/KGRID)+1;
        int koffset = (int)floor(max((float)(kstart-WIN_DK),0.0f)/KGRID);
        int Rsize = (int)ceil((min(float(Rend+WIN_DR),(float)MAX_R)-max(float(Rstart-WIN_DR),0.0f))/RGRID)+1;
        int Roffset = (int)floor(max((float)(Rstart-WIN_DR),0.0f)/RGRID);
        int qsize = (int)ceil((min(float(qend+WIN_DK),(float)MAX_KQ)-max(float(qstart-WIN_DK),0.0f))/KGRID)+1;
        int qoffset = (int)floor(max((float)(qstart-WIN_DK),0.0f)/KGRID);
        
        //OCL buffers
        cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Buffer chiFit(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*ksize,0,NULL);
        cl::Buffer FTchiFit(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*Rsize,0,NULL);
        cl::Buffer chiqFit(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*qsize,0,NULL);
        cl::Buffer dF2_old(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
        cl::Buffer dF2_new(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
        cl::Buffer tJJ(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*FreeParaSize*(FreeParaSize+1)/2,0,NULL);
        cl::Buffer tJdF(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*FreeParaSize,0,NULL);
        vector<cl::Buffer> Jacobian;
        cl::Buffer para_backup(cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*FreeParaSize,0,NULL));
        for(int i=0;i<FreeParaSize;i++){
            Jacobian.push_back(cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*qsize,0,NULL));
        }
        cl::Buffer J_dummy_k(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*ksize,0,NULL);
        cl::Buffer J_dummy_R(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*Rsize,0,NULL);
        cl::Buffer nyu_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
        cl::Buffer lambda_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
        cl::Buffer dp_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM*FreeParaSize, 0, NULL);
        cl::Buffer dL_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
        cl::Buffer rho_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
        cl::Buffer p_fix(context,CL_MEM_READ_WRITE,sizeof(cl_char)*FreeParaSize,0,NULL);
        queue.enqueueFillBuffer(p_fix, (cl_char)1, 0, sizeof(cl_char)*FreeParaSize);
        queue.enqueueFillBuffer(lambda_buff, (cl_float)lambda, 0, sizeof(float)*imageSizeM);
        queue.enqueueFillBuffer(nyu_buff, (cl_float)2.0f, 0, sizeof(float)*imageSizeM);
        cl::Buffer eval_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM,0,NULL);
        cl::Buffer freefix_fista(context, CL_MEM_READ_WRITE, sizeof(cl_char)*FreeParaSize, 0, NULL);
        queue.enqueueFillBuffer(freefix_fista, (cl_char)49, 0, sizeof(cl_char)*FreeParaSize);
        
        
        //kernel settings
        cl::Kernel kernel_kwindow(program,"hanningWindowFuncIMGarray");
        kernel_kwindow.setArg(1, (cl_float)kstart);
        kernel_kwindow.setArg(2, (cl_float)kend);
        kernel_kwindow.setArg(3, (cl_float)WIN_DK);
        kernel_kwindow.setArg(4, (cl_float)KGRID);
        cl::Kernel kernel_tJJ(program,"estimate_tJJ");
        kernel_tJJ.setArg(0, tJJ);
        kernel_tJJ.setArg(4, (cl_int)qsize);
        cl::Kernel kernel_tJdF(program,"estimate_tJdF");
        kernel_tJdF.setArg(0, tJdF);
        kernel_tJdF.setArg(2, chiqdata);
        kernel_tJdF.setArg(3, chiqFit);
        kernel_tJdF.setArg(5, (cl_int)qsize);
        cl::Kernel kernel_dF2(program,"estimate_dF2");
        kernel_dF2.setArg(1, chiqdata);
        kernel_dF2.setArg(2, chiqFit);
        kernel_dF2.setArg(3, (cl_int)qsize);
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
        cl::Kernel kernel_UR(program,"updateOrRestore");
        kernel_UR.setArg(0, S02);
        kernel_UR.setArg(1, para_backup);
        kernel_UR.setArg(2, rho_buff);
        kernel_UR.setArg(3, (cl_int)0);
        kernel_UR.setArg(4, (cl_int)0);
        cl::Kernel kernel_contrain1(program,"contrain_1");
        cl::Kernel kernel_contrain2(program,"contrain_2");
        kernel_contrain1.setArg(0, S02);
        kernel_contrain1.setArg(1, eval_img);
        kernel_contrain1.setArg(2, C_matrix_buff);
        kernel_contrain2.setArg(0, S02);
        kernel_contrain2.setArg(1, edgeJ);
        kernel_contrain2.setArg(2, eval_img);
        kernel_contrain2.setArg(3, C_matrix_buff);
        kernel_contrain2.setArg(4, D_vector_buff);
        kernel_contrain2.setArg(5, C2_vector_buff);
        kernel_contrain2.setArg(8, (cl_char)48);
        cl::Kernel kernel_Rfactor(program,"estimate_Rfactor");
        kernel_Rfactor.setArg(0, Rfactor);
        kernel_Rfactor.setArg(1, chiqdata);
        kernel_Rfactor.setArg(2, chiqFit);
        kernel_Rfactor.setArg(3, (cl_int)qsize);
        //FISTA
        cl::Kernel kernel_FISTA(program,"FISTA");
        if (CSbool) {
            kernel_FISTA.setArg(0, para_backup);
            kernel_FISTA.setArg(1, dp_img);
            kernel_FISTA.setArg(2, tJJ);
            kernel_FISTA.setArg(3, tJdF);
            kernel_FISTA.setArg(4, freefix_fista);
            kernel_FISTA.setArg(5, lambda_buff);
            kernel_FISTA.setArg(6, CSlambda_buff);
        }
        
        
        
        size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        const cl::NDRange local_item_size(min((int)maxWorkGroupSize,imageSizeX),1,1);
        const cl::NDRange global_item_size(imageSizeX,imageSizeY,1);
        const cl::NDRange global_item_size_stack(imageSizeX,imageSizeY,ksize);
        const cl::NDRange global_item_offset(0,0,koffset);
        
        //estimate chifit, jacobian , dF2
        cl_float2 iniChi={0.0f,0.0f};
        for (int trial=0; trial<numTrial; trial++) {
            //reset dF2, tJJ, tJdF
            queue.enqueueFillBuffer(dF2_old, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM);
            queue.enqueueFillBuffer(dF2_new, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM);
            queue.enqueueFillBuffer(tJJ, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM*FreeParaSize*(FreeParaSize+1)/2);
            queue.enqueueFillBuffer(tJdF, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM*FreeParaSize);
            
            
            //chiFit (fitもq空間)
            //reset chiFit
            queue.enqueueFillBuffer(chiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
            queue.enqueueFillBuffer(FTchiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*Rsize);
            queue.enqueueFillBuffer(chiqFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*qsize);
            //estimate chiFit
            for (int sh=0; sh<shObj.size(); sh++) {
                shObj[sh].outputChiFit(chiFit,S02,kw,kstart,kend);
            }
            //weighting by window function
            kernel_kwindow.setArg(0, chiFit);
            queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset,global_item_size_stack,local_item_size, NULL, NULL);
            queue.finish();
            //FFT & IFFT
            for (int offsetY=0; offsetY<imageSizeY; offsetY += FFTimageSizeY) {
                FFT(chiFit, FTchiFit, w_factor, queue, program,
                    imageSizeX, imageSizeY, FFTimageSizeY, offsetY,
                    koffset,ksize,Roffset,Rsize);
                IFFT(FTchiFit, chiqFit, w_factor, queue, program,
                     imageSizeX, imageSizeY, FFTimageSizeY, offsetY,
                     Roffset,Rsize,qoffset,qsize);
            }
            
            
            /*//Jacobian(fitはk空間)
            //reset Jacobian
            queue.enqueueFillBuffer(chiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM);
            for(int i=0;i<FreeParaSize;i++){
                queue.enqueueFillBuffer(Jacobian[i], iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
            }
            //estimate jacobian
            int fpn = (freeS02) ? 1:0;
            for (int sh=0; sh<shObj.size(); sh++) {
                //jacobian for S02
                if(freeS02){
                    shObj[sh].outputJacobiank(Jacobian[0], S02, kw, 0, qstart, qend, true);
                }
                
                //Jacobian
                for (int paraN=0; paraN<7; paraN++) {
                    if(shObj[sh].getFreeFixPara(paraN+1)){
                        shObj[sh].outputJacobiank(Jacobian[fpn], S02, kw, paraN+1, qstart, qend,true);
                        fpn++;
                    }
                }
            }
            //weighting by window function
            for(int fpn=0; fpn<FreeParaSize; fpn++){
                kernel_kwindow.setArg(0, Jacobian[fpn]);
                queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset,global_item_size_stack, local_item_size, NULL, NULL);
                queue.finish();
            }*/
            //Jacobian(fitもq空間)
            //reset Jacobian
            for(int i=0;i<FreeParaSize;i++){
                queue.enqueueFillBuffer(Jacobian[i], iniChi, 0, sizeof(cl_float2)*imageSizeM*qsize);
            }
            //estimate jacobian
            int fpn = (freeS02) ? 1:0;
            //jacobian for S02
            if(freeS02){
                //estimate Jacobian in k-space
                queue.enqueueFillBuffer(J_dummy_k, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
                queue.enqueueFillBuffer(J_dummy_R, iniChi, 0, sizeof(cl_float2)*imageSizeM*Rsize);
                for (int sh=0; sh<shObj.size(); sh++) {
                    shObj[sh].outputJacobiank(J_dummy_k, S02, kw, 0, kstart, kend,false);
                }
                
                //weighting by window function
                kernel_kwindow.setArg(0, J_dummy_k);
                queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset,global_item_size_stack, local_item_size, NULL, NULL);
                queue.finish();
                
                //FFT & IFFT
                for (int offsetY=0; offsetY<imageSizeY; offsetY += FFTimageSizeY) {
                    FFT(J_dummy_k, J_dummy_R, w_factor, queue, program,
                        imageSizeX, imageSizeY, FFTimageSizeY, offsetY,
                        koffset,ksize,Roffset,Rsize);
                    IFFT(J_dummy_R, Jacobian[0], w_factor, queue, program,
                         imageSizeX, imageSizeY, FFTimageSizeY, offsetY,
                         Roffset,Rsize,qoffset,qsize);
                }
            }
            for (int sh=0; sh<shObj.size(); sh++) {
                //Jacobian of other parameters
                for (int paraN=0; paraN<7; paraN++) {
                    if(shObj[sh].getFreeFixPara(paraN+1)){
                        //estimate Jacobian in k-space
                        queue.enqueueFillBuffer(J_dummy_k, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
                        queue.enqueueFillBuffer(J_dummy_R, iniChi, 0, sizeof(cl_float2)*imageSizeM*Rsize);
                        shObj[sh].outputJacobiank(J_dummy_k, S02, kw, paraN+1, kstart, kend,false);
                        
                        //weighting by window function
                        kernel_kwindow.setArg(0, J_dummy_k);
                        queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset,global_item_size_stack, local_item_size, NULL, NULL);
                        queue.finish();
                        
                        //FFT
                        for (int offsetY=0; offsetY<imageSizeY; offsetY += FFTimageSizeY) {
                            FFT(J_dummy_k, J_dummy_R, w_factor, queue, program,
                                imageSizeX, imageSizeY, FFTimageSizeY, offsetY,
                                koffset,ksize,Roffset,Rsize);
                            IFFT(J_dummy_R, Jacobian[fpn], w_factor, queue, program,
                                 imageSizeX, imageSizeY, FFTimageSizeY, offsetY,
                                 Roffset,Rsize,qoffset,qsize);
                        }
                        fpn++;
                    }
                }
            }
            
            
            
            //estimate tJJ and tJdF
            int pn=0;
            for(int fpn1=0; fpn1<FreeParaSize;fpn1++){
                kernel_tJJ.setArg(1, Jacobian[fpn1]);
                kernel_tJdF.setArg(1, Jacobian[fpn1]);
                kernel_tJdF.setArg(4, (cl_int)fpn1);
                queue.enqueueNDRangeKernel(kernel_tJdF,NULL,global_item_size,local_item_size, NULL, NULL);
                for(int fpn2=fpn1;fpn2<FreeParaSize;fpn2++){
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
            
            
            //FISTA (Soft Thresholding Function)
            if(CSbool){
                for (int k = 0; k < CSit; k++) {
                    queue.enqueueNDRangeKernel(kernel_FISTA, NULL, global_item_size, local_item_size, NULL, NULL);
                    queue.finish();
                }
            }
            
            
            //estimate dL
            queue.enqueueNDRangeKernel(kernel_dL,NULL,global_item_size,local_item_size, NULL, NULL);
            queue.finish();
            
            
            //update paramater as candidate
            //S02
            fpn = 0;
            if(freeS02){
                queue.enqueueCopyBuffer(S02, para_backup, 0, sizeof(float)*imageSizeM*fpn, sizeof(float)*imageSizeM);
                
                kernel_update.setArg(1, S02);
                kernel_update.setArg(2, (cl_int)0);
                queue.enqueueNDRangeKernel(kernel_update,NULL,global_item_size,local_item_size,NULL,NULL);
                queue.finish();
                fpn++;
            }
            //others
            for (int sh=0; sh<shObj.size(); sh++) {
                for (int paraN=0; paraN<7; paraN++) {
                    if(shObj[sh].getFreeFixPara(paraN+1)){
                        shObj[sh].copyPara(para_backup, fpn, paraN+1);
                        shObj[sh].updatePara(dp_img, paraN+1, fpn);
                        fpn++;
                    }
                }
            }
            
            
            //contrain
            for (int cn=0; cn<contrainSize; cn++) {
                kernel_contrain1.setArg(3, (cl_int)cn);
                kernel_contrain2.setArg(6, (cl_int)cn);
                queue.enqueueFillBuffer(eval_img, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM);
                //contrain 1
                fpn = 0;
                //S02
                if(freeS02){
                    kernel_contrain1.setArg(4, (cl_int)fpn);
                    queue.enqueueNDRangeKernel(kernel_contrain1,NULL,global_item_size,local_item_size,NULL,NULL);
                    queue.finish();
                    fpn++;
                }
                //others
                for (int sh=0; sh<shObj.size(); sh++) {
                    for (int paraN=0; paraN<7; paraN++) {
                        if(shObj[sh].getFreeFixPara(paraN+1)){
                            shObj[sh].constrain1(eval_img, C_matrix_buff, cn, fpn, paraN+1);
                            fpn++;
                        }
                    }
                }
                //contrain 2
                fpn = 0;
                //S02
                if(freeS02){
                    kernel_contrain2.setArg(7, (cl_int)fpn);
                    queue.enqueueNDRangeKernel(kernel_contrain2,NULL,global_item_size,local_item_size,NULL,NULL);
                    queue.finish();
                    fpn++;
                }
                //others
                for (int sh=0; sh<shObj.size(); sh++) {
                    for (int paraN=0; paraN<7; paraN++) {
                        if(shObj[sh].getFreeFixPara(paraN+1)){
                            shObj[sh].constrain2(eval_img, edgeJ, C_matrix_buff, D_vector_buff, C2_vector_buff, cn, fpn, paraN+1);
                            fpn++;
                        }
                    }
                }
            }
            
            
            //estimate new dF2　(fitもq空間)
            //reset chiFit
            queue.enqueueFillBuffer(chiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*ksize);
            queue.enqueueFillBuffer(FTchiFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*Rsize);
            queue.enqueueFillBuffer(chiqFit, iniChi, 0, sizeof(cl_float2)*imageSizeM*qsize);
            //estimate chiFit
            for (int sh=0; sh<shObj.size(); sh++) {
                shObj[sh].outputChiFit(chiFit,S02,kw,kstart,kend);
            }
            //weighting by window function
            kernel_kwindow.setArg(0, chiFit);
            queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset,global_item_size_stack,local_item_size, NULL, NULL);
            queue.finish();
            //FFT
            for (int offsetY=0; offsetY<imageSizeY; offsetY += FFTimageSizeY) {
                FFT(chiFit, FTchiFit, w_factor, queue, program,
                    imageSizeX, imageSizeY, FFTimageSizeY, offsetY,
                    koffset,ksize,Roffset,Rsize);
                IFFT(FTchiFit, chiqFit, w_factor, queue, program,
                     imageSizeX, imageSizeY, FFTimageSizeY, offsetY,
                     Roffset,Rsize,qoffset,qsize);
            }
            //estimate dF2
            kernel_dF2.setArg(0, dF2_new);
            queue.enqueueNDRangeKernel(kernel_dF2,NULL,global_item_size_stack, local_item_size, NULL, NULL);
            
            
            //evaluate updated parameter (hold new para candidate or restore to old para )
            //S02
            fpn = 0;
            if(freeS02){
                queue.enqueueNDRangeKernel(kernel_UR,NULL,global_item_size, local_item_size, NULL, NULL);
                queue.finish();
                fpn++;
            }
            //others
            for (int sh=0; sh<shObj.size(); sh++) {
                for (int paraN=0; paraN<7; paraN++) {
                    if(shObj[sh].getFreeFixPara(paraN+1)){
                        shObj[sh].restorePara(para_backup, fpn, rho_buff, paraN+1);
                        fpn++;
                    }
                }
            }
        }
        
        //estimate Rfactor
        queue.enqueueNDRangeKernel(kernel_Rfactor,NULL,global_item_size, local_item_size, NULL, NULL);
        queue.finish();
        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(ret.err());
    }
    
    return 0;
}


int ChiData_k(cl::CommandQueue queue, cl::Program program,
                  cl::Buffer chiData, vector<float*> chiData_pointer,
                  int kw, float kstart, float kend, int imageSizeX, int imageSizeY,
                  bool imgStckOrChiStck, int offsetM){
    
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
    int ksize = (int)ceil((min(float(kend+WIN_DK),(float)MAX_KQ)-max(float(kstart-WIN_DK),0.0f))/KGRID)+1;
    int koffset = (int)floor(max((float)(kstart-WIN_DK),0.0f)/KGRID);
    int imageSizeM = imageSizeX*imageSizeY;
    
    size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    
    
    //convert to complexed chi
    if (imgStckOrChiStck) {
        //image stack
        cl::Buffer chiDataDummy(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
        cl::Kernel kernel_cmplx(program,"chi2cmplxChi_imgStck");
        
        kernel_cmplx.setArg(0, chiDataDummy);
        kernel_cmplx.setArg(1, chiData);
        kernel_cmplx.setArg(3, (cl_int)koffset);
        kernel_cmplx.setArg(4, (cl_int)kw);
        const cl::NDRange global_item_size1(imageSizeX,imageSizeY,1);
        const cl::NDRange local_item_size1(min((int)maxWorkGroupSize,imageSizeX),1,1);
        for (int i=0; i<ksize; i++) {
            queue.enqueueWriteBuffer(chiDataDummy, CL_TRUE, 0, sizeof(cl_float)*imageSizeM, &chiData_pointer[i+koffset][offsetM]);
			queue.finish();
			kernel_cmplx.setArg(2, (cl_int)i);
            queue.enqueueNDRangeKernel(kernel_cmplx, NULL, global_item_size1, local_item_size1, NULL, NULL);
			queue.finish();
        }
    }else{
        //chi stack
        cl::Buffer chiDataDummy(context,CL_MEM_READ_ONLY,sizeof(cl_float)*MAX_KRSIZE,0,NULL);
        cl::Kernel kernel_cmplx(program,"chi2cmplxChi_chiStck");
        kernel_cmplx.setArg(0, chiDataDummy);
        kernel_cmplx.setArg(1, chiData);
        kernel_cmplx.setArg(3, (cl_int)kw);
        kernel_cmplx.setArg(4, (cl_int)imageSizeM);
        for (int i=0; i<min(imageSizeM,(int)chiData_pointer.size()); i++) {
            queue.enqueueFillBuffer(chiDataDummy, (cl_float)0.0f, 0, sizeof(cl_float)*MAX_KRSIZE);
            queue.enqueueWriteBuffer(chiDataDummy, CL_TRUE, 0, sizeof(cl_float)*MAX_KRSIZE, chiData_pointer[i]);
            
            kernel_cmplx.setArg(2, (cl_int)i);
            const cl::NDRange global_item_size1(ksize,1,1);
            const cl::NDRange global_item_offset1(koffset,0,0);
            const cl::NDRange local_item_size1(min((int)maxWorkGroupSize,ksize),1,1);
            queue.enqueueNDRangeKernel(kernel_cmplx, global_item_offset1, global_item_size1, local_item_size1, NULL, NULL);
            queue.finish();
        }
        
    }

    //set chi k-window
    cl::Kernel kernel_kwindow(program,"hanningWindowFuncIMGarray");
    kernel_kwindow.setArg(0, chiData);
    kernel_kwindow.setArg(1, (cl_float)kstart);
    kernel_kwindow.setArg(2, (cl_float)kend);
    kernel_kwindow.setArg(3, (cl_float)WIN_DK);
    kernel_kwindow.setArg(4, (cl_float)KGRID);
    const cl::NDRange local_item_size2(min((int)maxWorkGroupSize,imageSizeX),1,1);
    const cl::NDRange global_item_size2(imageSizeX,imageSizeY,ksize);
    const cl::NDRange global_item_offset2(0,0,koffset);
    queue.enqueueNDRangeKernel(kernel_kwindow,global_item_offset2,global_item_size2, local_item_size2, NULL, NULL);
    queue.finish();

	for (int i = 0; i<chiData_pointer.size(); i++) {
		delete[] chiData_pointer[i];
	}
    return 0;
}


int ChiData_R(cl::CommandQueue queue, cl::Program program,
              cl::Buffer FTchiData, vector<float*> chiData_pointer, cl::Buffer w_factor,
              int kw, float kstart, float kend, float Rstart, float Rend,
              int imageSizeX, int imageSizeY, int FFTimageSizeY,bool imgStckOrChiStck, int offsetM){
    
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
    int ksize = (int)ceil((min(float(kend+WIN_DK),(float)MAX_KQ)-max(float(kstart-WIN_DK),0.0f))/KGRID)+1;
    int koffset = (int)floor(max((float)(kstart-WIN_DK),0.0f)/KGRID);
    int Rsize = (int)ceil((min(float(Rend+WIN_DR),(float)MAX_R)-max(float(Rstart-WIN_DR),0.0f))/RGRID)+1;
    int Roffset = (int)floor(max((float)(Rstart-WIN_DR),0.0f)/RGRID);
    int imageSizeM = imageSizeX*imageSizeY;
    cl::Buffer chiData(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*ksize,0,NULL);
    
    size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    const cl::NDRange local_item_size(min((int)maxWorkGroupSize,imageSizeX),1,1);
    
    
    ChiData_k(queue,program,chiData,move(chiData_pointer),kw,kstart,kend,imageSizeX,imageSizeY,imgStckOrChiStck,offsetM);
    
    
    //FFT
    for (int offsetY=0; offsetY<imageSizeY; offsetY += FFTimageSizeY) {
        FFT(chiData,FTchiData,w_factor,queue,program,imageSizeX,imageSizeY,FFTimageSizeY,offsetY,koffset,ksize,Roffset,Rsize);
    }
    
    
    //set FTchi R-window
    cl::Kernel kernel_window(program,"hanningWindowFuncIMGarray");
    kernel_window.setArg(0, FTchiData);
    kernel_window.setArg(1, (cl_float)Rstart);
    kernel_window.setArg(2, (cl_float)Rend);
    kernel_window.setArg(3, (cl_float)WIN_DR);
    kernel_window.setArg(4, (cl_float)RGRID);
    const cl::NDRange global_item_size(imageSizeX,imageSizeY,Rsize);
    const cl::NDRange global_item_offset(0,0,Roffset);
    queue.enqueueNDRangeKernel(kernel_window,global_item_offset,global_item_size,local_item_size, NULL, NULL);
    queue.finish();
    

    return 0;
}


int ChiData_q(cl::CommandQueue queue, cl::Program program,
              cl::Buffer chiqData, vector<float*> chiData_pointer, cl::Buffer w_factor,
              int kw, float kstart, float kend, float Rstart, float Rend, float qstart, float qend,
              int imageSizeX, int imageSizeY, int FFTimageSizeY, bool imgStckOrChiStck, int offsetM){
    
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
    int Rsize = (int)ceil((min(float(Rend+WIN_DR),(float)MAX_R)-max(float(Rstart-WIN_DR),0.0f))/RGRID)+1;
    int Roffset = (int)floor(max((float)(Rstart-WIN_DR),0.0f)/RGRID);
    int qsize = (int)ceil((min(float(qend+WIN_DK),(float)MAX_KQ)-max(float(qstart-WIN_DK),0.0f))/KGRID)+1;
    int qoffset = (int)floor(max((float)(qstart-WIN_DK),0.0f)/KGRID);
    int imageSizeM = imageSizeX*imageSizeY;
    cl::Buffer FTchiData(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*Rsize,0,NULL);
    
    
    size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    const cl::NDRange local_item_size(min((int)maxWorkGroupSize,imageSizeX),1,1);
    
    
    ChiData_R(queue, program, FTchiData, move(chiData_pointer), w_factor, kw, kstart, kend, Rstart, Rend, imageSizeX, imageSizeY, FFTimageSizeY, imgStckOrChiStck, offsetM);
    
    
    //IFFT
    for (int offsetY=0; offsetY<imageSizeY; offsetY += FFTimageSizeY) {
        IFFT(FTchiData, chiqData, w_factor, queue, program, imageSizeX, imageSizeY, FFTimageSizeY, offsetY, Roffset,Rsize,qoffset,qsize);
    }
    
    
    //set FTchi q-window
    cl::Kernel kernel_window(program,"hanningWindowFuncIMGarray");
    kernel_window.setArg(0, chiqData);
    kernel_window.setArg(1, (cl_float)qstart);
    kernel_window.setArg(2, (cl_float)qend);
    kernel_window.setArg(3, (cl_float)WIN_DK);
    kernel_window.setArg(4, (cl_float)KGRID);
    const cl::NDRange global_item_size(imageSizeX,imageSizeY,qsize);
    const cl::NDRange global_item_offset(0,0,qoffset);
    queue.enqueueNDRangeKernel(kernel_window,global_item_offset,global_item_size, local_item_size, NULL, NULL);
    queue.finish();
    
    
    return 0;
}
