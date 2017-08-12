//
//  2D_EXAFS_fitting_ocl.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2016/01/04.
//  Copyright © 2016年 Nozomu Ishiguro. All rights reserved.
//

#include "EXAFS.hpp"

vector<int> GPUmemoryControl(int imageSizeX, int imageSizeY,int ksize,int Rsize,int qsize,
                             int fittingMode, int num_fpara, vector<shellObjects> shObjs, cl::CommandQueue queue){
    
    vector<int> processImageSizeY;
    processImageSizeY.push_back(imageSizeY); //for other process
    processImageSizeY.push_back(imageSizeY); //for FFT/IFFT size
    
    size_t GPUmemorySize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
    
    size_t usingMemorySize;
    size_t shellnum = shObjs.size();
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
        //tJJ, inv_tJJ
        usingMemorySize += 2*imageSizeX*processImageSizeY[0]*num_fpara*(num_fpara+1)/2*sizeof(float);
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
	}

    return 0;
}



int EXAFS_kFit(cl::CommandQueue queue, cl::Program program,
               cl::Buffer chidata, cl::Buffer S02, vector<shellObjects> shObj,
               int kw, float kstart, float kend, int imageSizeX, int imageSizeY,
               bool freeS02,int numTrial,float lambda){
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
        cl::Buffer inv_tJJ(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*FreeParaSize*(FreeParaSize+1)/2,0,NULL);
        cl::Buffer tJdF(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*FreeParaSize,0,NULL);
        vector<cl::Buffer> Jacobian;
        vector<cl::Buffer> para_backup;
        for(int i=0;i<FreeParaSize;i++){
            Jacobian.push_back(cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*ksize,0,NULL));
            para_backup.push_back(cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM,0,NULL));
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
        kernel_LM.setArg(5, inv_tJJ);
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
        kernel_UR.setArg(2, rho_buff);
        
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
            /*float* tJJ_data;
            tJJ_data = new float[imageSizeM*FreeParaSize*(FreeParaSize+1)/2];
            queue.enqueueReadBuffer(tJJ, CL_TRUE, 0, sizeof(float)*imageSizeM*FreeParaSize*(FreeParaSize+1)/2, tJJ_data);
            cout<<"tJJ"<<endl;
            for (int i=0; i<FreeParaSize*(FreeParaSize+1)/2; i++) {
                cout <<"tJJ["<<i<<"]: "<<tJJ_data[i*imageSizeM]<<endl;
            }
            cout<<endl;
            cout<<"tJdF"<<endl;
            float* tJdF_data;
            tJdF_data = new float[imageSizeM*FreeParaSize];
            queue.enqueueReadBuffer(tJdF, CL_TRUE, 0, sizeof(float)*imageSizeM*FreeParaSize, tJdF_data);
            for (int i=0; i<FreeParaSize; i++) {
                cout <<"tJdF["<<i<<"]: "<<tJdF_data[i*imageSizeM]<<endl;
            }
            cout<<endl;*/
            
            
            //estimate dF2
            kernel_dF2.setArg(0, dF2_old);
            queue.enqueueNDRangeKernel(kernel_dF2,NULL,global_item_size, local_item_size, NULL, NULL);
            queue.finish();
            /*float* dF2_data;
            dF2_data = new float[imageSizeM];
            queue.enqueueReadBuffer(dF2_old, CL_TRUE, 0, sizeof(float)*imageSizeM, dF2_data);
            cout <<"dF2 old: "<<dF2_data[0]<<endl<<endl;*/
            
            
            //Levenberg-Marquardt
            queue.enqueueNDRangeKernel(kernel_LM,NULL,global_item_size,local_item_size, NULL, NULL);
            queue.finish();
            /*float* inv_tJJ_data;
            inv_tJJ_data = new float[imageSizeM*FreeParaSize*(FreeParaSize+1)/2];
            queue.enqueueReadBuffer(inv_tJJ, CL_TRUE, 0, sizeof(float)*imageSizeM*FreeParaSize*(FreeParaSize+1)/2, inv_tJJ_data);
            queue.finish();
            cout<<"inv_tJdF"<<endl;
            for (int i=0; i<FreeParaSize*(FreeParaSize+1)/2; i++) {
                cout <<"inv_tJJ["<<i<<"]: "<<inv_tJJ_data[i*imageSizeM]<<endl;
            }
            cout<<endl;*/
            /*float* dp_data;
            dp_data = new float[imageSizeM*FreeParaSize];
            queue.enqueueReadBuffer(dp_img, CL_TRUE, 0, sizeof(float)*imageSizeM*FreeParaSize, dp_data);
            for (int i=0; i<FreeParaSize; i++) {
                cout <<"dp["<<i<<"]: "<<dp_data[i*imageSizeM]<<endl;
            }
            cout<<endl;*/
            
            
            //estimate dL
            queue.enqueueNDRangeKernel(kernel_dL,NULL,global_item_size,local_item_size, NULL, NULL);
            queue.finish();
            
            
            //update paramater as candidate
            //S02
            fpn = 0;
            if(freeS02){
                queue.enqueueCopyBuffer(S02, para_backup[fpn], 0, 0, sizeof(float)*imageSizeM);
                
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
                        shObj[sh].copyPara(para_backup[fpn], paraN+1);
                        shObj[sh].updatePara(dp_img, paraN+1, fpn);
                        fpn++;
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
                kernel_UR.setArg(1, para_backup[fpn]);
                queue.enqueueNDRangeKernel(kernel_UR,NULL,global_item_size, local_item_size, NULL, NULL);
                queue.finish();
                fpn++;
            }
            //others
            for (int sh=0; sh<shObj.size(); sh++) {
                for (int paraN=0; paraN<7; paraN++) {
                    if(shObj[sh].getFreeFixPara(paraN+1)){
                        shObj[sh].restorePara(para_backup[fpn], rho_buff, paraN+1);
                        fpn++;
                    }
                }
            }
        }
        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
    
    return 0;
}


int EXAFS_RFit(cl::CommandQueue queue, cl::Program program, cl::Buffer w_factor,
               cl::Buffer FTchidata, cl::Buffer S02, vector<shellObjects> shObj,
               int kw, float kstart, float kend, float Rstart, float Rend,
               int imageSizeX, int imageSizeY, int FFTimageSizeY,
               bool freeS02,int numTrial,float lambda){
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
        cl::Buffer inv_tJJ(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*FreeParaSize*(FreeParaSize+1)/2,0,NULL);
        cl::Buffer tJdF(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*FreeParaSize,0,NULL);
        vector<cl::Buffer> Jacobian;
        vector<cl::Buffer> para_backup;
        for(int i=0;i<FreeParaSize;i++){
            Jacobian.push_back(cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*Rsize,0,NULL));
            para_backup.push_back(cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM,0,NULL));
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
        kernel_LM.setArg(5, inv_tJJ);
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
        kernel_UR.setArg(2, rho_buff);
        
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
            
            
            //estimate dL
            queue.enqueueNDRangeKernel(kernel_dL,NULL,global_item_size,local_item_size, NULL, NULL);
            queue.finish();
            
            
            //update paramater as candidate
            //S02
            fpn = 0;
            if(freeS02){
                queue.enqueueCopyBuffer(S02, para_backup[fpn], 0, 0, sizeof(float)*imageSizeM);
                
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
                        shObj[sh].copyPara(para_backup[fpn], paraN+1);
                        shObj[sh].updatePara(dp_img, paraN+1, fpn);
                        fpn++;
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
                kernel_UR.setArg(1, para_backup[fpn]);
                queue.enqueueNDRangeKernel(kernel_UR,NULL,global_item_size, local_item_size, NULL, NULL);
                queue.finish();
                fpn++;
            }
            //others
            for (int sh=0; sh<shObj.size(); sh++) {
                for (int paraN=0; paraN<7; paraN++) {
                    if(shObj[sh].getFreeFixPara(paraN+1)){
                        shObj[sh].restorePara(para_backup[fpn], rho_buff, paraN+1);
                        fpn++;
                    }
                }
            }
        }
        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
    
    return 0;
}


int EXAFS_qFit(cl::CommandQueue queue, cl::Program program, cl::Buffer w_factor,
               cl::Buffer chiqdata, cl::Buffer S02, vector<shellObjects> shObj,
               int kw, float kstart, float kend, float Rstart, float Rend, float qstart, float qend,
               int imageSizeX, int imageSizeY, int FFTimageSizeY,
               bool freeS02,int numTrial,float lambda){
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
        cl::Buffer inv_tJJ(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*FreeParaSize*(FreeParaSize+1)/2,0,NULL);
        cl::Buffer tJdF(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM*FreeParaSize,0,NULL);
        vector<cl::Buffer> Jacobian;
        vector<cl::Buffer> para_backup;
        for(int i=0;i<FreeParaSize;i++){
            Jacobian.push_back(cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM*qsize,0,NULL));
            para_backup.push_back(cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*imageSizeM,0,NULL));
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
        kernel_LM.setArg(5, inv_tJJ);
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
        kernel_UR.setArg(2, rho_buff);
        
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
            
            
            //estimate dL
            queue.enqueueNDRangeKernel(kernel_dL,NULL,global_item_size,local_item_size, NULL, NULL);
            queue.finish();
            
            
            //update paramater as candidate
            //S02
            fpn = 0;
            if(freeS02){
                queue.enqueueCopyBuffer(S02, para_backup[fpn], 0, 0, sizeof(float)*imageSizeM);
                
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
                        shObj[sh].copyPara(para_backup[fpn], paraN+1);
                        shObj[sh].updatePara(dp_img, paraN+1, fpn);
                        fpn++;
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
                kernel_UR.setArg(1, para_backup[fpn]);
                queue.enqueueNDRangeKernel(kernel_UR,NULL,global_item_size, local_item_size, NULL, NULL);
                queue.finish();
                fpn++;
            }
            //others
            for (int sh=0; sh<shObj.size(); sh++) {
                for (int paraN=0; paraN<7; paraN++) {
                    if(shObj[sh].getFreeFixPara(paraN+1)){
                        shObj[sh].restorePara(para_backup[fpn], rho_buff, paraN+1);
                        fpn++;
                    }
                }
            }
        }
        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
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
    
    
    ChiData_k(queue, program, chiData, move(chiData_pointer), kw, kstart, kend, imageSizeX, imageSizeY,
              imgStckOrChiStck, offsetM);
    
    
    //FFT
    for (int offsetY=0; offsetY<imageSizeY; offsetY += FFTimageSizeY) {
        FFT(chiData, FTchiData, w_factor, queue, program,
            imageSizeX, imageSizeY, FFTimageSizeY, offsetY,
            koffset,ksize,Roffset,Rsize);
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
    
    
    ChiData_R(queue, program, FTchiData, move(chiData_pointer), w_factor, kw,
              kstart, kend, Rstart, Rend, imageSizeX, imageSizeY, FFTimageSizeY,
              imgStckOrChiStck, offsetM);
    
    
    //IFFT
    for (int offsetY=0; offsetY<imageSizeY; offsetY += FFTimageSizeY) {
        IFFT(FTchiData, chiqData, w_factor, queue, program,
             imageSizeX, imageSizeY, FFTimageSizeY, offsetY,
             Roffset,Rsize,qoffset,qsize);
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


int testEXAFS(vector<FEFF_shell> shells, OCL_platform_device plat_dev_list) {
	try {
		//kernel program source
		ifstream ifs("/Users/ishiguro/Desktop/CT_programs/CT-XANES_analysis/EXAFS_fitting/EXAFS_fit.cl", ios::in);
		if (!ifs) {
			cerr << "   Failed to load kernel \n" << endl;
			return -1;
		}
		istreambuf_iterator<char> it(ifs);
		istreambuf_iterator<char> last;
		string kernel_code1(it, last);
		ifs.close();
		//cout<<kernel_code1<<endl;

		//kernel program source
		ifstream ifs2("/Users/ishiguro/Desktop/CT_programs/CT-XANES_analysis/EXAFS_fitting/3D_FFT.cl", ios::in);
		if (!ifs2) {
			cerr << "   Failed to load kernel \n" << endl;
			return -1;
		}
		istreambuf_iterator<char> it2(ifs2);
		istreambuf_iterator<char> last2;
		string kernel_code2(it2, last2);
		ifs2.close();
        
        ifstream ifs3("/Users/ishiguro/Desktop/CT_programs/CT-XANES_analysis/share\ source/LevenbergMarquardt.cl", ios::in);
        if (!ifs3) {
            cerr << "   Failed to load kernel \n" << endl;
            return -1;
        }
        istreambuf_iterator<char> it3(ifs3);
        istreambuf_iterator<char> last3;
        string kernel_code3(it3, last3);
        ifs3.close();

		//cout << kernel_code<<endl;
		vector<cl::Program> programs;
		cl_int ret;
		for (int i = 0; i<plat_dev_list.contextsize(); i++) {
#if defined (OCL120)
			//cl::Program::Sources source(1,std::make_pair(kernel_code1.c_str(),kernel_code1.length()));
			cl::Program::Sources source;
			source.push_back(std::make_pair(kernel_code1.c_str(), kernel_code1.length()));
			source.push_back(std::make_pair(kernel_code2.c_str(), kernel_code2.length()));
            source.push_back(std::make_pair(kernel_code3.c_str(), kernel_code3.length()));
#else
			cl::Program::Sources source;
			source.push_back(kernel_code1);
			source.push_back(kernel_code2);
            source.push_back(kernel_code3);
#endif
			programs.push_back(cl::Program(plat_dev_list.context(i), source, &ret));
			//kernel build
			ostringstream oss;
			oss << "-D FFT_SIZE=" << FFT_SIZE << " ";
			oss << "-D PARA_NUM=" << 4 << " ";
			oss << "-D PARA_NUM_SQ=" << 16 << " ";
			string option = oss.str();
			option += "-cl-fp32-correctly-rounded-divide-sqrt -cl-single-precision-constant ";
#ifdef DEBUG
			option += "-D DEBUG ";//-Werror";
#endif
			string GPUvendor = plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0].getInfo<CL_DEVICE_VENDOR>();
			if (GPUvendor == "nvidia") {
				option += " -cl-nv-maxrregcount=64 -cl-nv-verbose";
			}
			else if (GPUvendor.find("NVIDIA Corporation") == 0) {
				option += " -cl-nv-maxrregcount=64";
			}
			ret = programs[i].build(option.c_str());
#ifdef DEBUG
			string logstr = programs[i].getBuildInfo<CL_PROGRAM_BUILD_LOG>(plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0]);
			cout << logstr << endl;
#endif

		}


		string inputpath = "/Users/ishiguro/Documents/実験/Fuel_Cell/FC_MEA(0.5mg_cm-2)_13PCCP/chi(igor)/Pt_foil.chi";
		vector<float*> chi_pointers;
		for (int j = 0; j<4; j++) {
			ifstream chi_ifs(inputpath, ios::in);
			vector<float> k_vec, chi_vec;
			int npnts = 0;
			do {
				string a, b;
				chi_ifs >> a >> b;

				if (chi_ifs.eof()) break;
				float aa, bb;
				try {
					aa = stof(a);
					bb = stof(b);
				}
				catch (invalid_argument ret) { //ヘッダータグが存在する場合に入力エラーになる際への対応
					continue;
				}
				k_vec.push_back(aa);
				chi_vec.push_back(bb);
				//cout<<npnts+1<<": "<<aa<<","<<bb<<endl;
				npnts++;
			} while (!chi_ifs.eof());
			chi_ifs.close();
			//cout<<endl;

			chi_pointers.push_back(new float[npnts]);
			for (int i = 0; i<npnts; i++) {
				chi_pointers[j][i] = chi_vec[i];//*(j+1.0f);
												//cout << chi_pointers[0][i] <<endl;
			}
		}
		float* ej;
		ej = new float[4];
		ej[0] = 1.0f;
		ej[1] = 2.0f;
		ej[2] = 3.0f;
		ej[3] = 4.0f;



		vector<vector<shellObjects>> ShObj;
		vector<cl::Buffer> chiData_buff;
		vector<cl::Buffer> FTchiData_buff;
		vector<cl::Buffer> chiqData_buff;
		vector<cl::Buffer> S02;
		vector<cl::Buffer> ej_buff;
		float kstart = 3.0f;
		float kend = 14.0f;
		float Rstart = 0.0f;
		float Rend = 3.0f;
		int kw = 3;
		float qstart = 3.0f;
		float qend = 14.0f;
		int ksize = ceil((min(float(kend + WIN_DK), (float)MAX_KQ) - max(float(kstart - WIN_DK), 0.0f)) / KGRID);
		int koffset = floor(max((float)(kstart - WIN_DK), 0.0f) / KGRID);
		int Rsize = ceil((min(float(Rend + WIN_DR), (float)MAX_R) - max(float(Rstart - WIN_DR), 0.0f)) / RGRID);
		int Roffset = floor(max((float)(Rstart - WIN_DR), 0.0f) / RGRID);
		int qsize = ceil((min(float(qend + WIN_DK), (float)MAX_KQ) - max(float(qstart - WIN_DK), 0.0f)) / KGRID);
		int qoffset = floor(max((float)(qstart - WIN_DK), 0.0f) / KGRID);
		int imagesizeX = 4;
		for (int i = 0; i<plat_dev_list.contextsize(); i++) {
			vector<shellObjects> ShObj_atP;
			for (int j = 0; j<shells.size(); j++) {
				ShObj_atP.push_back(shellObjects(plat_dev_list.queue(i, 0), programs[i], shells[j], imagesizeX, 1));
			}
			ShObj.push_back(ShObj_atP);



			chiData_buff.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float2)*ksize*imagesizeX, 0, NULL));
			FTchiData_buff.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float2)*Rsize*imagesizeX, 0, NULL));
			chiqData_buff.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float2)*qsize*imagesizeX, 0, NULL));
			S02.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float)*imagesizeX, 0, NULL));
			ej_buff.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float)*imagesizeX, 0, NULL));
			cl::Buffer w_factor(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float2)*FFT_SIZE / 2, 0, NULL);
			createSpinFactor(w_factor, plat_dev_list.queue(i, 0), programs[i]);
			plat_dev_list.queue(i, 0).enqueueFillBuffer(S02[i], (cl_float)1.0f, 0, sizeof(cl_float)*imagesizeX);

			vector<int> processImageSizeY = GPUmemoryControl(2048, 2048, ksize, Rsize, qsize, 2, 4, ShObj[i], plat_dev_list.queue(i, 0));

			plat_dev_list.queue(i, 0).enqueueWriteBuffer(ej_buff[i], CL_TRUE, 0, sizeof(cl_float)*imagesizeX, ej);
			//ShObj[i][0].inputIniCN(12.0f, ej_buff[i]);

			//chi data
			//ChiData_k(plat_dev_list.queue(i,0),programs[i],chiData_buff[i],chi_pointers,kw, kstart, kend, imagesizeX, 1, false,0);
			ChiData_R(plat_dev_list.queue(i, 0), programs[i], FTchiData_buff[i], chi_pointers, w_factor, kw, kstart, kend, Rstart, Rend, imagesizeX, 1, 1, false, 0);
			//ChiData_q(plat_dev_list.queue(i,0),programs[i],chiqData_buff[i],chi_pointers,w_factor,kw, kstart, kend, Rstart, Rend, qstart, qend, imagesizeX, 1, 1, false, 0);


			//EXAFS fit
			//EXAFS_kFit(plat_dev_list.queue(i,0),programs[i],chiData_buff[i],S02[i], ShObj[i],kw, kstart, kend, imagesizeX, 1, false, 30, 0.1f);
			EXAFS_RFit(plat_dev_list.queue(i, 0), programs[i], w_factor, FTchiData_buff[i], S02[i], ShObj[i], kw, kstart, kend, Rstart, Rend, imagesizeX, 1, 1, false, 30, 0.1f);
			//EXAFS_qFit(plat_dev_list.queue(i,0),programs[i], w_factor,chiqData_buff[i], S02[i],ShObj[i],kw, kstart, kend, Rstart, Rend, qstart, qend, imagesizeX, 1, 1,false,30,0.1f);


			float* CN;
			float* Rval;
			float* dE0;
			float* ss;
			CN = new float[imagesizeX];
			Rval = new float[imagesizeX];
			dE0 = new float[imagesizeX];
			ss = new float[imagesizeX];
			ShObj[i][0].readParaImage(CN, 1);
			ShObj[i][0].readParaImage(Rval, 2);
			ShObj[i][0].readParaImage(dE0, 3);
			ShObj[i][0].readParaImage(ss, 4);
			for (int j = 0; j<imagesizeX; j++) {
				cout << CN[j] << "\t" << Rval[j] << "\t" << dE0[j] << "\t" << ss[j] << endl;
			}

			/*cl_float2* w_data;
			w_data = new cl_float2[FFT_SIZE];
			plat_dev_list.queue(i,0).enqueueReadBuffer(w_factor, CL_TRUE, 0, sizeof(cl_float2)*FFT_SIZE, w_data);
			for (int j=0; j<FFT_SIZE; j++) {
			cout << w_data[j].x << "\t" << w_data[j].y <<endl;
			}
			delete [] w_data;*/

			/*cl_float2* chi_data;
			chi_data = new cl_float2[MAX_KRSIZE];
			for (int j=0; j<MAX_KRSIZE; j++) {
			chi_data[j].x = 0.0f;
			chi_data[j].y = 0.0f;
			}
			plat_dev_list.queue(i,0).enqueueReadBuffer(chiData_buff[i], CL_TRUE, 0, sizeof(cl_float2)*ksize, &chi_data[koffset]);
			for (int j=0; j<MAX_KRSIZE; j++) {
			cout << chi_data[j].y <<endl;
			}
			delete [] chi_data;*/

			/*cl_float2* FTchi_data;
			FTchi_data = new cl_float2[MAX_KRSIZE];
			for (int j=0; j<MAX_KRSIZE; j++) {
			FTchi_data[j].x = 0.0f;
			FTchi_data[j].y = 0.0f;
			}
			plat_dev_list.queue(i,0).enqueueReadBuffer(FTchiData_buff[i], CL_TRUE, 0, sizeof(cl_float2)*Rsize, &FTchi_data[Roffset]);
			for (int j=0; j<MAX_KRSIZE; j++) {
			cout << FTchi_data[j].y<< "\t"<< -FTchi_data[j].x <<endl;
			}
			delete [] FTchi_data;*/
		}

	}
	catch (const cl::Error ret) {
		cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
	}

	return 0;
}
