﻿//
//  imagingXAFSXAFS-CT_registration_ocl.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2016/10/10.
//  Copyright © 2016年 Nozomu Ishiguro. All rights reserved.
//

#include "imagingXAFS-CT_registration.hpp"
#ifdef XANES_FIT
#include "XANES_fitting.hpp"
#include "XANES_fit_cl.hpp"
extern fitting_eq fiteq;
#endif

#define PI 3.14159265358979323846

unsigned short* I0_imgs;
static int imageSizeX;
static int imageSizeY;
static int imageSizeM;

static int cout_thread(string message){
    
    cout << message;
    
    return 0;
}

int imQXAFSsmooting(cl::CommandQueue queue, cl::Kernel kernel,
                    vector<float*>mt_imgs, float *smoothed_mt_img,
                    int dA, int startRawImg, int numRawImg){
    
	try {
	    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
	    cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
	    cl::Buffer raw_mt_buffer(context, CL_MEM_READ_ONLY, sizeof(cl_float)*imageSizeM*dA*numRawImg, 0, NULL);
	    cl::Buffer smoothed_mt_buffer(context, CL_MEM_WRITE_ONLY,sizeof(cl_float)*imageSizeM*dA, 0, NULL);
    
	    int localsize = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(),imageSizeX);
	    const cl::NDRange global_item_size(imageSizeX,imageSizeY,dA);
	    const cl::NDRange local_item_size(localsize,1,1);
    
	    for(int i=0;i<numRawImg;i++){
	        queue.enqueueWriteBuffer(raw_mt_buffer, CL_FALSE, sizeof(cl_float)*imageSizeM*dA*i, sizeof(cl_float)*imageSizeM*dA, mt_imgs[startRawImg+i],NULL,NULL);
	    }
        queue.finish();
        
    
		kernel.setArg(0, raw_mt_buffer);
		kernel.setArg(1, smoothed_mt_buffer);
		kernel.setArg(2, numRawImg);
		queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_item_size, local_item_size, NULL, NULL);
		queue.finish();
    
		queue.enqueueReadBuffer(smoothed_mt_buffer,CL_TRUE,0,sizeof(cl_float)*imageSizeM*dA,smoothed_mt_img,NULL,NULL);
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


int createSmoothedEnergyList(input_parameter *inp){
    
    int startEnergyNo=(*inp).getStartEnergyNo();
    int endEnergyNo= (*inp).getEndEnergyNo();
    
	cout << (*inp).getRawAngleFilePath() << endl;
    ifstream rawAngle_ifs((*inp).getRawAngleFilePath(),ios::in);
    if (!rawAngle_ifs.is_open()) {
        cout<<"Angle file not found."<<endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(-1);
    }
    vector<float> rawAngle;
    int i=0;
	char* dummy;
	dummy = new char[128];
	rawAngle_ifs.getline(dummy, 128);
	cout << dummy << endl;
    do {
		rawAngle_ifs.getline(dummy, 128);
		//if (rawAngle_ifs.eof()) break;
		//cout << dummy << endl;
		istringstream iss((string)dummy);
		float a,b;
        iss>>a>>b; //a:mono angle(c), b:mono angle(o)
        rawAngle.push_back(b); 
		cout << i << ": " << rawAngle[i] << endl;
        i++;
    } while (!rawAngle_ifs.eof());
    
	cout << (*inp).getXAFSparameterFilePath() << endl;
	ifstream parameter_ifs((*inp).getXAFSparameterFilePath(), ios::in);
	if (!parameter_ifs.is_open()) {
		cout << "parameter file not found." << endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(-1);
	}
    vector<float> smoothedAngle;
    vector<float> blockStartAngle;
    vector<float> blockDeltaAngle;
    vector<int> blockNumPnts;
    float monoD;
    int numBlock;
    char *buffer;
    buffer =new char[256];
    parameter_ifs.getline(buffer,256/*,'\n'*/);
	//cout << buffer << endl;
    istringstream iss(buffer);
    iss>>monoD>>numBlock;
	//cout << monoD << ", " << numBlock << endl;
    for(int i=0;i<numBlock;i++){
        int a;
        iss>>a;
        blockNumPnts.push_back(a);
		//cout << blockNumPnts[i] << endl;
    }
    parameter_ifs.getline(buffer,256);
	//cout << buffer << endl;
    istringstream iss2(buffer);
    for(int i=0;i<numBlock+1;i++){
        float a;
        iss2>>a;
        blockStartAngle.push_back(a);
		//cout << blockStartAngle[i] << endl;
    }
    parameter_ifs.getline(buffer,256);
    istringstream iss3(buffer);
    for(int i=0;i<numBlock;i++){
        float a;
        iss3>>a;
        blockDeltaAngle.push_back(a);
		//cout << blockDeltaAngle[i] << endl;
    }
	//int t = 0;
    for(int i=0;i<numBlock;i++){
        for (int j=0; j<=blockNumPnts[i]; j++) {
            smoothedAngle.push_back(blockStartAngle[i]+blockDeltaAngle[i]*j);
			//cout <<t<<": "<< smoothedAngle[t] << endl;
			//t++;
        }
    }
    smoothedAngle.push_back(blockStartAngle[numBlock]);
	//cout << t << ": " << smoothedAngle[t] << endl;

	cout << smoothedAngle.size() << endl;
    vector<int> numPntsInSmoothedPnts(smoothedAngle.size(),0);
    i=startEnergyNo;
    int offset=0;
    for (int j=0; j<smoothedAngle.size();) {
        if(j==0 && rawAngle[i] > (3*smoothedAngle[j]-smoothedAngle[j+1])/2){
			//cout << i<<",";
			i++;
            offset++;
        }else if(j==0 && (rawAngle[i]-(3*smoothedAngle[j]-smoothedAngle[j+1])/2)*(rawAngle[i]-(smoothedAngle[j]+smoothedAngle[j+1])/2)<=0){
			if (i <= endEnergyNo) {
				numPntsInSmoothedPnts[j]++;
				//cout << i << ",";
			}
			i++;
        }else if(j==smoothedAngle.size()-1 && (rawAngle[i]-(smoothedAngle[j]+smoothedAngle[j-1])/2)*(rawAngle[i]-(3*smoothedAngle[j]-smoothedAngle[j-1])/2)<=0){
			if (i <= endEnergyNo) {
				numPntsInSmoothedPnts[j]++;
				//cout << i << ",";
			}
			i++;
        }else if (rawAngle[i]>smoothedAngle[j]) {
			i++;
        }else if((rawAngle[i]-smoothedAngle[j])*(rawAngle[i]-smoothedAngle[j+1])<=0){
			if (i <= endEnergyNo) {
				numPntsInSmoothedPnts[j]++;
				//cout << i << ",";
			}
			i++;
		}
		else {
			//cout << endl;
			cout << j << ": " << numPntsInSmoothedPnts[j] << endl;
			j++;
        }
    }
	(*inp).smoothingOffset=offset;
    //vector<float> smoothedEnergyList;
    for (int j=0; j<smoothedAngle.size();j++) {
        if(numPntsInSmoothedPnts[j]>0){
			(*inp).smoothedEnergyList.push_back(12398.52f/(2.0f*monoD*(float)sin(smoothedAngle[j]/180.0f*PI)));
			(*inp).numPntsInSmoothedPnts.push_back(numPntsInSmoothedPnts[j]);
        }
    }
    
    string fileName_output= (*inp).getEnergyFilePath();
    ofstream ofs(fileName_output,ios::out|ios::trunc);
    for (int j=0; j<(*inp).smoothedEnergyList.size();j++) {
        ofs.precision(7);
        ofs<< (*inp).smoothedEnergyList[j]<<endl;
        //cout <<j<<": "<<smoothedEnergyList[j]<<", "<<inp.numPntsInSmoothedPnts[j]<<endl;
    }
    ofs.close();
    
#ifdef XANES_FIT
    float startEnergy=inp.getStartEnergy();
    float endEnergy=inp.getEndEnergy();
    float E0=inp.getE0();
    
    //energy file input & processing
    
    vector<float> energy;
    fit_startEnergyNo=0, fit_endEnergyNo=0;
    for(int i=0;i<inp.smoothedEnergyList.size();i++){
        float a = inp.smoothedEnergyList;
        cout<<i<<": "<<a;
        if ((a>=startEnergy)&(a<=endEnergy)) {
            energy.push_back(a-E0);
            cout<<" <- fitting range";
            fit_endEnergyNo = i;
        } else if(a<startEnergy) {
            fit_startEnergyNo = i+1;
        }
        cout<<endl;
        i++;
    }
    int num_energy=fit_endEnergyNo-fit_startEnergyNo+1;
    inp.setFittingStartEnergyNo(fit_startEnergyNo);
    inp.setFittingEndEnergyNo(fit_endEnergyNo);
    ostringstream oss;
    oss << fit_startEnergyNo<<"-"<<fit_endEnergyNo;
    inp.setEnergyNoRange(oss.str());
    oss.flush();
    cout << "energy num for fitting: "<<num_energy<<endl<<endl;
#endif
    
    return 0;
}



int imXAFSCT_mt_conversion(cl::CommandQueue queue,cl::Kernel kernel,
                  cl::Buffer dark_buffer, cl::Buffer I0_buffer,
                  cl::Buffer mt_buffer,cl::Image2DArray mt_image,cl::Image2DArray mt_outputImg,
                  const cl::NDRange global_item_size,const cl::NDRange local_item_size,const cl::NDRange global_item_offset,
                  vector<unsigned short*> It_pointer, int Enum, int dA, mask msk, bool refBool){
    
	try {
		cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Buffer It_buffer(context, CL_MEM_READ_ONLY, sizeof(cl_ushort)*(imageSizeM + 32)*dA, 0, NULL);
        for (int i=0; i<dA; i++) {
            queue.enqueueWriteBuffer(It_buffer, CL_FALSE, sizeof(cl_ushort)*(imageSizeM + 32)*i, sizeof(cl_ushort)*(imageSizeM + 32), &It_pointer[i][Enum*(int64_t)(imageSizeM+32)], NULL, NULL);
        }
		queue.finish();
        
    
		kernel.setArg(0, dark_buffer);
		kernel.setArg(1, I0_buffer);
		kernel.setArg(2, It_buffer);
		kernel.setArg(3, mt_buffer);
		kernel.setArg(4, mt_image);
		kernel.setArg(5, mt_outputImg);
		if (refBool) {
			kernel.setArg(6, msk.refMask_shape);
			kernel.setArg(7, msk.refMask_x);
			kernel.setArg(8, msk.refMask_y);
			kernel.setArg(9, msk.refMask_width);
			kernel.setArg(10, msk.refMask_height);
			kernel.setArg(11, msk.refMask_angle);
		}else{
			kernel.setArg(6, msk.sampleMask_shape);
			kernel.setArg(7, msk.sampleMask_x);
			kernel.setArg(8, msk.sampleMask_y);
			kernel.setArg(9, msk.sampleMask_width);
			kernel.setArg(10, msk.sampleMask_height);
			kernel.setArg(11, msk.sampleMask_angle);
		}
		queue.enqueueNDRangeKernel(kernel, global_item_offset, global_item_size, local_item_size, NULL, NULL);
		queue.finish();
	}
	catch (cl::Error ret) {
		cerr << "ERROR at mt conversion: " << ret.what() << "(" << ret.err() << ")" << endl;
		cout << "Press 'Enter' to quit." << endl;
		string dummy;
		getline(cin, dummy);
		exit(ret.err());
	}
    
    return 0;
}

int smoothed_mt_output_thread(int startAngleNo, int EndAngleNo,
	input_parameter inp,
	vector<float*> mt_outputs, vector<float*> p, vector<float*> p_err,
	regMode regmode, int thread_id,bool cnt) {

	//スレッドを待機/ロック
	m2.lock();
	string output_dir = inp.getOutputDir();
	string output_base = inp.getOutputFileBase();
	int startEnergyNo = inp.getStartEnergyNo();
	int endEnergyNo = inp.getEndEnergyNo();


	string shift_dir = output_dir + "/imageRegShift";
	MKDIR(shift_dir.c_str());

    const int p_num = regmode.get_p_num();
    const int output_p_num =(cnt) ?p_num+2:p_num;
	//const int dA = EndAngleNo - startAngleNo + 1;


	int imageNum = (int)mt_outputs.size();
	ostringstream oss;
	for (int j = startAngleNo; j <= EndAngleNo; j++) {
		string fileName_output = shift_dir + AnumTagString(j, "/shift", ".txt");
		ofstream ofs(fileName_output, ios::out | ios::trunc);
		ofs << regmode.ofs_transpara();

		for (int i = 1; i <= imageNum; i++) {
			string fileName_output = output_dir + "/" + EnumTagString(i, "", "/") + AnumTagString(j, output_base, ".raw");
			oss << "output file: " << fileName_output << endl;
			outputRawFile_stream(fileName_output, mt_outputs[i-1] + (j - startAngleNo)*imageSizeM,imageSizeM);
		}

		for (int i = startEnergyNo; i <= endEnergyNo; i++) {
			for (int k = 0; k<output_p_num; k++) {
				ofs << fixed << setprecision(7) << p[i - startEnergyNo][k + (p_num+2)*(j - startAngleNo)] << "\t"
					<< p_err[i - startEnergyNo][k + (p_num+2)*(j - startAngleNo)] << "\t";
			}
			ofs << endl;
		}
		ofs.close();
	}
	oss << endl;
	cout << oss.str();
	//スレッドをアンロック
	m2.unlock();


	//delete [] mt_outputs_pointer;
	for (int i = 1; i <= imageNum; i++) {
		delete[] mt_outputs[i - 1];
	}
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
        delete[] p[i-startEnergyNo];
        delete[] p_err[i-startEnergyNo];
    }


	return 0;
}

int imXAFSCT_imageReg_thread(cl::CommandQueue command_queue, CL_objects CLO,
                             vector<unsigned short*> It_img_target,vector<unsigned short*> It_img_sample,
                             int startAngleNo,int EndAngleNo,
                             input_parameter inp, regMode regmode, mask msk, int thread_id){
    string errorArert="";
    try {
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
        string devicename = device.getInfo<CL_DEVICE_NAME>();
        int numComUnit = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        size_t maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        const int dA=EndAngleNo-startAngleNo+1;
		int dE = numComUnit / dA;
        const int p_num = regmode.get_p_num();
        cl::ImageFormat format(CL_RG,CL_FLOAT);
        int startEnergyNo=inp.getStartEnergyNo();
        int endEnergyNo=inp.getEndEnergyNo();
        int targetEnergyNo=inp.getTargetEnergyNo();
        float lambda=inp.getLambda_t();
        int Num_trial=inp.getNumTrial();
        float CI=10.0f;
        int mergeLevel=2;
        
       
        // p_vec, p_err_vec, mt_sample
        vector<float*>p_vec;
        vector<float*>p_err_vec;
        vector<float*>mt_sample_img;
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            p_vec.push_back(new float[(p_num+2)*dA]);
            p_err_vec.push_back(new float[(p_num+2)*dA]);
            mt_sample_img.push_back(new float[imageSizeM*dA]);
        }
        
        
        //Buffer declaration
        cl::Buffer dark_buffer=CLO.dark_buffer;
        cl::Buffer I0_target_buffer = CLO.I0_target_buffer;
        cl::Buffer I0_buffer(context, CL_MEM_READ_ONLY, sizeof(cl_float)*imageSizeM, 0, NULL);

        cl::Buffer mt_target_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*imageSizeM*dA*dE, 0, NULL);
        cl::Buffer mt_sample_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*imageSizeM*dA*dE, 0, NULL);
        vector<cl::Image2DArray> mt_target_image;
        vector<cl::Image2DArray> mt_sample_image;
        vector<cl::Image2DArray> weight_image;
        vector<cl::Buffer> devX, dF2X, dFX, tJJX, tJdFX;
        for (int i=0; i<=mergeLevel; i++) {
            int mergeN = 1<<i;
            mt_target_image.push_back(cl::Image2DArray(context, CL_MEM_READ_WRITE,format,dA*dE,
                                                       imageSizeX/mergeN,imageSizeY/mergeN,
                                                       0,0,NULL,NULL));
            mt_sample_image.push_back(cl::Image2DArray(context, CL_MEM_READ_WRITE,format,dA*dE,
                                                       imageSizeX/mergeN,imageSizeY/mergeN,
                                                       0,0,NULL,NULL));
            weight_image.push_back(cl::Image2DArray(context, CL_MEM_READ_WRITE,format,dA*dE,
                                                       imageSizeX/mergeN,imageSizeY/mergeN,
                                                       0,0,NULL,NULL));
            devX.push_back(cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeY/mergeN*dA*dE,0,NULL));
            dF2X.push_back(cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeY/mergeN*dA*dE,0,NULL));
            dFX.push_back(cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeY/mergeN*dA*dE,0,NULL));
            tJJX.push_back(cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeY/mergeN*dA*dE*(p_num+2)*(p_num+3)/2,0,NULL));
            tJdFX.push_back(cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeY/mergeN*dA*dE*(p_num+2),0,NULL));
        }
        cl::Image2DArray mt_target_outputImg(context, CL_MEM_READ_WRITE,format,dA*dE,imageSizeX,imageSizeY,0,0,NULL,NULL);
        cl::Image2DArray mt_sample_outputImg(context, CL_MEM_READ_WRITE,format,dA*dE,imageSizeX,imageSizeY,0,0,NULL,NULL);
        cl::Buffer p_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*(p_num+2)*dA*dE, 0, NULL);
        cl::Buffer dp_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*(p_num+2)*dA*dE, 0, NULL);
        cl::Buffer p_cnd_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*(p_num+2)*dA*dE, 0,NULL);
        cl::Buffer p_err_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*(p_num+2)*dA*dE, 0,NULL);
        cl::Buffer p_fix_buffer(context, CL_MEM_READ_ONLY, sizeof(cl_char)*(p_num+2), 0, NULL);
        cl::Buffer p_target_buffer(context, CL_MEM_READ_ONLY, sizeof(cl_float)*(p_num+2)*dA*dE, 0, NULL);
        cl::Buffer lambda_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dA*dE, 0, NULL);
        cl::Buffer nyu_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dA*dE, 0, NULL);
        cl::Buffer dL_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dA*dE, 0, NULL);
        cl::Buffer rho_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dA*dE, 0, NULL);
        cl::Buffer dF2old_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dA*dE, 0, NULL);
        cl::Buffer dF2new_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dA*dE, 0, NULL);
        cl::Buffer dF_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dA*dE, 0, NULL);
		cl::Buffer dev_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dA*dE, 0, NULL);
        cl::Buffer tJdF_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*(p_num+2)*dA*dE, 0, NULL);
        cl::Buffer tJJ_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*(p_num+2)*(p_num+3)/2*dA*dE, 0, NULL);
        
		
        
        //kernel dimension declaration
        //for mt conversion
        size_t workGroupSize = min(maxWorkGroupSize,(size_t)imageSizeX);
		const cl::NDRange global_item_size0(imageSizeX, imageSizeY, dA);
        const cl::NDRange local_item_size0(workGroupSize, 1, 1);
        //for merge his & smoothing processes
        workGroupSize = min(maxWorkGroupSize,(size_t)imageSizeX);
        const cl::NDRange global_item_size4(imageSizeX,imageSizeY, 1);
        const cl::NDRange local_item_size4(workGroupSize, 1, 1);
		//for output
		workGroupSize = min(maxWorkGroupSize, (size_t)imageSizeX);
		const cl::NDRange global_item_size5(imageSizeX, imageSizeY, dA*dE);
		const cl::NDRange local_item_size5(workGroupSize, 1, 1);


        
        //Energy loop setting
        vector<int> LoopEndenergyNo={startEnergyNo,endEnergyNo};
        vector<int> LoopStartenergyNo;
        LoopStartenergyNo.push_back(min(targetEnergyNo, endEnergyNo));
        LoopStartenergyNo.push_back(max(targetEnergyNo, startEnergyNo));
        
        
        //target mt conversion
        cl::Kernel kernel_mt = CLO.getKernel("mt_conversion");
        workGroupSize = min(maxWorkGroupSize,(size_t)imageSizeX);
        for (int en=0; en<dE; en++) {
            cl::NDRange global_item_offset(0,0,en*dA);
            imXAFSCT_mt_conversion(command_queue,kernel_mt,dark_buffer,I0_target_buffer,
                                   mt_target_buffer,mt_target_image[0], mt_target_outputImg,
                                   global_item_size0,local_item_size0,global_item_offset,
                                   It_img_target,0,dA,msk,true);
        }
        
        for (int i=0; i<dA; i++) {
            delete [] It_img_target[i];
        }
		
        
        //target image reg parameter (p_target_buffer) initialize
        for (int p=0; p<p_num+2; p++) {
            command_queue.enqueueFillBuffer(p_target_buffer, (cl_float)regmode.p_ini[p], sizeof(cl_float)*p*dA*dE, sizeof(cl_float)*dA*dE);
            command_queue.finish();
        }
        
        //free(1)/fix(0) settings of image reg parameter (p_fix_buffer)
        command_queue.enqueueWriteBuffer(p_fix_buffer, CL_TRUE, 0, sizeof(cl_char)*(p_num+2),regmode.p_fix,NULL,NULL);
        command_queue.finish();
        
        
        //mt_target merged image create
        cl::Kernel kernel_merge = CLO.getKernel("merge");
        if (regmode.get_regModeNo()>=0) {
            for (int i=mergeLevel; i>0; i--) {
                int mergeN = 1<<i;
                int localsize = min((int)maxWorkGroupSize,imageSizeX/mergeN);
                const cl::NDRange global_item_size_merge(imageSizeX/mergeN,imageSizeY/mergeN,dA*dE);
                const cl::NDRange local_item_size_merge(localsize,1,1);
                
                kernel_merge.setArg(0, mt_target_image[0]);
                kernel_merge.setArg(1, mt_target_image[i]);
                kernel_merge.setArg(2, (cl_uint)mergeN);
                command_queue.enqueueNDRangeKernel(kernel_merge, cl::NullRange, global_item_size_merge, local_item_size_merge, NULL, NULL);
                command_queue.finish();
            }
        }
        
        
        //process when (sample Enegry No. == target Energy No.)
        cl::Kernel kernel_output = CLO.getKernel("output_imgReg_result");
        if ((targetEnergyNo>=startEnergyNo)&(targetEnergyNo<=endEnergyNo)) {
            if (regmode.get_regModeNo()>=0) {
                //kernel setArgs of outputing image reg results to buffer
                kernel_output.setArg(0, mt_target_outputImg);
                kernel_output.setArg(1, mt_target_buffer);
                kernel_output.setArg(2, p_target_buffer);
                
                //output image reg results to buffer
                command_queue.enqueueNDRangeKernel(kernel_output, NULL, global_item_size5, local_item_size5, NULL, NULL);
                command_queue.finish();
            }
            
            
            //read mt data from GPU
            command_queue.enqueueReadBuffer(mt_target_buffer, CL_TRUE, 0, sizeof(cl_float)*imageSizeM*dA, mt_sample_img[targetEnergyNo-startEnergyNo], NULL, NULL);
            command_queue.finish();
            
            //read fp_target from GPU
            for (int k=0; k<dA; k++) {
                for (int p=0; p<p_num+2; p++) {
                    command_queue.enqueueReadBuffer(p_target_buffer,CL_FALSE,sizeof(cl_float)*(k+p*dA*dE),sizeof(cl_float),&p_vec[targetEnergyNo-startEnergyNo][p+k*(p_num+2)],NULL,NULL);
                }
            }
            command_queue.finish();
            
            ostringstream oss;
            for (int j=startAngleNo; j<=EndAngleNo; j++) {
                for (int t=0; t<p_num+2; t++) {
                    p_err_vec[targetEnergyNo - startEnergyNo][t+(p_num+2)*(j-startAngleNo)]=0;
                }
                
                oss << "Device("<<thread_id+1<<"): "<< devicename << ", angle: "<<j<< ", energy: "<<targetEnergyNo<<endl;
                oss <<regmode.get_oss_target();
            }
            
            thread th(cout_thread, oss.str());
			th.detach();
            //cout << oss.str();
        }
        
        
        //kernel setArgs of It_sample merged image create
        kernel_merge.setArg(0, mt_sample_image[0]);
        cl::Kernel kernel_mergeHis = CLO.getKernel("merge_rawhisdata");
        
        for (int s=0; s<2; s++) {//1st cycle:i<targetEnergyNo, 2nd cycle:i>targetEnergyNo
            int di=(-1+2*s); //1st -1, 2nd +1
            if (startEnergyNo==endEnergyNo){
                if(startEnergyNo==targetEnergyNo) break;
            }else if ((LoopStartenergyNo[s]+di)*di>LoopEndenergyNo[s]*di) {
                continue;
            }


			//image reg parameter (p_buffer) initialize
			for (int p = 0; p<p_num + 2; p++) {
				command_queue.enqueueFillBuffer(p_buffer, (cl_float)regmode.p_ini[p], sizeof(cl_float)*p*dA*dE, sizeof(cl_float)*dA*dE);
				command_queue.finish();
			}

            
            int ds = (LoopStartenergyNo[s]==targetEnergyNo) ? di:0;
            for (int i=LoopStartenergyNo[s]+ds; i*di<=LoopEndenergyNo[s]*di; i+=di*dE) {
                
				//image reg parameter (p_buffer) initialize
				for (int p = 0; p<p_num; p++) {
					command_queue.enqueueFillBuffer(p_buffer, (cl_float)regmode.p_ini[p], sizeof(cl_float)*p*dA*dE, sizeof(cl_float)*dA*dE);
					command_queue.finish();
				}

                //sample mt conversion
				mergeRawhisBuffers(command_queue,kernel_mergeHis, global_item_size4, local_item_size4, &I0_imgs[(imageSizeM + 32)*(int64_t)(i - startEnergyNo)], I0_buffer, 1, imageSizeM);
                workGroupSize = min(maxWorkGroupSize,(size_t)imageSizeX);
                for (int en=0; en<dE; en++) {
                    if(i*di+en>LoopEndenergyNo[s]*di) break;
                    cl::NDRange global_item_offset(0,0,en*dA);
                    imXAFSCT_mt_conversion(command_queue,kernel_mt,dark_buffer,I0_buffer,mt_sample_buffer,mt_sample_image[0], mt_sample_outputImg, global_item_size0,local_item_size0, global_item_offset, It_img_sample,i-startEnergyNo+en*di,dA,msk,false);
                }

                
                if (regmode.get_regModeNo()>=0) {
                    
                    //It_sample merged image create
                    for (int j=mergeLevel; j>0; j--) {
                        unsigned int mergeN = 1<<j;
                        int localsize = min((unsigned int)maxWorkGroupSize,imageSizeX/mergeN);
                        const cl::NDRange global_item_size_merge(imageSizeX/mergeN,imageSizeY/mergeN,dA*dE);
                        const cl::NDRange local_item_size_merge(localsize,1,1);
                        kernel_merge.setArg(1, mt_sample_image[j]);
                        kernel_merge.setArg(2, (cl_uint)mergeN);
                        command_queue.enqueueNDRangeKernel(kernel_merge, cl::NullRange, global_item_size_merge, local_item_size_merge, NULL, NULL);
                        command_queue.finish();
                    }

                    
                    imageRegistration(command_queue, CLO,
                                      mt_target_image, mt_sample_image, weight_image,
                                      mt_sample_outputImg, mt_sample_buffer,
                                      p_buffer,p_target_buffer,p_fix_buffer,p_cnd_buffer,p_err_buffer,
                                      dF2old_buffer,dF2new_buffer,dF_buffer,tJJ_buffer,tJdF_buffer,
                                      dev_buffer,dp_buffer,lambda_buffer,dL_buffer,nyu_buffer,
                                      rho_buffer, dF2X,dFX,tJJX,tJdFX,devX,
                                      mergeLevel,imageSizeX,imageSizeY,p_num,dA*dE,CI,
                                      Num_trial,lambda);
                    
                    
                    //read p_buffer & p_err_buffer
                    for (int en=0; en<dE; en++) {
                        if(i*di+en>LoopEndenergyNo[s]*di) break;
                        for (int k=0; k<dA; k++) {
                            for (int p=0; p<p_num+2; p++) {
                                command_queue.enqueueReadBuffer(p_buffer,CL_FALSE,sizeof(cl_float)*(k+(en+p*dE)*dA),sizeof(cl_float),&p_vec[i-startEnergyNo+en*di][p+k*(p_num+2)],NULL,NULL);
                                command_queue.enqueueReadBuffer(p_err_buffer,CL_FALSE,sizeof(cl_float)*(k+(en+p*dE)*dA),sizeof(cl_float),&p_err_vec[i-startEnergyNo+en*di][p+k*(p_num+2)],NULL,NULL);
                                command_queue.finish();
                            }
                        }
                    }
                    command_queue.finish();
                    
                    
                }
                
                
                ostringstream oss;
                for (int en=0; en<dE; en++) {
                    if(i*di+en>LoopEndenergyNo[s]*di) break;
                    
                    for (int k=0; k<dA; k++) {
                        if (startAngleNo+k>EndAngleNo) {
                            break;
                        }
                        int *p_precision, *p_err_precision;
                        p_precision=new int[p_num+2];
                        p_err_precision=new int[p_num+2];
                        for (int n=0; n<p_num+2; n++) {
                            int a = (int)floor(log10(abs(p_vec[i-startEnergyNo+en*di][n+k*(p_num+2)])));
                            int b = (int)floor(log10(abs(p_err_vec[i-startEnergyNo+en*di][n+k*(p_num+2)])));
                            p_err_precision[n] = max(0,b)+1;
                            
                            if(regmode.p_fix[n]==0.0f) p_precision[n]=3;
                            if (a>0) {
                                int c = (int)floor(log10(pow(10,a+1)-0.5));
                                if(a>c) a++;
                                
                                p_precision[n] = a+1 - min(0,b);
                            }else if(a<b){
                                p_precision[n] = 1;
                            }else{
                                p_precision[n]= a - max(-3,b) + 1;
                            }
                        }
                        oss << "Device(" << thread_id+1 << "): " << devicename << ", angle: " << startAngleNo + k << ", energy: " << i + en*di << endl;
                        oss << regmode.oss_sample(&p_vec[i-startEnergyNo+en*di][(p_num+2)*k], &p_err_vec[i-startEnergyNo+en*di][(p_num+2)*k],p_precision,p_err_precision);
                    }
                }
                thread th(cout_thread, oss.str());
				th.detach();
                //cout << oss.str();
                
                //read output buffer
                for (int en=0; en<dE; en++) {
					if (i*di + en>LoopEndenergyNo[s] * di) break;
                    command_queue.enqueueReadBuffer(mt_sample_buffer, CL_FALSE, sizeof(cl_float)*imageSizeM*dA*en, sizeof(cl_float)*imageSizeM*dA, mt_sample_img[i - startEnergyNo+en*di], NULL, NULL);
                }
                command_queue.finish();
            }
        }
        
        for (int i = 0; i<dA; i++) {
            delete[] It_img_sample[i];
        }
        
        
        //QXAFS smooting
        if(inp.getSmootingEnable()){
            vector<float*> smoothed_mt_sample_img;
			vector<float> smoothed_energy;
            int i=inp.smoothingOffset;
			//cout << inp.numPntsInSmoothedPnts.size() << endl;
            cl::Kernel kernel_smoothing = CLO.getKernel("imQXAFS_smoothing");
            for (int j=0; j<inp.numPntsInSmoothedPnts.size(); j++) {
				smoothed_mt_sample_img.push_back(new float[imageSizeM*dA]);
				smoothed_energy.push_back(inp.smoothedEnergyList[j]);
				imQXAFSsmooting(command_queue,kernel_smoothing,mt_sample_img,smoothed_mt_sample_img[j],dA, i, inp.numPntsInSmoothedPnts[j]);
				i += inp.numPntsInSmoothedPnts[j];
            }
            
            for (int i = startEnergyNo; i <= endEnergyNo; i++) {
                delete[] mt_sample_img[i - startEnergyNo];
            }
        
#ifdef XANES_FIT //XANES fitting batch
            for (int j=startAngleNo; j<=EndAngleNo; j++) {
                XANES_fit_thread(command_queue,CLO.kernels_fit,fiteq, j, thread_id,inp,
                                 CLO.energy_buffer, CLO.C_matrix_buffer, CLO.D_vector_buffer,CLO.freeFix_buffer,
                                 smoothed_mt_sample_img,(j-startAngleNo)*IMAGE_SIZE_M);
            }
#endif
            
            //output imageReg
            if (inp.getImgRegOutput()) {
                output_th[thread_id].join();
                output_th[thread_id]=thread(smoothed_mt_output_thread,
                                            startAngleNo,EndAngleNo,inp,
                                            move(smoothed_mt_sample_img),move(p_vec),move(p_err_vec),regmode,thread_id,true);
            } else {
                for (int i = 0; i < inp.numPntsInSmoothedPnts.size(); i++) {
                    delete[] mt_sample_img[i];
                }
                for (int i=startEnergyNo; i<=endEnergyNo; i++) {
                    delete[] p_vec[i-startEnergyNo];
                    delete[] p_err_vec[i-startEnergyNo];
                }
                
            }
            
        }else{
        
            //output imageReg
            if (inp.getImgRegOutput()) {
                output_th[thread_id].join();
                output_th[thread_id]=thread(mt_output_thread,
                                            startAngleNo,EndAngleNo,inp,
                                            move(mt_sample_img),move(p_vec),move(p_err_vec),regmode,thread_id,true);
            } else {
                for (int i = startEnergyNo; i <= endEnergyNo; i++) {
                    delete [] mt_sample_img[i - startEnergyNo];
                    delete [] p_vec[i - startEnergyNo];
                    delete [] p_err_vec[i - startEnergyNo];
                }
            }
        }
        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(ret.err());
    }
    
    return 0;
}




int imXAFSCT_data_input_thread(int thread_id, cl::CommandQueue command_queue, CL_objects CLO,
                               int startAngleNo,int EndAngleNo,
                               string fileName_base, input_parameter inp,
                               regMode regmode, mask msk){
    
    
    //スレッドを待機/ロック
    m1.lock();
    
    int startEnergyNo=inp.getStartEnergyNo();
    int endEnergyNo=inp.getEndEnergyNo();
    int targetEnergyNo=inp.getTargetEnergyNo();
    
    ostringstream oss;
    if (startAngleNo == EndAngleNo) {
        oss << "(" << thread_id + 1 << ") reading It files of angle " << startAngleNo << endl << endl;
    }
    else {
        oss << "(" << thread_id + 1 << ") reading It files of angle " << startAngleNo << "-" << EndAngleNo << endl << endl;
    }
    cout << oss.str();
    
    const int dA=EndAngleNo-startAngleNo+1;
    

    
    vector<unsigned short*>It_img_sample;
    vector<unsigned short*>It_img_target;
	int defaultDegit = 3;
    for (int i=0; i<dA; i++) {
        It_img_sample.push_back(new unsigned short[(imageSizeM+32)*(int64_t)(endEnergyNo-startEnergyNo+1)]);
        It_img_target.push_back(new unsigned short[imageSizeM+32]);

        string fileName_It = numTagString(startAngleNo+i, fileName_base, ".his", defaultDegit);
        // Sample It data input 
        int ret = readHisFile_stream(fileName_It,startEnergyNo,endEnergyNo,It_img_sample[i],imageSizeM);
		if (ret == -1) {
			for (int j = 1; j <= 4; j++) {
				fileName_It = numTagString(startAngleNo + i, fileName_base, ".his", j);
				ret = readHisFile_stream(fileName_It, startEnergyNo, endEnergyNo, It_img_sample[i],imageSizeM);
				if (ret == 0) {
					defaultDegit = j;
					cout << "   degit chaged to " << j << endl;
					break;
				}
			}
		}
		//target It data input
        readHisFile_stream(fileName_It,targetEnergyNo,targetEnergyNo,It_img_target[i],imageSizeM);
    }
    
    
    //スレッドをアンロック
    m1.unlock();
    
    //image_reg
    imageReg_th[thread_id].join();
    imageReg_th[thread_id] = thread(imXAFSCT_imageReg_thread,
                                    command_queue, CLO,
                                    move(It_img_target),move(It_img_sample),
                                    startAngleNo, EndAngleNo,
                                    inp,regmode,msk,thread_id);
    
    return 0;
}

//dummy thread
static int dummy(){
    return 0;
}



int imXAFSCT_Registration_ocl(string fileName_base, input_parameter inp,
                              OCL_platform_device plat_dev_list, regMode regmode)
{
    
    int startEnergyNo=inp.getStartEnergyNo();
    int endEnergyNo=inp.getEndEnergyNo();
    int targetEnergyNo=inp.getTargetEnergyNo();
    int startAngleNo=inp.getStartAngleNo();
    int endAngleNo=inp.getEndAngleNo();
    imageSizeX = inp.getImageSizeX();
    imageSizeY = inp.getImageSizeY();
    imageSizeM = inp.getImageSizeM();
    
    //create QXAFS smoothing lists
    if(inp.getSmootingEnable()) {
        createSmoothedEnergyList(&inp);
        
		string fileName_Energylist = inp.getOutputDir() + "/EnergyList.dat";
		ofstream ofs(fileName_Energylist, ios::out | ios::trunc);
        //create output dir
        for (int i=0; i<inp.numPntsInSmoothedPnts.size(); i++) {
            string fileName_output = inp.getOutputDir() + "/" + EnumTagString(i+1,"","");
            MKDIR(fileName_output.c_str());
			ofs << inp.smoothedEnergyList[i] <<endl;
		}
		ofs.close();

    //without smooting
    }else{
        //create output dir (fitting)
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            string fileName_output = inp.getOutputDir() + "/" + EnumTagString(i,"","");
            MKDIR(fileName_output.c_str());
        }
    }
	
    
    //OpenCL objects class
    vector<CL_objects> CLO;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        CL_objects CLO_contx;
        CLO.push_back(CLO_contx);
    }
    
    int scanN = inp.getScanN();
    
    
    
    
#ifdef XANES_FIT
    float startEnergy=inp.getStartEnergy();
    float endEnergy=inp.getEndEnergy();
    float E0=inp.getE0();
    
    //create output dir (fitting)
    for (int i=0; i<fiteq.ParaSize(); i++) {
        char buffer=fiteq.freefix_para()[i];
        if (atoi(&buffer)==1) {
            string fileName_output = inp.getFittingOutputDir() + "/"+fiteq.param_name(i);
            MKDIR(fileName_output.c_str());
        }
    }
    
    
    //energy file input & processing
    /*ifstream energy_ifs(inp.getEnergyFilePath(),ios::in);
    vector<float> energy;
    int i=startEnergyNo, fittingStartEnergyNo=0, fittingEndEnergyNo=0;
    do {
        float a;
        energy_ifs>>a;
        if (energy_ifs.eof()) break;
        cout<<i<<": "<<a;
        if ((a>=startEnergy)&(a<=endEnergy)) {
            energy.push_back(a-E0);
            cout<<" <- fitting range";
            fittingEndEnergyNo = i;
        } else if(a<startEnergy) {
            fittingStartEnergyNo = i+1;
        }
        cout<<endl;
        i++;
    } while (!energy_ifs.eof()||i>endEnergyNo);
    int num_energy=fittingEndEnergyNo-fittingStartEnergyNo+1;
    inp.setFittingStartEnergyNo(fittingStartEnergyNo);
    inp.setFittingEndEnergyNo(fittingEndEnergyNo);
    cout << "energy num for fitting: "<<num_energy<<endl<<endl;*/
    
    //kernel program source
    /*ifstream ifs("./XANES_fit.cl", ios::in);
     if(!ifs) {
     cerr << "   Failed to load kernel \n" << endl;
     return -1;
     }
     istreambuf_iterator<char> it(ifs);
     istreambuf_iterator<char> last;
     string kernel_src(it,last);
     ifs.close();*/
    
    
    //OpenCL Program
    string kernel_code="";
    kernel_code += fiteq.preprocessor_str();
    kernel_code += kernel_fit_src;
    size_t kernel_code_size = kernel_code.length();
#endif
    
    //OpenCL Program
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        cl::Program program=regmode.buildImageRegProgram(plat_dev_list.context(i),imageSizeX,imageSizeY);
        CLO[i].addKernel(program,"mt_conversion");
        CLO[i].addKernel(program,"merge");
        CLO[i].addKernel(program,"imageReg1X"); //estimate dF2(old), tJJ tJdF;
        CLO[i].addKernel(program,"imageReg1Y");
        CLO[i].addKernel(program,"LevenbergMarquardt");
        CLO[i].addKernel(program,"estimate_dL");
        CLO[i].addKernel(program,"updatePara");
        CLO[i].addKernel(program,"imageReg2X"); //estimate dF2(new)
        CLO[i].addKernel(program,"imageReg2Y");
        CLO[i].addKernel(program, "evaluateUpdateCandidate");
        CLO[i].addKernel(program,"updateOrHold");
        CLO[i].addKernel(program, "estimateParaError");
        CLO[i].addKernel(program,"output_imgReg_result");
        CLO[i].addKernel(program,"merge_rawhisdata");
        CLO[i].addKernel(program,"imQXAFS_smoothing");

        
#ifdef XANES_FIT
        cl::Program::Sources source(1,std::make_pair(kernel_code.c_str(),kernel_code_size));
        cl::Program program_fit(plat_dev_list.context(i), source,&ret);
        //kernel build
        string option = "";
#ifdef DEBUG
        option += "-D DEBUG";
#endif
        kernel_preprocessor_nums(E0, num_energy, fiteq.ParaSize(),fiteq.constrain_size);
        string GPUvendor =  plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0].getInfo<CL_DEVICE_VENDOR>();
        if(GPUvendor == "nvidia"){
            option += "-cl-nv-maxrregcount=64";
            //option += " -cl-nv-verbose -Werror";
        }
        else if (GPUvendor.find("NVIDIA Corporation") == 0) {
            option += " -cl-nv-maxrregcount=64";
        }
        ret=program_fit.build(option.c_str());
        //string logstr=program_fit.getBuildInfo<CL_PROGRAM_BUILD_LOG>(plat_dev_list.dev(0,0));
        //cout << logstr << endl;
        CLO[i].kernels_fit.push_back(cl::Kernel(program_fit,"XANES_fitting", &ret));//0
        CLO[i].kernels_fit.push_back(cl::Kernel(program_fit,"updateResults", &ret));//1
        CLO[i].kernels_fit.push_back(cl::Kernel(program_fit,"setMask", &ret));//2
        CLO[i].kernels_fit.push_back(cl::Kernel(program_fit,"applyThreshold", &ret));//3
#endif
    }
    
    
    //display OCL device
    int t=0;
    vector<int> dA;
    vector<int> maxWorkSize;
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
            cout << "CL DEVICE NAME: "<< device_pram<<endl;
			
            
            //working compute unit
            size_t device_pram_size[3]={0,0,0};
            plat_dev_list.dev(i,j).getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &device_pram_size);
            dA.push_back(min(inp.getNumParallel(),(int)device_pram_size[0]));
            cout<<"Number of working compute unit: "<<dA[t]<<"/"<<device_pram_size[0]<<endl<<endl;
            maxWorkSize.push_back((int)min((int)plat_dev_list.dev(i,j).getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), imageSizeX));
            t++;
        }
    }
    
    // dark data input
    cout << "Reading dark file..."<<endl;
    unsigned short *dark_img;
    dark_img = new unsigned short[(imageSizeM+32)*scanN];
    string fileName_dark = fileName_base+ "dark.his";
    readHisFile_stream(fileName_dark,1,scanN,dark_img,imageSizeM);

	// I0 sample data input
	cout << "Reading I0 files..." << endl << endl;
	unsigned short* I0_imgs_target;
	I0_imgs = new unsigned short[(imageSizeM + 32)*(int64_t)(endEnergyNo - startEnergyNo + 1)];
	I0_imgs_target = new unsigned short[imageSizeM + 32];
	string fileName_I0 = fileName_base + "I0.his";
	readHisFile_stream(fileName_I0, startEnergyNo, endEnergyNo, I0_imgs,imageSizeM);
	readHisFile_stream(fileName_I0, targetEnergyNo, targetEnergyNo, I0_imgs_target,imageSizeM);

	//create dark & I0_target buffers
	for (int i = 0; i<plat_dev_list.contextsize(); i++) {
		CLO[i].dark_buffer = cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
		CLO[i].I0_target_buffer = cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
        cl::Kernel kernel_mergeHis = CLO[i].getKernel("merge_rawhisdata");

		const cl::NDRange global_item_size(imageSizeX,imageSizeY, 1);
		const cl::NDRange local_item_size(maxWorkSize[i], 1, 1);

		//merge rawhis buffers to dark_buffer
		mergeRawhisBuffers(plat_dev_list.queue(i, 0), kernel_mergeHis, global_item_size, local_item_size, dark_img, CLO[i].dark_buffer, scanN,imageSizeM);

		//merge rawhis buffers to I0_target_buffer
		mergeRawhisBuffers(plat_dev_list.queue(i, 0), kernel_mergeHis, global_item_size, local_item_size, I0_imgs_target, CLO[i].I0_target_buffer, 1,imageSizeM);

#ifdef XANES_FIT
		int paramsize = (int)fiteq.ParaSize();
		int constrainsize = (int)fiteq.constrain_size;
		CLO[i].energy_buffer = cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*num_energy, 0, NULL);
		CLO[i].C_matrix_buffer = cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*paramsize*constrainsize, 0, NULL);
		CLO[i].D_vector_buffer = cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*constrainsize, 0, NULL);
		CLO[i].freeFix_buffer = cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_char)*paramsize, 0, NULL);

		plat_dev_list.queue(i, 0).enqueueWriteBuffer(CLO[i].energy_buffer, CL_TRUE, 0, sizeof(cl_float)*num_energy, &energy[0], NULL, NULL);
#endif
	}
	delete[] dark_img;
	delete[] I0_imgs_target;
    

	//mask settings
	mask msk(inp);

	//start threads
	for (int j = 0; j < plat_dev_list.contextsize(); j++) {
		input_th.push_back(thread(dummy));
		imageReg_th.push_back(thread(dummy));
		output_th.push_back(thread(dummy));
#ifdef XANES_FIT
		output_th_fit.push_back(thread(dummy));
#endif
	}

	for (int i = startAngleNo; i <= endAngleNo;) {
		for (int j = 0; j < plat_dev_list.contextsize(); j++) {
			if (input_th[j].joinable()) {
				input_th[j].join();
				input_th[j] = thread(imXAFSCT_data_input_thread, j, plat_dev_list.queue(j, 0), CLO[j],
					i, min(i + dA[j] - 1, endAngleNo),
					fileName_base, inp, regmode, msk);
				i += dA[j];
				if (i > endAngleNo) break;
			}
			else {
				this_thread::sleep_for(chrono::milliseconds(100));
			}

			if (i > endAngleNo) break;
		}
	}

	for (int j = 0; j < plat_dev_list.contextsize(); j++) {
		input_th[j].join();
	}
	for (int j = 0; j < plat_dev_list.contextsize(); j++) {
		imageReg_th[j].join();
	}
	for (int j = 0; j < plat_dev_list.contextsize(); j++) {
		output_th[j].join();
	}
#ifdef XANES_FIT
	for (int j = 0; j < plat_dev_list.contextsize(); j++) {
		output_th_fit[j].join();
	}
#endif
	delete[] I0_imgs;

	return 0;
}
