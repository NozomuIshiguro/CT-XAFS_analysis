//
//  CT_ImageRegistration_C++.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/02/06.
//  Copyright (c) 2015 Nozomu Ishiguro. All rights reserved.
//

#include "CTXAFS.hpp"
#ifdef XANES_FIT
#include "XANES_fitting.hpp"
#include "XANES_fit_cl.hpp"
extern fitting_eq fiteq;
#endif

int mt_output_thread(int startAngleNo, int EndAngleNo,
                     input_parameter inp,
                     vector<float*> mt_outputs,float* p_pointer,float* p_err_pointer,
                     regMode regmode,int thread_id){
    
	//スレッドを待機/ロック
	m2.lock();
    string output_dir=inp.getOutputDir();
    string output_base=inp.getOutputFileBase();
    int startEnergyNo=inp.getStartEnergyNo();
    int endEnergyNo=inp.getEndEnergyNo();
    
    vector<float*> p;
    vector<float*> p_err;
    
    string shift_dir=output_dir+ "/imageRegShift";
    MKDIR(shift_dir.c_str());
    
    const int p_num = regmode.get_p_num()+regmode.get_cp_num();
    const int dA = EndAngleNo-startAngleNo+1;
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
        p.push_back(&p_pointer[p_num*dA*(i-startEnergyNo)]);
        p_err.push_back(&p_err_pointer[p_num*dA*(i-startEnergyNo)]);
    }
    
    ostringstream oss;
    for (int j=startAngleNo; j<=EndAngleNo; j++) {
        string fileName_output= shift_dir+ AnumTagString(j,"/shift", ".txt");
        ofstream ofs(fileName_output,ios::out|ios::trunc);
        ofs<<regmode.ofs_transpara();
        
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            string fileName_output= output_dir+ "/" + EnumTagString(i,"","/") + AnumTagString(j,output_base, ".raw");
            oss << "output file: " << fileName_output <<endl;
            outputRawFile_stream(fileName_output,mt_outputs[i-startEnergyNo]+(j-startAngleNo)*IMAGE_SIZE_M,IMAGE_SIZE_M);
            
            for (int k=0; k<p_num; k++) {
                ofs.precision(7);
                ofs<<p[i-startEnergyNo][k+p_num*(j-startAngleNo)]<<"\t"
                <<p_err[i-startEnergyNo][k+p_num*(j-startAngleNo)]<<"\t";
            }
            ofs<<endl;
        }
        ofs.close();
    }
    oss << endl;
    cout << oss.str();
    //スレッドをアンロック
    m2.unlock();
    
    
    //delete [] mt_outputs_pointer;
	for (int i = startEnergyNo; i <= endEnergyNo; i++) {
		delete[] mt_outputs[i - startEnergyNo];
	}
    delete [] p_pointer;
    delete [] p_err_pointer;


    return 0;
}

int mt_conversion(cl::CommandQueue queue,cl::Kernel kernel,
                  cl::Buffer dark_buffer,cl::Buffer I0_buffer,
                  cl::Buffer mt_buffer,cl::Image2DArray mt_image,cl::Image2DArray mt_outputImg,
                  const cl::NDRange global_item_size,const cl::NDRange local_item_size,
                  unsigned short *It_pointer,int dA, mask msk, bool refBool){
    
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
    
    cl::Buffer It_buffer(context, CL_MEM_READ_ONLY, sizeof(cl_ushort)*(IMAGE_SIZE_M+32)*dA, 0, NULL);
    queue.enqueueWriteBuffer(It_buffer, CL_TRUE, 0, sizeof(cl_ushort)*(IMAGE_SIZE_M+32)*dA, It_pointer, NULL, NULL);
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
    kernel.setArg(12, 0);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_item_size, local_item_size, NULL, NULL);
    queue.finish();
    
    return 0;
}

int imageReg_thread(cl::CommandQueue command_queue, CL_objects CLO,
                    unsigned short *It_img_target,vector<unsigned short*> It_img_sample,
                    int startAngleNo,int EndAngleNo,
					input_parameter inp, regMode regmode, mask msk, int thread_id){
    try {
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
        string devicename = device.getInfo<CL_DEVICE_NAME>();
		size_t WorkGroupSize = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), IMAGE_SIZE_X);
        const int dA=EndAngleNo-startAngleNo+1;
        const int p_num = regmode.get_p_num()+regmode.get_cp_num();
        cl::ImageFormat format(CL_RG,CL_FLOAT);
        int startEnergyNo=inp.getStartEnergyNo();
        int endEnergyNo=inp.getEndEnergyNo();
        int targetEnergyNo=inp.getTargetEnergyNo();
        float lambda=inp.getLambda_t();
        int Num_trial=inp.getNumTrial();
        
        vector<cl::Kernel> kernel=CLO.kernels;
        cl::Buffer dark_buffer=CLO.dark_buffer;
        cl::Buffer I0_target_buffer=CLO.I0_target_buffer;
        vector<cl::Buffer> I0_sample_buffers=CLO.I0_sample_buffers;
        
        // p_vec, p_err_vec, mt_sample
        vector<float*>p_vec;
        vector<float*>p_err_vec;
        vector<float*>mt_sample_img;
        float* p_vec_pointer;
        float* p_err_vec_pointer;
        p_vec_pointer = new float[p_num*dA*(endEnergyNo-startEnergyNo+1)];
        p_err_vec_pointer = new float[p_num*dA*(endEnergyNo-startEnergyNo+1)];
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            p_vec.push_back(&p_vec_pointer[p_num*dA*(i-startEnergyNo)]);
            p_err_vec.push_back(&p_err_vec_pointer[p_num*dA*(i-startEnergyNo)]);
            mt_sample_img.push_back(new float[IMAGE_SIZE_M*dA]);
		}
        
        
        //Buffer declaration
		cl::Buffer mt_target_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*IMAGE_SIZE_M*dA, 0, NULL);
		cl::Buffer mt_sample_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*IMAGE_SIZE_M*dA, 0, NULL);
        vector<cl::Image2DArray> mt_target_image;
        vector<cl::Image2DArray> mt_sample_image;
        for (int i=0; i<4; i++) {
            int mergeN = 1<<i;
            mt_target_image.push_back(cl::Image2DArray(context, CL_MEM_READ_WRITE,format,dA,
                                                       IMAGE_SIZE_X/mergeN,IMAGE_SIZE_Y/mergeN,
                                                       0,0,NULL,NULL));
            mt_sample_image.push_back(cl::Image2DArray(context, CL_MEM_READ_WRITE,format,dA,
                                                       IMAGE_SIZE_X/mergeN,IMAGE_SIZE_Y/mergeN,
                                                       0,0,NULL,NULL));
        }
        cl::Buffer p_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*p_num*dA, 0, NULL);
        cl::Buffer p_err_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*p_num*dA, 0, NULL);
        cl::Buffer p_target_buffer(context, CL_MEM_READ_ONLY, sizeof(cl_float)*p_num*dA, 0, NULL);
        cl::Buffer lambda_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*dA, 0, NULL);
        cl::Image2DArray mt_target_outputImg(context, CL_MEM_READ_WRITE,format,dA,IMAGE_SIZE_X,IMAGE_SIZE_Y,0,0,NULL,NULL);
        cl::Image2DArray mt_sample_outputImg(context, CL_MEM_READ_WRITE,format,dA,IMAGE_SIZE_X,IMAGE_SIZE_Y,0,0,NULL,NULL);
        
        
        //kernel dimension declaration
        const cl::NDRange global_item_size(WorkGroupSize*dA,IMAGE_SIZE_Y,1);
        const cl::NDRange local_item_size(WorkGroupSize,1,1);
        
        
        //Energy loop setting
        vector<int> LoopEndenergyNo={startEnergyNo,endEnergyNo};
        vector<int> LoopStartenergyNo;
        LoopStartenergyNo.push_back(min(targetEnergyNo, endEnergyNo));
        LoopStartenergyNo.push_back(max(targetEnergyNo, startEnergyNo));        
        

		//target mt conversion
        mt_conversion(command_queue,kernel[0],dark_buffer,I0_target_buffer,
                      mt_target_buffer,mt_target_image[0], mt_target_outputImg,
                      global_item_size,local_item_size,It_img_target,dA,msk,true);
        delete [] It_img_target;
        
        //target image reg parameter (p_target_buffer) initialize
        for (int k=0; k<dA; k++) {
            command_queue.enqueueWriteBuffer(p_target_buffer, CL_TRUE, sizeof(cl_float)*p_num*k, sizeof(cl_float)*p_num,regmode.p_ini,NULL,NULL);
        }
        command_queue.finish();
        
        
        //It_target merged image create
        if (regmode.get_regModeNo()>=0) {
            for (int i=3; i>0; i--) {
                int mergeN = 1<<i;
                int localsize = min((int)WorkGroupSize,IMAGE_SIZE_X/mergeN);
                const cl::NDRange global_item_size_merge(localsize*dA,1,1);
                const cl::NDRange local_item_size_merge(localsize,1,1);
                
                kernel[1].setArg(0, mt_target_image[0]);
                kernel[1].setArg(1, mt_target_image[i]);
                kernel[1].setArg(2, mergeN);
                command_queue.enqueueNDRangeKernel(kernel[1], cl::NullRange, global_item_size_merge, local_item_size_merge, NULL, NULL);
            }
            command_queue.finish();
        }
        
        
        //process when (sample Enegry No. == target Energy No.)
        if ((targetEnergyNo>=startEnergyNo)&(targetEnergyNo<=endEnergyNo)) {
            if (regmode.get_regModeNo()>=0) {
                //kernel setArgs of outputing image reg results to buffer
                kernel[3].setArg(0, mt_target_outputImg);
                kernel[3].setArg(1, mt_target_buffer);
                kernel[3].setArg(2, p_target_buffer);
                
                //output image reg results to buffer
                command_queue.enqueueNDRangeKernel(kernel[3], NULL, global_item_size, local_item_size, NULL, NULL);
                command_queue.finish();
            }
                
            command_queue.enqueueReadBuffer(mt_target_buffer, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M*dA, mt_sample_img[targetEnergyNo - startEnergyNo], NULL, NULL);
            command_queue.finish();
            
            command_queue.enqueueReadBuffer(p_target_buffer,CL_TRUE,0,sizeof(cl_float)*p_num*dA,p_vec[targetEnergyNo-startEnergyNo],NULL,NULL);
			command_queue.finish();
			ostringstream oss;
            for (int j=startAngleNo; j<=EndAngleNo; j++) {
                for (int t=0; t<p_num; t++) {
                    //p_vec[targetEnergyNo - startEnergyNo][t+p_num*(j-startAngleNo)]=0;
                    p_err_vec[targetEnergyNo - startEnergyNo][t+p_num*(j-startAngleNo)]=0;
                }
                
                oss << "Device("<<thread_id+1<<"): "<< devicename << ", angle: "<<j<< ", energy: "<<targetEnergyNo<<endl;
                oss <<regmode.get_oss_target();
            }
            cout << oss.str();
        }
        

        //kernel setArgs of It_sample merged image create
        kernel[1].setArg(0, mt_sample_image[0]);
        
        //kernel setArgs of Image registration
        kernel[2].setArg(2, lambda_buffer);
        kernel[2].setArg(3, p_buffer);
        kernel[2].setArg(4, p_err_buffer);
        kernel[2].setArg(5, p_target_buffer);
        kernel[2].setArg(8, 1.0f);
        
        //kernel setArgs of outputing image reg results to buffer
        kernel[3].setArg(0, mt_sample_outputImg);
        kernel[3].setArg(1, mt_sample_buffer);
        kernel[3].setArg(2, p_buffer);
        
        
        for (int s=0; s<2; s++) {//1st cycle:i<targetEnergyNo, 2nd cycle:i>targetEnergyNo
            int di=(-1+2*s); //1st -1, 2nd +1
            if (startEnergyNo==endEnergyNo){
                if(startEnergyNo==targetEnergyNo) break;
            }else if ((LoopStartenergyNo[s]+di)*di>LoopEndenergyNo[s]*di) {
                continue;
            }
            
            //image reg parameter (p_buffer) initialize
            for (int k=0; k<dA; k++) {
                command_queue.enqueueWriteBuffer(p_buffer, CL_TRUE, sizeof(cl_float)*p_num*k, sizeof(cl_float)*p_num,regmode.p_ini,NULL,NULL);
            }
            command_queue.finish();
            
            int ds = (LoopStartenergyNo[s]==targetEnergyNo) ? di:0;
            for (int i=LoopStartenergyNo[s]+ds; i*di<=LoopEndenergyNo[s]*di; i+=di) {

                //sample mt conversion
                mt_conversion(command_queue,kernel[0],dark_buffer,I0_sample_buffers[i-startEnergyNo],mt_sample_buffer,mt_sample_image[0], mt_sample_outputImg,global_item_size,local_item_size,It_img_sample[i-startEnergyNo],dA,msk,true);
                
        
                if (regmode.get_regModeNo()>=0) {
                    
                    //It_sample merged image create
                    for (int j=3; j>0; j--) {
                        int mergeN = 1<<j;
                        int localsize = min((int)WorkGroupSize,IMAGE_SIZE_X/mergeN);
                        const cl::NDRange global_item_size_merge(localsize*dA,1,1);
                        const cl::NDRange local_item_size_merge(localsize,1,1);
                        kernel[1].setArg(1, mt_sample_image[j]);
                        kernel[1].setArg(2, mergeN);
                        command_queue.enqueueNDRangeKernel(kernel[1], cl::NullRange, global_item_size_merge, local_item_size_merge, NULL, NULL);
                        command_queue.finish();
                    }
                    
                    
                    //lambda_buffer reset
                    command_queue.enqueueFillBuffer(lambda_buffer, (cl_float)lambda, 0, sizeof(cl_float)*dA,NULL,NULL);
                    command_queue.finish();
                    
                    
                    //Image registration
                    for (int j=3; j>=0; j--) {
                        int mergeN = 1<<j;
                        //cout<<mergeN<<endl;
                        int localsize = min((int)WorkGroupSize,IMAGE_SIZE_X/mergeN);
                        const cl::NDRange global_item_size_reg(localsize*dA,1,1);
                        const cl::NDRange local_item_size_reg(localsize,1,1);
                        kernel[2].setArg(0, mt_target_image[j]);
                        kernel[2].setArg(1, mt_sample_image[j]);
                        kernel[2].setArg(6, cl::Local(sizeof(cl_float)*localsize));//locmem
                        kernel[2].setArg(7, mergeN);
                        for (int trial=0; trial < Num_trial; trial++) {
                            command_queue.enqueueNDRangeKernel(kernel[2], NULL, global_item_size_reg, local_item_size_reg, NULL, NULL);
                            command_queue.finish();
                        }
                    }
                    
                    
                    //output image reg results to buffer
                    command_queue.enqueueNDRangeKernel(kernel[3], NULL, global_item_size, local_item_size, NULL, NULL);
                    command_queue.finish();
                    
                    
                    //read p_buffer & p_err_buffer
                    command_queue.enqueueReadBuffer(p_buffer,CL_TRUE,0,sizeof(cl_float)*p_num*dA,p_vec[i-startEnergyNo],NULL,NULL);
                    command_queue.enqueueReadBuffer(p_err_buffer,CL_TRUE,0,sizeof(cl_float)*p_num*dA,p_err_vec[i-startEnergyNo],NULL,NULL);
                    command_queue.finish();
                }
                
                
                ostringstream oss;
                for (int k=0; k<dA; k++) {
                    if (startAngleNo+k>EndAngleNo) {
                        break;
                    }
                    int *p_precision, *p_err_precision;
                    p_precision=new int[p_num];
                    p_err_precision=new int[p_num];
                    for (int n=0; n<p_num; n++) {
                        int a = (int)floor(log10(abs(p_vec[i-startEnergyNo][n+k*p_num])));
                        int b = (int)floor(log10(abs(p_err_vec[i-startEnergyNo][n+k*p_num])));
                        p_err_precision[n] = max(0,b)+1;
                        
                        
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
					oss << "Device(" << thread_id+1 << "): " << devicename << ", angle: " << startAngleNo + k << ", energy: " << i << endl;
                    oss << regmode.oss_sample(p_vec[i-startEnergyNo]+p_num*k, p_err_vec[i-startEnergyNo]+p_num*k,p_precision,p_err_precision);
                }
                cout << oss.str();

				//read output buffer
                command_queue.enqueueReadBuffer(mt_sample_buffer, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_M*dA, mt_sample_img[i - startEnergyNo], NULL, NULL);
                command_queue.finish();
            }
        }
        
		for (int i = startEnergyNo; i <= endEnergyNo; i++) {
			delete[] It_img_sample[i - startEnergyNo];
		}
        
        
        
#ifdef XANES_FIT //XANES fitting batch
        for (int j=startAngleNo; j<=EndAngleNo; j++) {
            XANES_fit_thread(command_queue,CLO.kernels_fit,fiteq, j, thread_id,
                             inp,CLO.energy_buffer,mt_sample_img,(j-startAngleNo)*IMAGE_SIZE_M);
        }
#endif
        //output imageReg
        if (inp.getImgRegOutput()) {
            output_th[thread_id].join();
            output_th[thread_id]=thread(mt_output_thread,
                                        startAngleNo,EndAngleNo,inp,
                                        move(mt_sample_img),move(p_vec_pointer),move(p_err_vec_pointer),regmode,thread_id);
		}
		else {
			for (int i = startEnergyNo; i <= endEnergyNo; i++) {
				delete[] mt_sample_img[i - startEnergyNo];
			}
            delete [] p_vec_pointer;
            delete [] p_err_vec_pointer;
        }
		

        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
    
    return 0;
}

int data_input_thread(int thread_id, cl::CommandQueue command_queue, CL_objects CLO,
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
    /*target It data input*/
    unsigned short *It_img_target;
    It_img_target = new unsigned short[(IMAGE_SIZE_M+32)*dA];
    string fileName_It_target = EnumTagString(targetEnergyNo,fileName_base,".his");
    readHisFile_stream(fileName_It_target,startAngleNo,EndAngleNo,It_img_target);
    /* Sample It data input */
    vector<unsigned short*>It_img_sample;
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
        It_img_sample.push_back(new unsigned short[(IMAGE_SIZE_M+32)*dA]);
        string fileName_It = EnumTagString(i,fileName_base,".his");
        readHisFile_stream(fileName_It,startAngleNo,EndAngleNo,It_img_sample[i-startEnergyNo]);
    }
    
	//スレッドをアンロック
	m1.unlock();

    //image_reg
    imageReg_th[thread_id].join();
    imageReg_th[thread_id] = thread(imageReg_thread,
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

int mergeRawhisBuffers(cl::CommandQueue queue, cl::Kernel kernel,
                       const cl::NDRange global_item_size,const cl::NDRange local_item_size,
                       unsigned short *img, cl::Buffer img_buffer,int mergeN){
    
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
    
    cl::Buffer rawhis_buffer(context, CL_MEM_READ_ONLY, sizeof(cl_ushort)*(IMAGE_SIZE_M+32)*mergeN, 0, NULL);
    
    queue.enqueueWriteBuffer(rawhis_buffer, CL_TRUE, 0, sizeof(cl_ushort)*(IMAGE_SIZE_M+32)*mergeN, img, NULL, NULL);
    kernel.setArg(0, rawhis_buffer);
    kernel.setArg(1, img_buffer);
    kernel.setArg(2, mergeN);
    queue.enqueueNDRangeKernel(kernel, NULL, global_item_size,local_item_size, NULL, NULL);
    
    return 0;
}


int imageRegistlation_ocl(string fileName_base, input_parameter inp,
                          OCL_platform_device plat_dev_list, regMode regmode)
{
    cl_int ret;
    
    int startEnergyNo=inp.getStartEnergyNo();
    int endEnergyNo=inp.getEndEnergyNo();
    int targetEnergyNo=inp.getTargetEnergyNo();
    int startAngleNo=inp.getStartAngleNo();
    int endAngleNo=inp.getEndAngleNo();
    
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
        string fileName_output = inp.getOutputDir() + "/" + EnumTagString(i,"","");
        MKDIR(fileName_output.c_str());
    }
    
    //OpenCL objects class
    vector<CL_objects> CLO;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        CL_objects CLO_contx;
        CLO.push_back(CLO_contx);
    }
    
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
    ifstream energy_ifs(inp.getEnergyFilePath(),ios::in);
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
    cout << "energy num for fitting: "<<num_energy<<endl<<endl;
    
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
    kernel_code += kernel_preprocessor_nums(E0,num_energy,fiteq.freeParaSize());
    kernel_code += fiteq.preprocessor_str();
    kernel_code += kernel_fit_src;
    size_t kernel_code_size = kernel_code.length();
#endif
    
    //OpenCL Program
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        cl::Program program=regmode.buildImageRegProgram(plat_dev_list.context(i));
        CLO[i].kernels.push_back(cl::Kernel(program,"mt_conversion", &ret));//0
        CLO[i].kernels.push_back(cl::Kernel(program,"merge", &ret));//1
        CLO[i].kernels.push_back(cl::Kernel(program,"imageRegistration", &ret));//2
        CLO[i].kernels.push_back(cl::Kernel(program,"output_imgReg_result", &ret));//3
        CLO[i].kernels.push_back(cl::Kernel(program,"merge_rawhisdata", &ret));//4
        
#ifdef XANES_FIT
        cl::Program::Sources source(1,std::make_pair(kernel_code.c_str(),kernel_code_size));
        cl::Program program_fit(plat_dev_list.context(i), source,&ret);
        //kernel build
        string option = "-cl-nv-maxrregcount=64";
        //option += " -cl-nv-verbose -Werror";
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
            maxWorkSize.push_back((int)min((int)plat_dev_list.dev(i,j).getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), IMAGE_SIZE_X));
            t++;
        }
    }
    
    
    // dark data input
    cout << "Reading dark file..."<<endl;
    unsigned short *dark_img;
    dark_img = new unsigned short[(IMAGE_SIZE_M+32)*30];
    string fileName_dark = fileName_base+ "dark.his";
    readHisFile_stream(fileName_dark,1,30,dark_img);
    
    
    // I0 sample data input
    cout << "Reading I0 files..."<<endl<<endl;
    vector<unsigned short*> I0_imgs;
    for (int i = startEnergyNo; i <= endEnergyNo; i++) {
        I0_imgs.push_back(new unsigned short[(IMAGE_SIZE_M+32)*30]);
        string fileName_I0 = EnumTagString(i,fileName_base,"_I0.his");
        int readHis_err=readHisFile_stream(fileName_I0,1,30,I0_imgs[i-startEnergyNo]);
        if (readHis_err<0) {
            endEnergyNo=i-1;
            cout <<"No more I0 at Energy No. at over"<<i-1<<endl;
            break;
        }
    }
    
        
    // I0 target data input
    unsigned short *I0_img_target;
    I0_img_target = new unsigned short[(IMAGE_SIZE_M+32)*30];
    string fileName_I0_target;
    fileName_I0_target =  EnumTagString(targetEnergyNo,fileName_base,"_I0.his");
    readHisFile_stream(fileName_I0_target,1,30,I0_img_target);
    
    
    //create dark, I0_target, I0_sample buffers
    const int p_num = regmode.get_p_num()+regmode.get_cp_num();
	for (int i = 0; i<plat_dev_list.contextsize(); i++) {
        CLO[i].dark_buffer=cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        CLO[i].I0_target_buffer=cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        CLO[i].p_freefix_buffer=cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float)*p_num*dA[i], 0, NULL);
        
        const cl::NDRange global_item_size(IMAGE_SIZE_X,IMAGE_SIZE_Y,1);
        const cl::NDRange local_item_size(maxWorkSize[i],1,1);
        
        //merge rawhis buffers to dark_buffer
        mergeRawhisBuffers(plat_dev_list.queue(i, 0), CLO[i].kernels[4], global_item_size, local_item_size, dark_img, CLO[i].dark_buffer,30);
        
        //merge rawhis buffers to I0_target_buffer
        mergeRawhisBuffers(plat_dev_list.queue(i, 0), CLO[i].kernels[4], global_item_size, local_item_size, I0_img_target, CLO[i].I0_target_buffer,30);
        
        //merge rawhis buffers to I0_sample_buffer
        for (int j = startEnergyNo; j <= endEnergyNo; j++) {
            CLO[i].I0_sample_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL));
            mergeRawhisBuffers(plat_dev_list.queue(i, 0), CLO[i].kernels[4], global_item_size, local_item_size, I0_imgs[j-startEnergyNo], CLO[i].I0_sample_buffers[j-startEnergyNo],30);
        }
#ifdef XANES_FIT
        CLO[i].energy_buffer=cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*num_energy, 0, NULL);
        plat_dev_list.queue(i,0).enqueueWriteBuffer(CLO[i].energy_buffer, CL_TRUE, 0, sizeof(cl_float)*num_energy, &energy[0], NULL, NULL);
#endif
    }
    delete[] dark_img;
    delete[] I0_img_target;
	for (int i = startEnergyNo; i <= endEnergyNo; i++) {
		delete[] I0_imgs[i-startEnergyNo];
	}
    
    //mask settings
    mask msk(inp);
    
    //start threads
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        input_th.push_back(thread(dummy));
        imageReg_th.push_back(thread(dummy));
		output_th.push_back(thread(dummy));
#ifdef XANES_FIT
        output_th_fit.push_back(thread(dummy));
#endif
    }
    for (int i=startAngleNo; i<=endAngleNo;) {
		for (int j = 0; j<plat_dev_list.contextsize(); j++) {
            if (input_th[j].joinable()) {
				input_th[j].join();
				input_th[j] = thread(data_input_thread,j,plat_dev_list.queue(j,0),CLO[j],
                                     i, min(i + dA[j] - 1, endAngleNo),
                                     fileName_base,inp,regmode,msk);
				i+=dA[j];
				if (i > endAngleNo) break;
            } else{
                this_thread::sleep_for(chrono::milliseconds(100));
            }

			if (i > endAngleNo) break;
        }
    }
    
	for (int j = 0; j<plat_dev_list.contextsize(); j++) {
		input_th[j].join();
	}
	for (int j = 0; j<plat_dev_list.contextsize(); j++) {
		imageReg_th[j].join();
	}
	for (int j = 0; j<plat_dev_list.contextsize(); j++) {
		output_th[j].join();
	}
#ifdef XANES_FIT
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        output_th_fit[j].join();
    }
#endif
    
    return 0;
}