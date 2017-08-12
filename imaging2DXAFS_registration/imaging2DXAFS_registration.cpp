//
//  imaging2DXAFS_registration.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/05/28.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//


#include "imaging2DXAFS.hpp"
static int buffsize=512;

int mt_output_thread(int startEnergyNo, int endEnergyNo,
                     string output_filepath, float* mt_outputs,vector<float*> p,vector<float*> p_err,
                     regMode regmode,int imageSizeM){
    
    const int p_num = regmode.get_p_num();
    
    string result_output;
    size_t ini = output_filepath.rfind(".");
    if(ini<output_filepath.length()-5) result_output= output_filepath+"_result.txt";
    else if(ini!=-1){
        result_output= output_filepath;
        result_output.replace(output_filepath.rfind("."), 4, "_result.txt");
    }else result_output= output_filepath+"_result.txt";
    ofstream ofs(result_output,ios::out|ios::trunc);
    ofs<<regmode.ofs_transpara();
    for(int i=startEnergyNo; i<=endEnergyNo; i++) {
        for (int k=0; k<p_num; k++) {
            ofs<<p[i-startEnergyNo][k]<<"\t"
            <<p_err[i-startEnergyNo][k]<<"\t";
        }
        ofs<<endl;
    }
    ofs.close();
    
    outputRawFile_stream(output_filepath,mt_outputs,imageSizeM*(endEnergyNo-startEnergyNo+1));
    
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
        delete [] p[i-startEnergyNo];
        delete [] p_err[i-startEnergyNo];
    }
    delete [] mt_outputs;
    return 0;
}

int mt_transfer(cl::CommandQueue queue,cl::Kernel kernel,
                  cl::Buffer mt_buffer,cl::Image2DArray mt_image,cl::Image2DArray mt_outputImg,
                  const cl::NDRange global_item_size,const cl::NDRange local_item_size,
                  float *mt_pointer, mask msk, bool refBool,int imageSizeM){
    
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
    
    queue.enqueueWriteBuffer(mt_buffer, CL_TRUE, 0, sizeof(cl_float)*imageSizeM, mt_pointer, NULL, NULL);
    //cout<<kernel.getInfo<CL_KERNEL_FUNCTION_NAME>()<<endl;
    kernel.setArg(0, mt_buffer);
    kernel.setArg(1, mt_image);
    kernel.setArg(2, mt_outputImg);
    if (refBool) {
        kernel.setArg(3, msk.refMask_shape);
        kernel.setArg(4, msk.refMask_x);
        kernel.setArg(5, msk.refMask_y);
        kernel.setArg(6, msk.refMask_width);
        kernel.setArg(7, msk.refMask_height);
        kernel.setArg(8, msk.refMask_angle);
    }else{
        kernel.setArg(3, msk.sampleMask_shape);
        kernel.setArg(4, msk.sampleMask_x);
        kernel.setArg(5, msk.sampleMask_y);
        kernel.setArg(6, msk.sampleMask_width);
        kernel.setArg(7, msk.sampleMask_height);
        kernel.setArg(8, msk.sampleMask_angle);
    }
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_item_size, local_item_size, NULL, NULL);
    queue.finish();
    
    return 0;
}

int imageReg_2D_thread(cl::CommandQueue command_queue, CL_objects CLO,
                       input_parameter inp, regMode regmode, mask msk){
    try {
        int imageSizeX = inp.getImageSizeX();
        int imageSizeY = inp.getImageSizeY();
        int imageSizeM = inp.getImageSizeM();
        
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
        string devicename = device.getInfo<CL_DEVICE_NAME>();
        size_t WorkGroupSize = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), imageSizeX);
        int startEnergyNo=inp.getStartEnergyNo();
        int endEnergyNo=inp.getEndEnergyNo();
        int dE = endEnergyNo - startEnergyNo+1;
        int targetEnergyNo=inp.getTargetEnergyNo();
        float lambda=inp.getLambda_t();
        int Num_trial=inp.getNumTrial();
        const int p_num = regmode.get_p_num();
        cl::ImageFormat format(CL_RG,CL_FLOAT);
        
        cl::Buffer dark_buffer=CLO.dark_buffer;
        cl::Buffer I0_target_buffer=CLO.I0_target_buffer;
        vector<cl::Buffer> I0_sample_buffers=CLO.I0_sample_buffers;
        
        // p_vec, p_err_vec, mt_sample
        vector<float*>p_vec;
        vector<float*>p_err_vec;
        /*float* p_vec_pointer;
        float* p_err_vec_pointer;
        p_vec_pointer = new float[p_num*(endEnergyNo-startEnergyNo+1)];
        p_err_vec_pointer = new float[p_num*(endEnergyNo-startEnergyNo+1)];
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            p_vec.push_back(&p_vec_pointer[p_num*(i-startEnergyNo)]);
            p_err_vec.push_back(&p_err_vec_pointer[p_num*(i-startEnergyNo)]);
            mt_sample_img.push_back(new float[IMAGE_SIZE_M]);
        }*/
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            p_vec.push_back(new float[p_num]);
            p_err_vec.push_back(new float[p_num]);
        }
        
        ifstream ini_ifs(inp.getIniFilePath(),ios::in);
        if (ini_ifs) {
            char *buffer;
            buffer = new char[buffsize];
            ini_ifs.getline(buffer, buffsize);
            int en=1;
            //cout<<buffer<<endl;
            while (!ini_ifs.eof()) {
                ini_ifs.getline(buffer, buffsize);
                if ((en>=startEnergyNo)&&(en<=endEnergyNo)) {
                    istringstream iss(buffer);
                    float a;
                    int j=0;
                    for (iss>>a; !iss.eof(); iss.ignore()>>a) {
                        p_vec[en-startEnergyNo][j]=a;
                        //cout<<p_vec[en-startEnergyNo][j]<<",";
                        j++;
                    }
                    p_vec[en-startEnergyNo][j]=a;
                    //cout<<p_vec[en-startEnergyNo][j];
                    j++;
                    
                    while (j<p_num) {
                        p_vec[en-startEnergyNo][j]=0.0f;
                        //cout<<","<<p_vec[en-startEnergyNo][j];
                        j++;
                    }
                }
                //cout<<endl;
                en++;
            }
        }
        ini_ifs.close();
        
    
        //Buffer declaration
        cl::Buffer mt_target_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
        cl::Buffer mt_sample_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
        vector<cl::Image2DArray> mt_target_image;
        vector<cl::Image2DArray> mt_sample_image;
        for (int i=0; i<4; i++) {
            int mergeN = 1<<i;
            mt_target_image.push_back(cl::Image2DArray(context, CL_MEM_READ_WRITE,format,1,
                                                       imageSizeX/mergeN,imageSizeY/mergeN,
                                                       0,0,NULL,NULL));
            mt_sample_image.push_back(cl::Image2DArray(context, CL_MEM_READ_WRITE,format,1,
                                                       imageSizeX/mergeN,imageSizeY/mergeN,
                                                       0,0,NULL,NULL));
        }
        cl::Buffer p_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*p_num, 0, NULL);
        cl::Buffer p_err_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*p_num, 0, NULL);
        cl::Buffer p_target_buffer(context, CL_MEM_READ_ONLY, sizeof(cl_float)*p_num, 0, NULL);
        cl::Buffer p_fix_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*p_num, 0, NULL);
        cl::Buffer lambda_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float), 0, NULL);
        cl::Image2DArray mt_target_outputImg(context, CL_MEM_READ_WRITE,format,1,imageSizeX,imageSizeY,0,0,NULL,NULL);
        cl::Image2DArray mt_sample_outputImg(context, CL_MEM_READ_WRITE,format,1,imageSizeX,imageSizeY,0,0,NULL,NULL);
        
        
        //kernel dimension declaration
        const cl::NDRange global_item_size(WorkGroupSize,imageSizeY,1);
        const cl::NDRange local_item_size(WorkGroupSize,1,1);
        
        
        //Energy loop setting
        vector<int> LoopEndenergyNo={startEnergyNo,endEnergyNo};
        vector<int> LoopStartenergyNo;
        LoopStartenergyNo.push_back(min(targetEnergyNo, endEnergyNo));
        LoopStartenergyNo.push_back(max(targetEnergyNo, startEnergyNo));
        
        
        //target mt input
        string inputfilepath = inp.getInputDir(); //inputDirだが実際はファイルパス
        float* mt_target;
        mt_target=new float[imageSizeM];
        readRawFile(inputfilepath,mt_target,targetEnergyNo,targetEnergyNo,imageSizeM);
        cl::Kernel kernel_trans = CLO.getKernel("mt_transfer");
        mt_transfer(command_queue,kernel_trans,
                    mt_target_buffer,mt_target_image[0], mt_target_outputImg,
                    global_item_size,local_item_size,mt_target,msk,true,imageSizeM);
        
        // Sample mt data input
        float *mt_sample;
        mt_sample=new float[imageSizeM*dE];
        readRawFile(inputfilepath,mt_sample,startEnergyNo,endEnergyNo,imageSizeM);
        
        //target image reg parameter (p_target_buffer) initialize
        ifstream ini_ifs2(inp.getIniFilePath(),ios::in);
        if (!ini_ifs2) {
            command_queue.enqueueWriteBuffer(p_target_buffer, CL_TRUE, 0, sizeof(cl_float)*p_num,regmode.p_ini,NULL,NULL);
        }else{
            command_queue.enqueueWriteBuffer(p_target_buffer, CL_TRUE, 0, sizeof(cl_float)*p_num,p_vec[targetEnergyNo-startEnergyNo],NULL,NULL);
        }
        command_queue.enqueueWriteBuffer(p_fix_buffer, CL_TRUE, 0, sizeof(cl_float)*p_num,regmode.p_fix,NULL,NULL);
        command_queue.finish();
        delete [] mt_target;
        
        //It_target merged image create
        cl::Kernel kernel_merge = CLO.getKernel("merge");
        if (regmode.get_regModeNo()>=0) {
            for (int i=3; i>0; i--) {
                int mergeN = 1<<i;
                int localsize = min((int)WorkGroupSize,imageSizeX/mergeN);
                const cl::NDRange global_item_size_merge(localsize,1,1);
                const cl::NDRange local_item_size_merge(localsize,1,1);
                
                kernel_merge.setArg(0, mt_target_image[0]);
                kernel_merge.setArg(1, mt_target_image[i]);
                kernel_merge.setArg(2, mergeN);
                command_queue.enqueueNDRangeKernel(kernel_merge, cl::NullRange, global_item_size_merge, local_item_size_merge, NULL, NULL);
            }
            command_queue.finish();
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
                command_queue.enqueueNDRangeKernel(kernel_output, NULL, global_item_size, local_item_size, NULL, NULL);
                command_queue.finish();
            }
            
            command_queue.enqueueReadBuffer(mt_target_buffer, CL_TRUE, 0, sizeof(cl_float)*imageSizeM, &mt_sample[imageSizeM*(targetEnergyNo - startEnergyNo)], NULL, NULL);
            command_queue.finish();
            
            command_queue.enqueueReadBuffer(p_target_buffer,CL_TRUE,0,sizeof(cl_float)*p_num,p_vec[targetEnergyNo-startEnergyNo],NULL,NULL);
            command_queue.finish();
            ostringstream oss;
            for (int t=0; t<p_num; t++) {
                //p_vec[targetEnergyNo - startEnergyNo][t+p_num*(j-startAngleNo)]=0;
                p_err_vec[targetEnergyNo - startEnergyNo][t]=0;
            }
                
            oss << "Device: "<< devicename << ", energy: "<<targetEnergyNo<<endl;
            if(!ini_ifs2) oss <<regmode.get_oss_target();
            else oss <<regmode.get_oss_target(p_vec[targetEnergyNo-startEnergyNo]);
            cout << oss.str();
        }
        
        
        //kernel setArgs of It_sample merged image create
        kernel_merge.setArg(0, mt_sample_image[0]);
        
        //kernel setArgs of Image registration
        cl::Kernel kernel_imgReg = CLO.getKernel("ImageRegistration");
        kernel_imgReg.setArg(2, lambda_buffer);
        kernel_imgReg.setArg(3, p_buffer);
        kernel_imgReg.setArg(4, p_err_buffer);
        kernel_imgReg.setArg(5, p_target_buffer);
        kernel_imgReg.setArg(6, p_fix_buffer);
        kernel_imgReg.setArg(9, 1.0f);
        
        //kernel setArgs of outputing image reg results to buffer
        kernel_output.setArg(0, mt_sample_outputImg);
        kernel_output.setArg(1, mt_sample_buffer);
        kernel_output.setArg(2, p_buffer);
        
        
        for (int s=0; s<2; s++) {//1st cycle:i<targetEnergyNo, 2nd cycle:i>targetEnergyNo
            int di=(-1+2*s);
            if (startEnergyNo==endEnergyNo){
                if(startEnergyNo==targetEnergyNo) break;
            }else if ((LoopStartenergyNo[s]+di)*di>LoopEndenergyNo[s]*di) {
                continue;
            }
            
            //transpara reset
            if (!ini_ifs2) {
                command_queue.enqueueFillBuffer(p_buffer, (cl_float)0.0, 0, sizeof(cl_float)*p_num,NULL,NULL);
            }else{
                command_queue.enqueueWriteBuffer(p_buffer, CL_TRUE, 0, sizeof(cl_float)*p_num,p_vec[targetEnergyNo-startEnergyNo],NULL,NULL);
            }
            command_queue.finish();
            
            int ds = (LoopStartenergyNo[s]==targetEnergyNo) ? di:0;
            for (int i=LoopStartenergyNo[s]+ds; i*di<=LoopEndenergyNo[s]*di; i+=di) {
                
                //set transpara
                if (ini_ifs2) {
                    float *p_dummy;
                    p_dummy=new float[p_num];
                    command_queue.enqueueReadBuffer(p_buffer, CL_TRUE, 0, sizeof(cl_float)*p_num,p_dummy,NULL,NULL);
                    for(int t=0;t<p_num;t++){
                        //cout<<"p_vec "<<p_vec[i-startEnergyNo][t]<<" ";
                        p_dummy[t] = p_dummy[t]*regmode.p_fix[t]+p_vec[i-startEnergyNo][t]*(1.0f-regmode.p_fix[t]);
                        //cout << p_dummy[t]<<" ";
                    }
                    //cout<<endl;
                    command_queue.enqueueWriteBuffer(p_buffer, CL_TRUE, 0, sizeof(cl_float)*p_num,p_dummy,NULL,NULL);
                    delete [] p_dummy;
                }
                
                
                //sample mt conversion
                mt_transfer(command_queue,kernel_trans,
                             mt_sample_buffer,mt_sample_image[0], mt_sample_outputImg,
                             global_item_size,local_item_size,&mt_sample[imageSizeM*(i-startEnergyNo)],msk,false,imageSizeM);
                
                
                if(regmode.get_regModeNo()>=0){
                    //mt_sample merged image create
                    for (int i=3; i>0; i--) {
                        unsigned int mergeN = 1<<i;
                        unsigned int localsize = min((unsigned int)WorkGroupSize,imageSizeX/mergeN);
                        const cl::NDRange global_item_size_merge(localsize,1,1);
                        const cl::NDRange local_item_size_merge(localsize,1,1);
                        
                        kernel_merge.setArg(1, mt_sample_image[i]);
                        kernel_merge.setArg(2, mergeN);
                        command_queue.enqueueNDRangeKernel(kernel_merge, cl::NullRange, global_item_size_merge, local_item_size_merge, NULL, NULL);
                        command_queue.finish();
                    }
                    
                    
                    //lambda_buffer reset
                    command_queue.enqueueFillBuffer(lambda_buffer, (cl_float)lambda, 0, sizeof(cl_float),NULL,NULL);
                    command_queue.finish();
                    
                    
                    //Image registration
                    for (int i=0; i>=0; i--) {
                        unsigned int mergeN = 1<<i;
                        //cout<<mergeN<<endl;
                        unsigned int localsize = min((unsigned int)WorkGroupSize,imageSizeX/mergeN);
                        const cl::NDRange global_item_size_merge(localsize,1,1);
                        const cl::NDRange local_item_size_merge(localsize,1,1);
                        
                        kernel_imgReg.setArg(0, mt_target_image[i]);
                        kernel_imgReg.setArg(1, mt_sample_image[i]);
                        kernel_imgReg.setArg(7, cl::Local(sizeof(cl_float)*localsize));//locmem
                        kernel_imgReg.setArg(8, mergeN);
                        for (int trial=0; trial < Num_trial; trial++) {
                            command_queue.enqueueNDRangeKernel(kernel_imgReg, NULL, global_item_size_merge, local_item_size_merge, NULL, NULL);
                            command_queue.finish();
                        }
                    }
                    
                    
                    //output image reg results to buffer
                    command_queue.enqueueNDRangeKernel(kernel_output, NULL, global_item_size, local_item_size, NULL, NULL);
                    command_queue.finish();
                    
                    
                    //transpara read buffer
                    command_queue.enqueueReadBuffer(p_buffer,CL_TRUE,0,sizeof(cl_float)*p_num,p_vec[i-startEnergyNo],NULL,NULL);
                    command_queue.enqueueReadBuffer(p_err_buffer,CL_TRUE,0,sizeof(cl_float)*p_num,p_err_vec[i-startEnergyNo],NULL,NULL);
                    command_queue.finish();
                }
                
                
                ostringstream oss;
                int *p_precision, *p_err_precision;
                p_precision=new int[p_num];
                p_err_precision=new int[p_num];
                for (int n=0; n<p_num; n++) {
                    int a = floor(log10(abs(p_vec[i-startEnergyNo][n])));
                    int b = floor(log10(abs(p_err_vec[i-startEnergyNo][n])));
                    p_err_precision[n] = max(0,b)+1;
                        
                        
                    if(regmode.p_fix[n]==0.0f) p_precision[n]=3;
                    else if (a>0) {
                        int c = floor(log10(pow(10,a+1)-0.5));
                        if(a>c) a++;
                        
                        p_precision[n] = a+1 - min(0,b);
                    }else if(a<b){
                        p_precision[n] = 1;
                    }else{
                        p_precision[n]= a - b + 1;
                    }
                }
                oss << "Device: "<< devicename << ", energy: "<<i<<endl;
                oss << regmode.oss_sample(p_vec[i-startEnergyNo],p_err_vec[i-startEnergyNo],p_precision,p_err_precision);

                cout << oss.str();
                
                //read output buffer
                command_queue.enqueueReadBuffer(mt_sample_buffer, CL_TRUE, 0, sizeof(cl_float)*imageSizeM, &mt_sample[imageSizeM*(i - startEnergyNo)], NULL, NULL);
                command_queue.finish();
            }
        }
        
        //delete [] mt_sample;
        mt_output_thread(startEnergyNo,endEnergyNo,
                         inp.getOutputDir(), move(mt_sample), move(p_vec),move(p_err_vec),regmode,imageSizeM);
        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
    
    
    return 0;
}




int imageRegistlation_2D_ocl(input_parameter inp, OCL_platform_device plat_dev_list, regMode regmode)
{
    int imageSizeX = inp.getImageSizeX();
    int imageSizeY = inp.getImageSizeY();
    
    //OpenCL objects class
    vector<CL_objects> CLO;
    CL_objects CLO_contx;
    CLO.push_back(CLO_contx);
    
    //OpenCL Program
    cl::Program program=regmode.buildImageRegProgram(plat_dev_list.context(0),imageSizeX, imageSizeY);
    CLO[0].addKernel(program,"mt_transfer");//0
    CLO[0].addKernel(program,"merge");//1
    CLO[0].addKernel(program,"imageRegistration");//2
    CLO[0].addKernel(program,"output_imgReg_result");//3
    CLO[0].addKernel(program,"merge_rawhisdata");//4
    
    
    //display OCL device
    int maxWorkSize;
    string platform_param;
    plat_dev_list.plat(0).getInfo(CL_PLATFORM_NAME, &platform_param);
    cout << "CL PLATFORM NAME: "<< platform_param<<endl;
    plat_dev_list.plat(0).getInfo(CL_PLATFORM_VERSION, &platform_param);
    cout << "   "<<platform_param<<endl;
    string device_pram;
    plat_dev_list.dev(0,0).getInfo(CL_DEVICE_NAME, &device_pram);
    cout << "CL DEVICE NAME: "<< device_pram<<endl;
    //working compute unit
    maxWorkSize=(int)min((int)plat_dev_list.dev(0,0).getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), imageSizeX);
    
    
    //mask settings
    mask msk(inp);
    
    
    //queueを通し番号に変換
    cl::CommandQueue queues = plat_dev_list.queue(0,0);
    
    
    //start threads
    imageReg_2D_thread(queues, CLO[0], inp, regmode, msk);



    return 0;
}
