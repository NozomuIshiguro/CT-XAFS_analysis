//
//  imageBatchBinning.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2017/07/18.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#include "imageBatchBinning.hpp"
#include "imageregistration_kernel_src_cl.hpp"

int output_img_thread(int startAngleNo, int EndAngleNo,
                     input_parameter inp,vector<float*> mt_outputs,int thread_id){
    
    //スレッドを待機/ロック
    m2.lock();
    string output_dir=inp.getOutputDir();
    string output_base=inp.getOutputFileBase();
    int startEnergyNo=inp.getStartEnergyNo();
    int endEnergyNo=inp.getEndEnergyNo();
    
    int mergeN=inp.getMergeN();
    int OutputImgSizeX = inp.getImageSizeX()/mergeN;
    int OutputImgSizeY = inp.getImageSizeY()/mergeN;
    int OutputImgSizeM = OutputImgSizeX*OutputImgSizeY;
    
    ostringstream oss;
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
        for (int j=startAngleNo; j<=EndAngleNo; j++) {
            string fileName_output= output_dir+ "/" + EnumTagString(i,"","/") + AnumTagString(j,output_base, ".raw");
            oss << "output file: " << fileName_output <<endl;
            outputRawFile_stream(fileName_output,&mt_outputs[i-startEnergyNo][(j-startAngleNo)*OutputImgSizeM],OutputImgSizeM);
        }
    }
    oss << endl;
    cout << oss.str();
    //スレッドをアンロック
    m2.unlock();
    
    
    //delete [] mt_outputs_pointer;
    for (int i = startEnergyNo; i <= endEnergyNo; i++) {
        delete[] mt_outputs[i - startEnergyNo];
    }
    
    return 0;
}

int binning_thread(cl::CommandQueue command_queue, CL_objects CLO,
                   vector<vector<float*>> input_img_data,
                   int startAngleNo,int EndAngleNo,
                   input_parameter inp, int thread_id){
    
    try {
        int mergeN=inp.getMergeN();
        int imageSizeX = inp.getImageSizeX();
        int imageSizeY = inp.getImageSizeY();
        int OutputImageSizeX = imageSizeX/mergeN;
        int OutputImageSizeY = imageSizeY/mergeN;
        int OutputImageSizeM = OutputImageSizeX*OutputImageSizeY;
        
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
        string devicename = device.getInfo<CL_DEVICE_NAME>();
        size_t WorkGroupSize = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), OutputImageSizeX);
        const int dA=EndAngleNo-startAngleNo+1;
        cl::ImageFormat format(CL_R,CL_FLOAT);
        int startEnergyNo=inp.getStartEnergyNo();
        int endEnergyNo=inp.getEndEnergyNo();
        int num_energy = endEnergyNo - startEnergyNo +1;
        
       
        
        //create output_image memory objects
        vector<float*>output_img_data;
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            output_img_data.push_back(new float[OutputImageSizeM*dA]);
        }
        
        //Buffer declaration
        cl::Image2DArray input_image(context, CL_MEM_READ_ONLY,format,dA,imageSizeX,imageSizeY,
                                     0,0,NULL,NULL);
        cl::Image2DArray output_image(context, CL_MEM_WRITE_ONLY,format,dA,OutputImageSizeX,
                                      OutputImageSizeY,0,0,NULL,NULL);
        cl::size_t<3> origin;
        cl::size_t<3> region;
        cl::size_t<3> region2;
        origin[0] = 0;
        origin[1] = 0;
        region[0] = imageSizeX;
        region[1] = imageSizeY;
        region[2] = 1;
        region2[0] = OutputImageSizeX;
        region2[1] = OutputImageSizeY;
        region2[2] = 1;
        
        
        //kernel dimension declaration
        const cl::NDRange global_item_size(WorkGroupSize*dA,OutputImageSizeY,1);
        const cl::NDRange local_item_size(WorkGroupSize,1,1);
        
        for (int i=0; i<num_energy; i++) {
            //write input image to GPU
            for (int k=0; k<dA; k++) {
                origin[2]=k;
                command_queue.enqueueWriteImage(input_image, CL_TRUE, origin, region, 0, 0, input_img_data[i][k]);
            }
            command_queue.finish();
            
            
            //merge process by GPU
            cl::Kernel kernel_merge = CLO.getKernel("merge");
            kernel_merge.setArg(0, input_image);
            kernel_merge.setArg(1, output_image);
            kernel_merge.setArg(2, mergeN);
            command_queue.enqueueNDRangeKernel(kernel_merge, cl::NullRange, global_item_size, local_item_size, NULL, NULL);
            command_queue.finish();
            
            //read output image from GPU
            for (int k=0; k<dA; k++) {
                origin[2]=k;
                command_queue.enqueueReadImage(output_image, CL_TRUE, origin, region2, 0, 0,
                                               &output_img_data[i][k*OutputImageSizeM]);
            }
            command_queue.finish();
        }
        
        for (int i = startEnergyNo; i <= endEnergyNo; i++) {
            for (int j=startAngleNo; j<=EndAngleNo; j++) {
                delete[] input_img_data[i-startEnergyNo][j-startAngleNo];
            }
        }
        
        output_th[thread_id].join();
        output_th[thread_id]=thread(output_img_thread,
                                    startAngleNo,EndAngleNo,inp,move(output_img_data),thread_id);
        
        
        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
    
    return 0;
    

}


int data_input_thread(int thread_id, cl::CommandQueue command_queue, CL_objects CLO,
                      int startAngleNo,int EndAngleNo, input_parameter inp){
    
    
    //スレッドを待機/ロック
    //m1.lock();
    
    int startEnergyNo=inp.getStartEnergyNo();
    int endEnergyNo=inp.getEndEnergyNo();
    
    ostringstream oss;
    if (startAngleNo == EndAngleNo) {
        oss << "(" << thread_id + 1 << ") reading image files of angle " << startAngleNo << endl << endl;
    }
    else {
        oss << "(" << thread_id + 1 << ") reading image files of angle " << startAngleNo << "-" << EndAngleNo << endl << endl;
    }
    cout << oss.str();
    
    string fileName_base = inp.getFittingFileBase();
    string input_dir=inp.getInputDir();
    vector<vector<float*>> mt_vec;
    vector<vector<string>> filepath_input;
    const int imgSizeM = inp.getImageSizeM();
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
        vector<float*> mt_vec_atE;
        vector<string> filepath_input_atE;
        for (int j=startAngleNo; j<=EndAngleNo; j++) {
            mt_vec_atE.push_back(new float[imgSizeM]);
            filepath_input_atE.push_back(input_dir + EnumTagString(i,"/",fileName_base) + AnumTagString(j,"",".raw"));
        }
        mt_vec.push_back(mt_vec_atE);
        filepath_input.push_back(filepath_input_atE);
    }
    
    //スレッドをアンロック
   // m1.unlock();
    //input mt data
    int num_energy = endEnergyNo -startEnergyNo +1;
    int num_angle = EndAngleNo -startAngleNo +1;
    for (int i=0; i<num_energy; i++) {
        for (int j=0; j<num_angle; j++) {
            readRawFile(filepath_input[i][j],mt_vec[i][j],imgSizeM);
        }
    }
    m1.unlock();
    
    //image_reg
    binning_th[thread_id].join();
    binning_th[thread_id] = thread(binning_thread,
                                   command_queue, CLO, move(mt_vec),
                                    startAngleNo, EndAngleNo, inp,thread_id);
    
    return 0;
}


//dummy thread
static int dummy(){
    return 0;
}


int imageBatchBinning_OCL(input_parameter inp,OCL_platform_device plat_dev_list){
    
    int startEnergyNo=inp.getStartEnergyNo();
    int endEnergyNo=inp.getEndEnergyNo();
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
    
    //int scanN = inp.getScanN();
    
    
    //OpenCL Program
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        cl_int ret;
        
        string option="-cl-fp32-correctly-rounded-divide-sqrt -cl-single-precision-constant ";
#if DEBUG
        option+="-D DEBUG ";
#endif
        string GPUvendor = plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0].getInfo<CL_DEVICE_VENDOR>();
        if(GPUvendor == "nvidia"){
            option += "-cl-nv-maxrregcount=64 ";
            //option += " -cl-nv-verbose -Werror";
        }else if (GPUvendor.find("NVIDIA")==0) {
            option += "-cl-nv-maxrregcount=64 ";
            //option += " -cl-nv-verbose -Werror";
        }
        
#if defined (OCL120)
        cl::Program::Sources source(1,std::make_pair(kernel_src.c_str(),kernel_src.length()));
#else
        cl::Program::Sources source(1,kernel_src);
#endif
        cl::Program program(plat_dev_list.context(i), source);
        //cout << kernel_code<<endl;
        //kernel build
        ret=program.build(option.c_str());
        string logstr=program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0]);
        cout << logstr << endl;
        CLO[i].addKernel(program,"merge");
    }
    
    
    //display OCL device
    int t=0;
    vector<int> dA;
    vector<int> maxWorkSize;
    const int imgSizeX = inp.getImageSizeX();
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
            maxWorkSize.push_back((int)min((int)plat_dev_list.dev(i,j).getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), imgSizeX));
            t++;
        }
    }
    
    //start threads
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        input_th.push_back(thread(dummy));
        binning_th.push_back(thread(dummy));
        output_th.push_back(thread(dummy));
    }
    for (int i=startAngleNo; i<=endAngleNo;) {
        for (int j = 0; j<plat_dev_list.contextsize(); j++) {
            if (input_th[j].joinable()) {
                input_th[j].join();
                input_th[j] = thread(data_input_thread,j,plat_dev_list.queue(j,0),CLO[j],
                                     i, min(i + dA[j] - 1, endAngleNo),inp);
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
        binning_th[j].join();
    }
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        output_th[j].join();
    }
    
    return 0;
}
