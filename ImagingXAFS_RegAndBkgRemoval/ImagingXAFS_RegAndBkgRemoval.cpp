//
//  ImagingXAFS_RegAndBkgRemoval.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2017/05/19.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#include "ImagingXAFS_RegAndBkgRemoval.hpp"

int imXAFSRegAndBkg_thread(cl::CommandQueue command_queue, CL_objects CLO,
                           float* mt_img_sample, int angleN,
                           input_parameter inp, regMode regmode, mask msk, int thread_id){
    try {
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
        string devicename = device.getInfo<CL_DEVICE_NAME>();
        size_t WorkGroupSize = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), IMAGE_SIZE_X);
        const int p_num = regmode.get_p_num()+regmode.get_cp_num();
        cl::ImageFormat format1(CL_R,CL_FLOAT);
        cl::ImageFormat format2(CL_RG,CL_FLOAT);
        int startEnergyNo=inp.getStartEnergyNo();
        int endEnergyNo=inp.getEndEnergyNo();
        int Enum = (endEnergyNo-startEnergyNo);
        int targetEnergyNo=inp.getTargetEnergyNo();
        float lambda=inp.getLambda_t();
        int Num_trial=inp.getNumTrial();
        
        vector<cl::Kernel> kernel=CLO.kernels;
        cl::Image2D dummy(context, CL_MEM_READ_WRITE,format1,IMAGE_SIZE_X,IMAGE_SIZE_Y,
                          0,NULL,NULL);
        cl::Image2D bkg_img(context, CL_MEM_READ_WRITE,format2,IMAGE_SIZE_X,IMAGE_SIZE_Y,
                          0,NULL,NULL);
        cl::Image2D grid_img(context, CL_MEM_READ_WRITE,format2,IMAGE_SIZE_X,IMAGE_SIZE_Y,
                            0,NULL,NULL);
        cl::Image2D sample_img(context, CL_MEM_READ_WRITE,format2,IMAGE_SIZE_X,IMAGE_SIZE_Y,
                            0,NULL,NULL);
        cl::Image2DArray mt_data_img(context, CL_MEM_READ_WRITE, format2, Enum,
                         IMAGE_SIZE_X,IMAGE_SIZE_Y, 0,0, NULL,NULL);
        cl::Image2DArray residue_img1(context, CL_MEM_READ_WRITE, format2, Enum,
                                     IMAGE_SIZE_X,IMAGE_SIZE_Y, 0,0, NULL,NULL);
        cl::Image2DArray residue_img2(context, CL_MEM_READ_WRITE, format2, Enum,
                                     IMAGE_SIZE_X,IMAGE_SIZE_Y, 0,0, NULL,NULL);
        cl::Buffer p_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*p_num*Enum, 0, NULL);
        cl::Buffer p_err_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*p_num*Enum, 0, NULL);
        cl::Buffer p_fix_buffer(context, CL_MEM_READ_ONLY, sizeof(cl_float)*p_num, 0, NULL);
        cl::Buffer p_target_buffer(context, CL_MEM_READ_ONLY, sizeof(cl_float)*p_num, 0, NULL);
        cl::Buffer lambda_buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float), 0, NULL);
        cl::Buffer cnt_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*Enum, 0, NULL);
        cl::Buffer wgt_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*Enum, 0, NULL);
        cl::Buffer sum_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_Y*Enum, 0, NULL);
        
        const cl::NDRange global_item_size_1(IMAGE_SIZE_X,IMAGE_SIZE_Y,1);
        const cl::NDRange global_item_size_2(IMAGE_SIZE_X,IMAGE_SIZE_Y,Enum);
        const cl::NDRange global_item_size_3(WorkGroupSize,IMAGE_SIZE_Y,Enum);
        const cl::NDRange global_item_size_4(WorkGroupSize,Enum,1);
        const cl::NDRange local_item_size(WorkGroupSize,1,1);
        
        
        //initialize bkg_img, grid_img, sample_img
        cl_float4 color = {0.0f,0.0f,0.0f,1.0f};
        cl::size_t<3> origin0;
        cl::size_t<3> region0;
        origin0[0] = 0;
        origin0[1] = 0;
        origin0[2] = 0;
        region0[0] = IMAGE_SIZE_X;
        region0[1] = IMAGE_SIZE_Y;
        region0[2] = 1;
        command_queue.enqueueFillImage(bkg_img, color, origin0, region0);
        command_queue.enqueueFillImage(grid_img, color, origin0, region0);
        command_queue.enqueueFillImage(sample_img, color, origin0, region0);
        command_queue.enqueueFillBuffer(cnt_buffer, (cl_float)1.0f, 0, sizeof(cl_float)*Enum);
        command_queue.enqueueFillBuffer(wgt_buffer, (cl_float)1.0f, 0, sizeof(cl_float)*Enum);
        
        
        //transfer mt image to GPU
        int imageSizeX = inp.imageSizeX;
        int imageSizeY = inp.imageSizeY;
        cl::size_t<3> origin1;
        cl::size_t<3> region1;
        origin1[0] = (IMAGE_SIZE_X-imageSizeX)/2 ;
        origin1[1] = (IMAGE_SIZE_Y-imageSizeY)/2 ;
        origin1[2] = 0;
        region1[0] = imageSizeX;
        region1[1] = imageSizeY;
        region1[2] = 1;
        kernel[0].setArg(0, dummy);
        kernel[0].setArg(1, mt_data_img);
        kernel[0].setArg(2, 0);
        kernel[0].setArg(3, (cl_int)inp.getRefMask_shape());
        kernel[0].setArg(4, (cl_int)inp.getRefMask_x());
        kernel[0].setArg(5, (cl_int)inp.getRefMask_y());
        kernel[0].setArg(6, (cl_int)inp.getRefMask_width());
        kernel[0].setArg(7, (cl_int)inp.getRefMask_height());
        kernel[0].setArg(8, (cl_int)inp.getRefMask_angle());
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            origin1[2] = i-startEnergyNo;
            kernel[0].setArg(2, i-startEnergyNo);
            
            command_queue.enqueueReadImage(dummy, CL_TRUE, origin1, region1, imageSizeX, 0, mt_img_sample);
            command_queue.finish();
            
            command_queue.enqueueNDRangeKernel(kernel[0], cl::NullRange, global_item_size_1, local_item_size, NULL, NULL);
            command_queue.finish();
        }
        
        //set kernel args
        //estimate Bkg_image
        kernel[1].setArg(0, bkg_img);
        kernel[1].setArg(1, mt_data_img);
        kernel[1].setArg(2, grid_img);
        kernel[1].setArg(3, sample_img);
        kernel[1].setArg(4, p_buffer);
        kernel[1].setArg(5, cnt_buffer);
        kernel[1].setArg(6, wgt_buffer);
        //estimate residue img
        kernel[2].setArg(0, residue_img1);
        kernel[2].setArg(1, mt_data_img);
        kernel[2].setArg(2, bkg_img);
        //estimate grid_img
        kernel[3].setArg(0, grid_img);
        kernel[3].setArg(1, residue_img2);
        kernel[3].setArg(2, sample_img);
        kernel[3].setArg(3, cnt_buffer);
        kernel[3].setArg(4, wgt_buffer);
        //estimate sample_img
        kernel[4].setArg(0, sample_img);
        kernel[4].setArg(1, residue_img2);
        kernel[4].setArg(2, grid_img);
        //estimate contrast 1
        kernel[5].setArg(0, residue_img2);
        kernel[5].setArg(1, grid_img);
        kernel[5].setArg(2, sample_img);
        kernel[5].setArg(3, sum_buffer);
        kernel[5].setArg(4, cl::Local(sizeof(cl_float)*WorkGroupSize));
        //estimate contrast 2
        kernel[6].setArg(0, sum_buffer);
        kernel[6].setArg(1, cnt_buffer);
        kernel[6].setArg(2, cl::Local(sizeof(cl_float)*WorkGroupSize));
        cl::size_t<3> region2;
        region2[0] = IMAGE_SIZE_X;
        region2[1] = IMAGE_SIZE_Y;
        region2[2] = Enum;
        for (int i=0; i<1/*Num_trial*/; i++) {
            //estimate Bkg_image
            command_queue.enqueueNDRangeKernel(kernel[1], cl::NullRange, global_item_size_1, local_item_size, NULL, NULL);
            command_queue.finish();
            
            
            //estimate residue img
            command_queue.enqueueNDRangeKernel(kernel[2], cl::NullRange, global_item_size_2, local_item_size, NULL, NULL);
            command_queue.finish();
            
            
            /*image registration of residue img*/
            command_queue.enqueueCopyImage(residue_img1, residue_img2, origin0, origin0, region2);
            command_queue.finish();
            
            //estimate grid img
            command_queue.enqueueNDRangeKernel(kernel[3], cl::NullRange, global_item_size_1, local_item_size, NULL, NULL);
            command_queue.finish();
            
            //estimate sample img
            command_queue.enqueueNDRangeKernel(kernel[4], cl::NullRange, global_item_size_1, local_item_size, NULL, NULL);
            command_queue.finish();
            
            //estimate contrast 1
            command_queue.enqueueNDRangeKernel(kernel[5], cl::NullRange, global_item_size_3, local_item_size, NULL, NULL);
            command_queue.finish();
            
            //estimate contrast 2
            command_queue.enqueueNDRangeKernel(kernel[6], cl::NullRange, global_item_size_4, local_item_size, NULL, NULL);
            command_queue.finish();
        }
        
        float* bkg_data;
        bkg_data = new float[IMAGE_SIZE_M];
        command_queue.enqueueReadImage(bkg_img, CL_TRUE, origin0, region0, 0, 0, bkg_data);
        float* grid_data;
        grid_data = new float[IMAGE_SIZE_M];
        command_queue.enqueueReadImage(grid_img, CL_TRUE, origin0, region0, 0, 0, grid_data);
        float* sample_data;
        sample_data = new float[IMAGE_SIZE_M];
        command_queue.enqueueReadImage(sample_img, CL_TRUE, origin0, region0, 0, 0, sample_data);
        float* residue_data;
        residue_data = new float[IMAGE_SIZE_M*Enum];
        kernel[7].setArg(0, dummy);
        kernel[7].setArg(1, residue_img2);
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            kernel[7].setArg(2, i-startEnergyNo);
            command_queue.enqueueNDRangeKernel(kernel[7], cl::NullRange, global_item_size_1, local_item_size, NULL, NULL);
            command_queue.finish();
            
            command_queue.enqueueReadImage(dummy, CL_TRUE, origin0, region0, 0, 0, &residue_data[(i-startEnergyNo)*IMAGE_SIZE_M]);
        }
        string filename = inp.getOutputDir() + "/bkg_" + to_string(angleN);
        outputRawFile_stream(filename, bkg_data, IMAGE_SIZE_M);
        filename = inp.getOutputDir() + "/grid_" + to_string(angleN);
        outputRawFile_stream(filename, grid_data, IMAGE_SIZE_M);
        filename = inp.getOutputDir() + "/sample_" + to_string(angleN);
        outputRawFile_stream(filename, sample_data, IMAGE_SIZE_M);
        filename = inp.getOutputDir() + "/residue_" + to_string(angleN);
        outputRawFile_stream(filename, residue_data, IMAGE_SIZE_M*Enum);
        
    }catch (cl::Error ret) {
            cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
        
    return 0;
}


int imXAFSRegBkg_data_input_thread(int AngleN, int thread_id,
                                   cl::CommandQueue command_queue,CL_objects CLO,
                                   string fileName_base,input_parameter inp,
                                   regMode regmode, mask msk){

    //スレッドを待機/ロック
    m1.lock();
    
    int startEnergyNo=inp.getStartEnergyNo();
    int endEnergyNo=inp.getEndEnergyNo();
    int imageSizeM = inp.imageSizeM;
    
    float* mt_img_sample;
    mt_img_sample = new float[(int64_t)imageSizeM*(endEnergyNo-startEnergyNo+1)];
    ostringstream oss;
    oss << fileName_base <<"_" << AngleN <<".raw";
    string fpath = oss.str();
    readRawFile(fpath,mt_img_sample, startEnergyNo, endEnergyNo, imageSizeM);
    
    
    //スレッドをアンロック
    m1.unlock();
    
    //image_reg
    imageReg_th[thread_id].join();
    imageReg_th[thread_id] = thread(imXAFSRegAndBkg_thread,
                                    command_queue, CLO,move(mt_img_sample),
                                    AngleN, inp,regmode,msk,thread_id);
    
    
    return 0;
}

//dummy thread
static int dummy(){
    return 0;
}


int imXAFS_RegAndBkgRemoval_ocl(string fileName_base, input_parameter inp,
                              OCL_platform_device plat_dev_list, regMode regmode)
{
    
    int startEnergyNo=inp.getStartEnergyNo();
    int endEnergyNo=inp.getEndEnergyNo();
    int targetEnergyNo=inp.getTargetEnergyNo();
    int startAngleNo=1;//inp.getStartAngleNo();
    int endAngleNo=1;//inp.getEndAngleNo();
    
    
    //OpenCL objects class
    vector<CL_objects> CLO;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        CL_objects CLO_contx;
        CLO.push_back(CLO_contx);
    }
    
    
    //OpenCL Program
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        cl::Program program=regmode.buildImageRegProgram(plat_dev_list.context(i));
        CLO[i].kernels.push_back(cl::Kernel(program,"mt_conversion"));//0
        CLO[i].kernels.push_back(cl::Kernel(program,"merge"));//1
        CLO[i].kernels.push_back(cl::Kernel(program,"imageRegistration"));//2
        CLO[i].kernels.push_back(cl::Kernel(program,"output_imgReg_result"));//3
        CLO[i].kernels.push_back(cl::Kernel(program,"merge_rawhisdata"));//4
        CLO[i].kernels.push_back(cl::Kernel(program,"imQXAFS_smoothing"));//5
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
            dA.push_back(1/*min(inp.getNumParallel(),(int)device_pram_size[0])*/);
            cout<<"Number of working compute unit: "<<dA[t]<<"/"<<device_pram_size[0]<<endl<<endl;
            maxWorkSize.push_back((int)min((int)plat_dev_list.dev(i,j).getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), IMAGE_SIZE_X));
            t++;
        }
    }
    
    //mask settings
    mask msk(inp);
    
    for (int i = startAngleNo; i <= endAngleNo;) {
        for (int j = 0; j < plat_dev_list.contextsize(); j++) {
            if (input_th[j].joinable()) {
                input_th[j].join();
                input_th[j] = thread(imXAFSRegBkg_data_input_thread,i,j,
                                     plat_dev_list.queue(j, 0), CLO[j],
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
    
    
    return 0;
}
