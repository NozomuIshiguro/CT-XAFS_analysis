//
//  reslice_ocl.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/06/22.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "reslice.hpp"
#include "reslice_cl.hpp"


int prj_output_thread(int startZ,int EndZ,int EnergyNo,int num_angle,
                      string output_dir,string fileName_base,
                      vector<float*> prj_vec){
    
    //output resliced data
    ostringstream oss;
    for (int i=startZ; i<=EndZ; i++) {
        string fileName_output= output_dir+ EnumTagString(EnergyNo,"/","/")+AnumTagString(i,fileName_base, ".raw");
        oss << "output file: " << fileName_output << "\n";
        outputRawFile_stream(fileName_output,prj_vec[i-startZ],(size_t)(IMAGE_SIZE_X*num_angle));
    }
    oss <<endl;
    cout << oss.str();
    
    //delete pointer
    for (int i=startZ; i<=EndZ; i++) {
        delete [] prj_vec[i-startZ];
    }
    return 0;
}

int xyshift_output_thread(int num_angle,string output_dir,float *xshift, float *zshift){
    
    //output x/z correction value
    string fileName_output= output_dir+ "/xyshift";
    ofstream ofs(fileName_output,ios::out|ios::trunc);
    ofs<<"Angle No."<<"\t"<<"x shift"<<"\t"<<"z shift"<<endl;
    for (int i=0; i<num_angle; i++) {
        ofs<<i+1<<"\t"<<xshift[i]<<"\t"<<zshift[i]<<endl;
    }
    ofs.close();
    delete [] xshift;
    delete [] zshift;
    return 0;
}

int reslice_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                   int startAngleNo, int EndAngleNo,int EnergyNo,float baseup,
                   string input_dir, string output_dir,
                   string fileName_base_i,string fileName_base_o,
                   int startX, int endX, int startZ, int endZ,
                   bool Zcorr, bool Xcorr, bool last)
{
    try {
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        string devicename = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>();
        size_t maxWorkGroupSize = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        size_t globalMemSize = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
        
        const cl::NDRange local_item_size(maxWorkGroupSize,1,1);
        const cl::NDRange global_item_size(IMAGE_SIZE_X,1,1);
        
        int num_angle = EndAngleNo - startAngleNo + 1;
        int iter=1;
        int64_t memobj = sizeof(float)*((int64_t)IMAGE_SIZE_M*num_angle/iter*2+num_angle*2);
        while (memobj>globalMemSize) {
            iter*=2;
            memobj = sizeof(float)*((int64_t)IMAGE_SIZE_M*num_angle/iter*2+num_angle*2);
        }
        
        
        cout<<"Device: "<<devicename<<endl<<"Processing energy No. "<<EnergyNo<<"..."<<endl<<"   Iteration: "<<iter<<endl<<endl;
        
        cl::ImageFormat format(CL_R, CL_FLOAT);
        cl::Image2DArray mt_img(context, CL_MEM_READ_WRITE,format,num_angle/iter,IMAGE_SIZE_X, IMAGE_SIZE_Y, 0, 0, NULL, NULL);
        cl::Image2DArray prj_img(context, CL_MEM_READ_WRITE,format,IMAGE_SIZE_Y,IMAGE_SIZE_X, num_angle/iter, 0, 0, NULL, NULL);
        cl::Buffer xshift_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*num_angle, 0, NULL);
        cl::Buffer zshift_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*num_angle, 0, NULL);
        vector<float*> mt_vec;
        vector<float*> prj_vec;
        float *xshift, *zshift;
        xshift = new float[num_angle];
        zshift = new float[num_angle];
        
        cl::size_t<3> origin;
        cl::size_t<3> region_mt;
        cl::size_t<3> region_prj;
        cl::size_t<3> region_xprj;
        origin[0] = 0;
        origin[1] = 0;
        region_mt[0] = IMAGE_SIZE_X;
        region_mt[1] = IMAGE_SIZE_Y;
        region_mt[2] = 1;
        region_prj[0] = IMAGE_SIZE_X;
        region_prj[1] = num_angle/iter;
        region_prj[2] = 1;
        region_xprj[0] = IMAGE_SIZE_X;
        region_xprj[1] = num_angle;
        region_xprj[2] = 1;
        
        //input mt data
        for (int i=startAngleNo; i<=EndAngleNo; i++) {
            mt_vec.push_back(new float[IMAGE_SIZE_M]);
            
            string filepath_input = input_dir+EnumTagString(EnergyNo,"/",fileName_base_i)+AnumTagString(i, "", ".raw");
            //cout<<filepath_input;
            readRawFile(filepath_input,mt_vec[i-startAngleNo]);
        }
        
        //reserve pointer for resliced projection
        for (int i = 0; i < IMAGE_SIZE_Y; i++) {
            prj_vec.push_back(new float[IMAGE_SIZE_X*num_angle]);
        }
        
        command_queue.enqueueFillBuffer(xshift_buff, (cl_float)0.0, 0, sizeof(float)*num_angle);
        command_queue.enqueueFillBuffer(zshift_buff, (cl_float)0.0, 0, sizeof(float)*num_angle);
        
        //zcorrection
        if(Zcorr){
            cout<< "Device: "<< devicename<<endl;
            cout<< "   Z-correction"<<endl<<endl;
            cl::Buffer xprj_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_Y*num_angle, 0, NULL);
            cl::Image2D xprj_img(context, CL_MEM_READ_WRITE,format,IMAGE_SIZE_X,num_angle, 0, NULL, NULL);
            
            const cl::NDRange local_item_size_xprj(maxWorkGroupSize,1,1);
            const cl::NDRange global_item_size_xprj(IMAGE_SIZE_Y,num_angle/iter,1);
            
            for (int j=0; j<iter; j++) {
                //write input mt_img object
                for (int i=0; i<num_angle/iter; i++) {
                    origin[2] = i;
                    command_queue.enqueueWriteImage(mt_img, CL_TRUE, origin, region_mt, IMAGE_SIZE_X*sizeof(float), 0, mt_vec[i+j*num_angle/iter], NULL, NULL);
                }
                
                //xprojection
                kernel[1].setArg(0, mt_img);
                kernel[1].setArg(1, xprj_buff);
                kernel[1].setArg(2, startX);
                kernel[1].setArg(3, endX);
                kernel[1].setArg(4, j);
                command_queue.enqueueNDRangeKernel(kernel[1], NULL, global_item_size_xprj, local_item_size_xprj, NULL, NULL);
                command_queue.finish();
                
            }
            origin[2] =0;
            command_queue.enqueueCopyBufferToImage(xprj_buff, xprj_img, 0, origin, region_xprj,NULL,NULL);
            
            //zcorrection
            const cl::NDRange local_item_size_zcorr(maxWorkGroupSize,1,1);
            const cl::NDRange global_item_size_zcorr(maxWorkGroupSize*num_angle,1,1);
            
            kernel[2].setArg(0, xprj_img);
            kernel[2].setArg(1, zshift_buff);
            kernel[2].setArg(2, cl::Local(sizeof(cl_float)*IMAGE_SIZE_Y)); //target_xproj
            kernel[2].setArg(3, cl::Local(sizeof(cl_float)*maxWorkGroupSize)); //loc_mem
            kernel[2].setArg(4, startZ);
            kernel[2].setArg(5, endZ);
            
            command_queue.enqueueNDRangeKernel(kernel[2], NULL, global_item_size_zcorr, local_item_size_zcorr, NULL, NULL);
            
        }
        
        /*float *zshift;
        zshift = new float[num_angle];
        command_queue.enqueueReadBuffer(zshift_buff, CL_TRUE, 0, sizeof(cl_float)*num_angle,zshift);
        for(int i=0;i<num_angle;i++){
            cout<<i<<":"<<zshift[i]<<endl;
        }*/
        
        //reslice
        cout<< "Device: "<< devicename<<endl;
        cout<< "    reslice to sinogram"<<endl<<endl;
        for (int j=0; j<iter; j++) {
            //write input mt_img object
            for (int i=0; i<num_angle/iter; i++) {
                origin[2] = i;
                command_queue.enqueueWriteImage(mt_img, CL_TRUE, origin, region_mt, IMAGE_SIZE_X*sizeof(float), 0, mt_vec[i+j*num_angle/iter], NULL, NULL);
            }
            
            //reslice
            kernel[0].setArg(0, mt_img);
            kernel[0].setArg(1, prj_img);
            kernel[0].setArg(2, xshift_buff);
            kernel[0].setArg(3, zshift_buff);
            kernel[0].setArg(4, iter);
            kernel[0].setArg(5, j);
            kernel[0].setArg(6, baseup);
            command_queue.enqueueNDRangeKernel(kernel[0], NULL, global_item_size, local_item_size, NULL, NULL);
            command_queue.finish();
            
            
            for (int i = 0; i < IMAGE_SIZE_Y; i++) {
                origin[2]=i;
                command_queue.enqueueReadImage(prj_img, CL_TRUE, origin, region_prj, IMAGE_SIZE_X*sizeof(float), 0, prj_vec[i]+j*num_angle*IMAGE_SIZE_X/iter, NULL, NULL);
            }
        }
        
        command_queue.enqueueReadBuffer(xshift_buff, CL_TRUE, 0, sizeof(cl_float)*num_angle, xshift);
        command_queue.enqueueReadBuffer(zshift_buff, CL_TRUE, 0, sizeof(cl_float)*num_angle, zshift);
        
        
        for (int i=startAngleNo; i<=EndAngleNo; i++) {
            delete [] mt_vec[i-startAngleNo];
        }
        
        if (Zcorr||Xcorr) {
            thread th_shift_output(xyshift_output_thread,num_angle,output_dir,
                                   move(xshift), move(zshift));
            th_shift_output.detach();
        }
        
        thread th_output(prj_output_thread,1,IMAGE_SIZE_Y,EnergyNo, num_angle,
                         output_dir,fileName_base_o,
                         move(prj_vec));
        if(last) th_output.join();
        else th_output.detach();
        
        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
    
    return 0;
}

int reslice_ocl(input_parameter inp,OCL_platform_device plat_dev_list,string fileName_base_i)
{
    cl_int ret;
    float startEnergyNo=inp.getStartEnergyNo();
    float endEnergyNo=inp.getEndEnergyNo();
    int startAngleNo=inp.getStartAngleNo();
    int endAngleNo=inp.getEndAngleNo();
    float baseup = inp.getBaseup();
    baseup = (baseup==NAN) ? 0:baseup;
    string fileName_base_o = inp.getOutputFileBase();
    
    //create output dir
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
        string fileName_output = inp.getOutputDir() + "/" + EnumTagString(i,"","");
        MKDIR(fileName_output.c_str());
    }
    

    //kernel program source
    /*ifstream ifs("./reslice.cl", ios::in);
    if(!ifs) {
        cerr << "   Failed to load kernel" << endl;
        return -1;
    }
    istreambuf_iterator<char> it(ifs);
    istreambuf_iterator<char> last;
    string kernel_src(it,last);
    ifs.close();*/
    
    
    //OpenCL Program
    vector<vector<cl::Kernel>> kernels;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        //OpenCL Program
        string kernel_code = "";
        ostringstream oss;
        oss<<"#define PRJ_ANGLESIZE "<<endAngleNo-startAngleNo+1<<endl<<endl;
        oss<<"#define NUM_TRIAL "<<20<<endl<<endl;
        oss<<"#define LAMBDA "<<0.001f<<endl<<endl;
        kernel_code += oss.str();
        kernel_code += kernel_src;
        //cout << kernel_code<<endl;
        size_t kernel_code_size = kernel_code.length();
        cl::Program::Sources source(1,std::make_pair(kernel_code.c_str(),kernel_code_size));
        cl::Program program(plat_dev_list.context(i), source,&ret);
        //kernel build
        ret=program.build();
        //cout<<ret<<endl;
        vector<cl::Kernel> kernels_plat;
        kernels_plat.push_back(cl::Kernel::Kernel(program,"reslice", &ret));//0
        kernels_plat.push_back(cl::Kernel::Kernel(program,"xprojection", &ret));//0
        kernels_plat.push_back(cl::Kernel::Kernel(program,"zcorrection", &ret));//0
        kernels.push_back(kernels_plat);
    }
    
    
    //display OCL device
    int t=0;
    vector<int> dA;
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
            cout << "CL DEVICE NAME: "<< device_pram<<endl<<endl;
            
            t++;
        }
    }
    
    
    //queueを通し番号に変換
    /*vector<cl::CommandQueue> queues;
    vector<int> cotextID_OfQueue;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        for (int j=0; j<plat_dev_list.queuesize(i); j++) {
            queues.push_back(plat_dev_list.queue(i,j));
            cotextID_OfQueue.push_back(i);
        }
    }*/
    
    
    //start thread
    vector<thread> th;
    for (int i=startEnergyNo; i<=endEnergyNo;) {
        for (int j=0; j<plat_dev_list.contextsize(); j++) {
            if (th.size()<plat_dev_list.contextsize()) {
                bool last = (i>endEnergyNo-plat_dev_list.contextsize());
                th.push_back(thread(reslice_thread,
                                    plat_dev_list.queue(j, 0),kernels[j],
                                    startAngleNo,endAngleNo,i,baseup,
                                    inp.getInputDir(),inp.getOutputDir(),
                                    fileName_base_i,fileName_base_o,
                                    inp.getStartX(),inp.getEndX(),
                                    inp.getStartZ(),inp.getEndZ(),
                                    inp.getZcorr(),inp.getXcorr(),last));
                i++;
                if (i > endEnergyNo) break;
                else continue;
            }else if (th[j].joinable()) {
                bool last = (i>endEnergyNo-plat_dev_list.contextsize());
                th[j].join();
                th[j] = thread(reslice_thread,
                               plat_dev_list.queue(j, 0),kernels[j],
                               startAngleNo,endAngleNo,i,baseup,
                               inp.getInputDir(),inp.getOutputDir(),
                               fileName_base_i,fileName_base_o,
                               inp.getStartX(),inp.getEndX(),
                               inp.getStartZ(),inp.getEndZ(),
                               inp.getZcorr(),inp.getXcorr(),last);
                i++;
                if (i > endEnergyNo) break;
            } else{
                this_thread::sleep_for(chrono::seconds(1));
            }
            
        }
        if (i > endAngleNo) break;
    }
    
    for (int j=0; j<plat_dev_list.contextsize(); j++) {
        if (th[j].joinable()) th[j].join();
    }
    
    return 0;
}