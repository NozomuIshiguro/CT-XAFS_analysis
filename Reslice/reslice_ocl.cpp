//
//  reslice_ocl.cpp
//  Reslice
//
//  Created by Nozomu Ishiguro on 2015/06/22.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "reslice.hpp"
#include "reslice_cl.hpp"
extern vector<thread> input_th, reslice_th, output_th1, output_th2;

int reslice_mtImg(OCL_platform_device plat_dev_list,cl::Kernel kernel,
                  vector<float*> mt_img_vec, vector<float*> prj_img_vec,
                  input_parameter inp){
    
    cl::Context context = plat_dev_list.context(0);
    cl::CommandQueue queue = plat_dev_list.queue(0, 0);
    string devicename = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>();
    size_t globalMemSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    
    int imageSizeX = inp.getImageSizeX();
    int imageSizeY = inp.getImageSizeY();
    int64_t imageSizeM = inp.getImageSizeM();
    
    
    int num_angle=inp.getEndAngleNo()-inp.getStartAngleNo()+1;
    int iter=1;
    size_t memobj = sizeof(float)*((int64_t)imageSizeM*(num_angle/iter+1)+num_angle*2);
    while (memobj>globalMemSize) {
        iter*=2;
        memobj = sizeof(float)*((int64_t)imageSizeM*(num_angle/iter+1)+num_angle*2);
    }
    
    
    cl::ImageFormat format(CL_R, CL_FLOAT);
    cl::Image2D mt_img(context, CL_MEM_READ_WRITE,format,imageSizeX, imageSizeY, 0, NULL, NULL);
    cl::Image2DArray prj_img(context, CL_MEM_READ_WRITE,format,imageSizeY,imageSizeX, num_angle/iter, 0, 0, NULL, NULL);
    cl::Buffer xshift_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*num_angle, 0, NULL);
    cl::Buffer zshift_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*num_angle, 0, NULL);
    queue.enqueueFillBuffer(xshift_buff, (cl_float)0.0, 0, sizeof(float)*num_angle);
    queue.enqueueFillBuffer(zshift_buff, (cl_float)0.0, 0, sizeof(float)*num_angle);
    
    
    reslice(queue,kernel,num_angle,iter,0,mt_img_vec,prj_img_vec,
            xshift_buff,zshift_buff,
            imageSizeX,imageSizeY);
    
    
    return 0;
}

int prj_output_thread(int startZ,int EndZ,string subDir_str,int num_angle,
                      string output_dir,string fileName_base,
                      vector<float*> prj_vec, int prjImageSize){
    
    //output resliced data
    ostringstream oss;
    for (int i=startZ; i<=EndZ; i++) {
        string fileName_output= output_dir+"/"+subDir_str+"/"+AnumTagString(i,fileName_base,".raw");
        oss << "output file: " << fileName_output << "\n";
        outputRawFile_stream(fileName_output,prj_vec[i-startZ],(size_t)(prjImageSize*num_angle));
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

int Z_Correction(cl::CommandQueue command_queue,
                 cl::Kernel kernel_xproj,cl::Kernel kernel_zcorr,
                 int num_angle,int startX, int endX, int startZ, int endZ,
                 vector<float*> mt_vec,cl::Buffer zshift_buff,
                 int imageSizeX, int imageSizeY){
    
    cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
    size_t maxWorkGroupSize = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    
    cl::ImageFormat format(CL_R, CL_FLOAT);
    cl::size_t<3> origin;
    cl::size_t<3> region_mt;
    cl::size_t<3> region_xprj;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    region_mt[0] = imageSizeX;
    region_mt[1] = imageSizeY;
    region_mt[2] = 1;
    region_xprj[0] = imageSizeX;
    region_xprj[1] = num_angle;
    region_xprj[2] = 1;
    
    //image object declaration
    cl::Image2D mt_img(context, CL_MEM_READ_WRITE,format,imageSizeX, imageSizeY, 0,NULL, NULL);
    cl::Buffer xprj_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeY*num_angle, 0, NULL);
    cl::Image2D xprj_img(context, CL_MEM_READ_WRITE,format,imageSizeX,num_angle, 0, NULL, NULL);
    
    const cl::NDRange local_item_size_xprj(min(imageSizeY,(int)maxWorkGroupSize),1,1);
    const cl::NDRange global_item_size_xprj(imageSizeY,1,1);
    
    //write input mt_img object
    for (int i=0; i<num_angle; i++) {
        command_queue.enqueueWriteImage(mt_img, CL_TRUE, origin, region_mt, imageSizeX*sizeof(float), 0, mt_vec[i], NULL, NULL);
        
        //xprojection
        kernel_xproj.setArg(0, mt_img);
        kernel_xproj.setArg(1, xprj_buff);
        kernel_xproj.setArg(2, startX);
        kernel_xproj.setArg(3, endX);
        kernel_xproj.setArg(4, i);
        command_queue.enqueueNDRangeKernel(kernel_xproj, NULL, global_item_size_xprj, local_item_size_xprj, NULL, NULL);
        command_queue.finish();
        
    }
    command_queue.enqueueCopyBufferToImage(xprj_buff, xprj_img, 0, origin, region_xprj,NULL,NULL);
    
    //zcorrection
    const cl::NDRange local_item_size_zcorr(maxWorkGroupSize,1,1);
    const cl::NDRange global_item_size_zcorr(maxWorkGroupSize*num_angle,1,1);
    
    kernel_zcorr.setArg(0, xprj_img);
    kernel_zcorr.setArg(1, zshift_buff);
    kernel_zcorr.setArg(2, cl::Local(sizeof(cl_float)*imageSizeY)); //target_xproj
    kernel_zcorr.setArg(3, cl::Local(sizeof(cl_float)*maxWorkGroupSize)); //loc_mem
    kernel_zcorr.setArg(4, startZ);
    kernel_zcorr.setArg(5, endZ);
    
    command_queue.enqueueNDRangeKernel(kernel_zcorr, NULL, global_item_size_zcorr, local_item_size_zcorr, NULL, NULL);
    
    return 0;
}


int reslice(cl::CommandQueue command_queue, cl::Kernel kernel,
            int num_angle,int iter,float baseup,
            vector<float*> mt_vec, vector<float*> prj_vec,
            cl::Buffer xshift_buff,cl::Buffer zshift_buff,
            int imageSizeX,int imageSizeY){
    
    cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
    size_t maxWorkGroupSize = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    
    cl::ImageFormat format(CL_R, CL_FLOAT);
    const cl::NDRange local_item_size(min(imageSizeX,(int)maxWorkGroupSize),1,1);
    const cl::NDRange global_item_size(imageSizeX,imageSizeY,1);
    cl::size_t<3> origin;
    cl::size_t<3> region_mt;
    cl::size_t<3> region_prj;
    origin[0] = 0;
    origin[1] = 0;
    region_mt[0] = imageSizeX;
    region_mt[1] = imageSizeY;
    region_mt[2] = 1;
    region_prj[0] = imageSizeX;
    region_prj[1] = num_angle/iter;
    region_prj[2] = 1;
    
    //image object declaration
    cl::Image2D mt_img(context, CL_MEM_READ_WRITE,format,imageSizeX, imageSizeY, 0, NULL, NULL);
    cl::Image2DArray prj_img(context, CL_MEM_READ_WRITE,format,imageSizeY,imageSizeX, num_angle/iter, 0, 0, NULL, NULL);
    
    for (int j=0; j<num_angle; j+=num_angle/iter) {
        //write input mt_img object
        origin[2] = 0;
        for (int i=0; i<num_angle/iter; i++) {
            command_queue.enqueueWriteImage(mt_img, CL_TRUE, origin, region_mt, imageSizeX*sizeof(cl_float), 0, mt_vec[i+j], NULL, NULL);
        
            //reslice
            kernel.setArg(0, mt_img);
            kernel.setArg(1, prj_img);
            kernel.setArg(2, xshift_buff);
            kernel.setArg(3, zshift_buff);
            kernel.setArg(4, baseup);
            kernel.setArg(5, i);
            kernel.setArg(6, j);
            kernel.setArg(7, (char)0);
            command_queue.enqueueNDRangeKernel(kernel, NULL, global_item_size, local_item_size, NULL, NULL);
            command_queue.finish();
        }
        
        for (int i = 0; i < imageSizeY; i++) {
            origin[2]=i;
            command_queue.enqueueReadImage(prj_img, CL_TRUE, origin, region_prj, imageSizeX*sizeof(cl_float), 0, prj_vec[i]+j*imageSizeX, NULL, NULL);
        }
    }
    
    
    return 0;
}


int reslice_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                   int startAngleNo, int EndAngleNo,string subDir_str,float baseup,
                   input_parameter inp,string fileName_base_o,
                   vector<float*> mt_vec, int thread_id)
{
    string output_dir = inp.getOutputDir();
    int startX=inp.getStartX();
    int endX=inp.getEndX();
    int startZ=inp.getStartZ();
    int endZ=inp.getEndZ();
    int imageSizeX = inp.getImageSizeX();
    int imageSizeY = inp.getImageSizeY();
    int64_t imageSizeM = inp.getImageSizeM();
    
    try {
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        string devicename = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>();
        size_t globalMemSize = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
        
        
        int num_angle = EndAngleNo - startAngleNo + 1;
        int iter=1;
        size_t memobj = sizeof(cl_float)*((int64_t)imageSizeM*(num_angle/iter+1)+num_angle*2);
        while (memobj>globalMemSize) {
            iter*=2;
            memobj = sizeof(cl_float)*((int64_t)imageSizeM*(num_angle/iter+1)+num_angle*2);
        }
        
        
        cout<<"Device: "<<devicename<<endl<<"Processing "<<subDir_str<<"..."<<endl<<"   Iteration: "<<iter<<endl<<endl;
        
        
        cl::Buffer xshift_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*num_angle, 0, NULL);
        cl::Buffer zshift_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*num_angle, 0, NULL);
        command_queue.enqueueFillBuffer(xshift_buff, (cl_float)0.0, 0, sizeof(float)*num_angle);
        command_queue.enqueueFillBuffer(zshift_buff, (cl_float)0.0, 0, sizeof(float)*num_angle);
        
        
        vector<float*> prj_vec;
        float *xshift, *zshift;
        xshift = new float[num_angle];
        zshift = new float[num_angle];
        //reserve pointer for resliced projection
        for (int i = 0; i < imageSizeY; i++) {
            prj_vec.push_back(new float[imageSizeX*num_angle]);
        }
        
        
        //zcorrection
        if(inp.getZcorr()){
            cout<< "Device: "<< devicename<<endl;
            cout<< "   Z-correction"<<endl<<endl;
            Z_Correction(command_queue,kernel[1],kernel[2],num_angle,
                         startX,endX,startZ,endZ,mt_vec, zshift_buff,
                         imageSizeX,imageSizeY);
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
        reslice(command_queue, kernel[0],num_angle,iter,baseup,mt_vec,prj_vec,
                xshift_buff,zshift_buff,
                imageSizeX,imageSizeY);
        
        
        //delete input mt_vec
        for (int i=startAngleNo; i<=EndAngleNo; i++) {
            delete [] mt_vec[i-startAngleNo];
        }
        
        //output
        if (inp.getZcorr()||inp.getXcorr()) {
            output_th1[thread_id].join();
            output_th1[thread_id]=thread(xyshift_output_thread,num_angle,output_dir,
                                   move(xshift), move(zshift));
        }
        output_th2[thread_id].join();
        output_th2[thread_id]=thread(prj_output_thread,
                                     inp.getStartLayer(),inp.getEndLayer(),
                                     subDir_str, num_angle,
                                     output_dir,fileName_base_o,move(prj_vec),imageSizeX);
        
        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
    
    return 0;
}

int data_input_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                      int startAngleNo, int endAngleNo,string subDir_str,float baseup,
                      input_parameter inp,string fileName_base_i,string fileName_base_o,
                      int thread_id){
    
    int imageSizeM = inp.getImageSizeM();
    
    vector<float*> mt_vec;
    for (int i=startAngleNo; i<=endAngleNo; i++) {
        mt_vec.push_back(new float[inp.getImageSizeM()]);
    }
    
    
    //input mt data
    string input_dir =inp.getInputDir();
    for (int i=startAngleNo; i<=endAngleNo; i++) {
        string filepath_input = input_dir+"/"+subDir_str+fileName_base_i+AnumTagString(i, "", ".raw");
        //cout<<filepath_input;
        readRawFile(filepath_input,mt_vec[i-startAngleNo],imageSizeM);
    }
    
    
    reslice_th[thread_id].join();
    reslice_th[thread_id] = thread(reslice_thread,command_queue,kernel,
                                   startAngleNo,endAngleNo,subDir_str,baseup,
                                   inp,fileName_base_o,move(mt_vec),thread_id);
    return 0;
}

//dummy thread
static int dummy(){
    return 0;
}

int reslice_programBuild(cl::Context context,vector<cl::Kernel> *kernels,
                        int startAngleNo, int endAngleNo, input_parameter inp){
    cl_int ret;
    
    //OpenCL Program
    //kernel program source
    /*ifstream ifs("../reslice.cl", ios::in);
     if(!ifs) {
     cerr << "   Failed to load kernel" << endl;
     return -1;
     }
     istreambuf_iterator<char> it(ifs);
     istreambuf_iterator<char> last;
     string kernel_src_reslice(it,last);
     ifs.close();*/
    ostringstream oss;
    oss<<"-D PRJ_ANGLESIZE="<<endAngleNo-startAngleNo+1<<" ";
    oss<<"-D NUM_TRIAL="<<20<<" ";
    oss<<"-D LAMBDA="<<0.001f<<" ";
    oss<<"-D IMAGESIZE_X="<<inp.getImageSizeX()<<" ";
    oss<<"-D IMAGESIZE_Y="<<inp.getImageSizeY()<<" ";
    oss<<"-D IMAGESIZE_M="<<inp.getImageSizeM()<<" ";
    string option = oss.str();
    //cout << kernel_code<<endl;
#if defined (OCL120)
    cl::Program::Sources source(1,std::make_pair(kernel_src_reslice.c_str(),kernel_src_reslice.length()));
#else
    cl::Program::Sources source(1,kernel_src_reslice);
#endif
    cl::Program program(context, source,&ret);
    //kernel build
    ret=program.build(option.c_str());
    //cout<<ret<<endl;
    kernels[0].push_back(cl::Kernel::Kernel(program,"reslice", &ret));//0
    kernels[0].push_back(cl::Kernel::Kernel(program,"xprojection", &ret));//1
    kernels[0].push_back(cl::Kernel::Kernel(program,"zcorrection", &ret));//2
    
    return 0;
}


int reslice_ocl(input_parameter inp,OCL_platform_device plat_dev_list,string fileName_base_i)
{
    
    int startAngleNo=inp.getStartAngleNo();
    int endAngleNo=inp.getEndAngleNo();
    float baseup = inp.getBaseup();
    baseup = (baseup==NAN) ? 0:baseup;
    string fileName_base_o = inp.getOutputFileBase();

    //OpenCL Program Build
    vector<vector<cl::Kernel>> kernels;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        vector<cl::Kernel> kernels_plat;
        reslice_programBuild(plat_dev_list.context(i),&kernels_plat,startAngleNo,endAngleNo,inp);
        kernels.push_back(kernels_plat);
    }
    
    
    //display OCL device
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
            cout << "CL DEVICE NAME: "<< device_pram<<endl<<endl;
            
            t++;
        }
    }
    
    
    //start threads
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        input_th.push_back(thread(dummy));
        reslice_th.push_back(thread(dummy));
        output_th1.push_back(thread(dummy));
        output_th2.push_back(thread(dummy));
    }
    if ((inp.getStartEnergyNo()>0)&(inp.getEndEnergyNo()>0)){
        int startEnergyNo=inp.getStartEnergyNo();
        int endEnergyNo=inp.getEndEnergyNo();
        //create output dir
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            string fileName_output = inp.getOutputDir() + "/" + EnumTagString(i,"","");
            MKDIR(fileName_output.c_str());
        }
        for (int i=startEnergyNo; i<=endEnergyNo;) {
            for (int j=0; j<plat_dev_list.contextsize(); j++) {
                if (input_th[j].joinable()) {
                    input_th[j].join();
                    input_th[j] = thread(data_input_thread,
                                        plat_dev_list.queue(j, 0),kernels[j],
                                         startAngleNo,endAngleNo,
                                         EnumTagString(i,"",""),baseup,
                                         inp,fileName_base_i,fileName_base_o,j);
                    i++;
                    if (i > endEnergyNo) break;
                } else{
                    this_thread::sleep_for(chrono::seconds(1));
                }
            
            }
            if (i > endEnergyNo) break;
        }
    }else if (inp.getFittingParaName().size()>0) {
        //create output dir
        vector<string> paraname;
        for (int i=0; i<inp.getFittingParaName().size(); i++) {
            paraname.push_back(inp.getFittingParaName()[i]);
            string fileName_output = inp.getOutputDir() + "/" + paraname[i];
            MKDIR(fileName_output.c_str());
        }
        for (int i=0; i<inp.getFittingParaName().size();) {
            for (int j=0; j<plat_dev_list.contextsize(); j++) {
                if (input_th[j].joinable()) {
                    input_th[j].join();
                    input_th[j] = thread(data_input_thread,
                                         plat_dev_list.queue(j, 0),kernels[j],
                                         startAngleNo,endAngleNo,
                                         paraname[i],baseup,
                                         inp,fileName_base_i,fileName_base_o,j);
                    i++;
                    if (i >= inp.getFittingParaName().size()) break;
                } else{
                    this_thread::sleep_for(chrono::seconds(1));
                }
                
            }
            if (i >= inp.getFittingParaName().size()) break;
        }
    }
    
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        input_th[j].join();
    }
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        reslice_th[j].join();
    }
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        output_th1[j].join();
    }
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        output_th2[j].join();
    }
    
    return 0;
}
