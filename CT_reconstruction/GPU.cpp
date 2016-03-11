//
//  GPU.cpp
//  CT_reconstruction
//
//  Created by Nozomu Ishiguro on 2015/06/21.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "CT_reconstruction.hpp"
#include "OSEM_cl.hpp"
#include "FBP_cl.hpp"

string base_f1;
string tale_f1;
string base_f2;
string tale_f2;

extern vector<thread> input_th, reconst_th, output_th;

int GPU(){
    
    OCL_platform_device plat_dev_list(g_devList,true);
    
    base_f1=g_f1;
    base_f1.erase(g_f1.find_first_of("*"),g_f1.length()-1);
    tale_f1=g_f1;
    tale_f1.erase(0,g_f1.find_last_of("*")+1);
    base_f2=g_f2;
    base_f2.erase(g_f2.find_first_of("*"),g_f2.length()-1);
    tale_f2=g_f2;
    tale_f2.erase(0,g_f2.find_last_of("*")+1);
    
    switch (g_mode){
        case 1: //add (加算型) algebraic reconstruction technique 法
            //AART(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_it, N, g_wt1);
            break;
        case 2: //multiply (乗算型) algebraic reconstruction technique 法
            OSEM_ocl(plat_dev_list, g_ang, g_pa); //OS-EMのg_ss=g_paと同等
            break;
        case 3: //add (加算型) simultaneous reconstruction technique 法
            //ASIRT(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_it, N, g_wt2);
            break;
        case 4: //multiply(乗算型) simultaneous reconstruction technique 法
            //MSIRT(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_it, N);
            break;
        case 5: //maximum likelihood-expection maximumization (ML-EM)法
			cout << "Processing by ML-EM method" << endl << endl;
            OSEM_ocl(plat_dev_list, g_ang, 1);  //OS-EMのg_ss=1と同等
            break;
        case 6:  //ordered subset EM (OS-EM)法
			cout << "Processing by OS-EM method" << endl << endl;
            OSEM_ocl(plat_dev_list, g_ang, g_ss);
            break;
        case 7: //filter back-projection法
			cout << "Processing by FBP method" << endl << endl;
            FBP_ocl(plat_dev_list, g_ang);
            break;
        default:
            break;
    }
    
    return 0;
}

int output_thread(int startN, int endN, vector<float*> imgs){
    time_t t1;
    for (int i=startN; i<=endN; i++) {
        ostringstream oss2;
        oss2<<base_f2<<setfill('0')<<setw(4)<<i+1<<tale_f2;
        string f2=oss2.str();
        write_data_output(g_d2,f2,imgs[i-startN],g_nx*g_nx);
    }
    
    
    time(&t1);
    cout<<endl;
    for (int i=startN; i<=endN; i++) {
        cout<<"[Layer " << setfill('0')<<setw(4)<< i+1 <<"/"<< g_st -1 + g_num<<"] ";
        cout<<(float)(t1-g_t0)<<"秒経過"<<endl;
        
        delete [] imgs[i-startN];
    }
	cout << endl;
    
    return 0;
}

int OSEM_execution(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                   cl::Buffer angle_buffer, /*cl::Buffer*/int *sub/*_buffer*/,
                   cl::Image2DArray reconst_img, cl::Image2DArray prj_img,
                   int dN, int offsetN, int it){
    
    cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
    cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
    string devicename = device.getInfo<CL_DEVICE_NAME>();
    size_t WorkGroupSize = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), IMAGE_SIZE_X);
    
    //OpenCL memory objects
    cl::ImageFormat format(CL_R,CL_FLOAT);
    cl::Image2DArray reconst_dest_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_nx,0,0,NULL,NULL);
    cl::Image2DArray rprj_img(context,CL_MEM_READ_WRITE,format,dN,g_px,g_pa,0,0,NULL,NULL);
    
    //write memory objects to GPUs
    cl::size_t<3> origin;
    cl::size_t<3> region;
    cl::size_t<3> region_prj;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    region[0] = g_nx;
    region[1] = g_nx;
    region[2] = dN;
    region_prj[0] = g_nx;
    region_prj[1] = g_pa;
    region_prj[2] = dN;
    command_queue.enqueueCopyImage(reconst_img, reconst_dest_img, origin, origin, region, NULL,NULL);
    
    //NDrange settings
    const cl::NDRange global_item_size(WorkGroupSize*dN,1,1);
    const cl::NDRange local_item_size(WorkGroupSize,1,1);
    
    //sinogram correction
    if(correctionMode>0){
        command_queue.enqueueCopyImage(prj_img, rprj_img, origin, origin, region_prj, NULL,NULL);
		command_queue.finish();
        kernel[2].setArg(0, rprj_img);
        kernel[2].setArg(1, prj_img);
        kernel[2].setArg(2, angle_buffer);
        kernel[2].setArg(3, (cl_int)correctionMode);
        command_queue.enqueueNDRangeKernel(kernel[2], NULL, global_item_size, local_item_size, NULL, NULL);
        command_queue.finish();
    }
				
    for (int i=0; i<it; i++) {
        for (int k=0; k<g_ss; k++) {
            kernel[0].setArg(0, reconst_img);
            kernel[0].setArg(1, prj_img);
            kernel[0].setArg(2, rprj_img);
            kernel[0].setArg(3, angle_buffer);
            kernel[0].setArg(4, sub[k]);
            command_queue.enqueueNDRangeKernel(kernel[0], NULL, global_item_size, local_item_size, NULL, NULL);
            command_queue.finish();
            
            kernel[1].setArg(0, reconst_img);
            kernel[1].setArg(1, reconst_dest_img);
            kernel[1].setArg(2, rprj_img);
            kernel[1].setArg(3, angle_buffer);
            kernel[1].setArg(4, sub[k]);
            command_queue.enqueueNDRangeKernel(kernel[1], NULL, global_item_size, local_item_size, NULL, NULL);
            command_queue.finish();
            
            command_queue.enqueueCopyImage(reconst_dest_img, reconst_img, origin, origin, region,NULL,NULL);
            //command_queue.enqueueNDRangeKernel(kernel[2], NULL, global_item_size, local_item_size, NULL, NULL);
            command_queue.finish();
        }
    }
    
    
    return 0;
}

int OSEM_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                cl::Buffer angle_buffer, int *sub, vector<float*> imgs, vector<float*> prjs,
                int startN, int endN, int it, int thread_id){
    
    int dN = endN-startN+1;
    cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
#ifdef BATCHEXE
    int offsetN=startN;
#else
    int offsetN=0;
#endif
    
    //OpenCL memory objects
    cl::ImageFormat format(CL_R,CL_FLOAT);
    cl::Image2DArray reconst_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_nx,0,0,NULL,NULL);
    cl::Image2DArray prj_img(context,CL_MEM_READ_WRITE,format,dN,g_px,g_pa,0,0,NULL,NULL);
    
    //write memory objects to GPUs
    cl::size_t<3> origin;
    cl::size_t<3> region_img;
    cl::size_t<3> region_prj;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    region_img[0] = g_nx;
    region_img[1] = g_nx;
    region_img[2] = 1;
    region_prj[0] = g_px;
    region_prj[1] = g_pa;
    region_prj[2] = 1;
    for (int i=0; i<dN; i++) {
        origin[2] = i;
        command_queue.enqueueWriteImage(reconst_img, CL_TRUE, origin, region_img, sizeof(cl_float)*g_nx,0,imgs[i],NULL,NULL);
        command_queue.enqueueWriteImage(prj_img,CL_TRUE,origin,region_prj,sizeof(cl_float)*g_px,0,prjs[i+offsetN],NULL,NULL);
    }

    
    //OSEM execution
    OSEM_execution(command_queue, kernel,angle_buffer,sub,reconst_img,prj_img,dN,0,it);
    
    
    //read memory objects from GPUs
    for (int i=0; i<dN; i++) {
        origin[2] = i;
        command_queue.enqueueReadImage(reconst_img,CL_TRUE,origin,region_img,sizeof(float)*g_nx,0,imgs[i],NULL,NULL);
    }
    
    //output file
    output_th[thread_id].join();
    output_th[thread_id]=thread(output_thread,startN,endN,move(imgs));
    
#ifndef BATCHEXE
    for (int i=startN; i<=endN; i++) {
        delete [] prjs[i-startN];
    }
#endif
    
    return 0;
}

//dummy thread
static int dummy(){
    return 0;
}

int OSEM_input_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                  cl::Buffer angle_buffer, /*cl::Buffer*/int *sub/*_buffer*/,
                  int startN, int endN, int it, int thread_id){
    
    //read data
    vector<float*> imgs;
    vector<float*> prjs;
    for (int i=startN; i<=endN; i++) {
        prjs.push_back(new float[(unsigned long)g_px*g_pa]);
        imgs.push_back(new float[(unsigned long)g_nx*g_nx]);
    }
    
    for (int i=startN; i<=endN; i++) {
        ostringstream oss1;
        oss1<<base_f1<<setfill('0')<<setw(4)<<i+1<<tale_f1;
        string f1=oss1.str();
        read_data_input(g_d1,f1, prjs[i-startN], g_px*g_pa);
        first_image(g_f4, imgs[i-startN], g_nx*g_nx);
    }
    
    reconst_th[thread_id].join();
    reconst_th[thread_id]=thread(OSEM_thread,command_queue,kernel,
                                 angle_buffer, sub,
                                 move(imgs), move(prjs),
                                 startN, endN, it, thread_id);
    
    return 0;
}


int OSEM_programBuild(cl::Context context,vector<cl::Kernel> *kernels){
    cl_int ret;
    
    string kernel_code="";
    //kernel program source
    /*ifstream ifs("./OSEM.cl", ios::in);
     if(!ifs) {
     cerr << "   Failed to load kernel \n" << endl;
     return -1;
     }
     istreambuf_iterator<char> it(ifs);
     istreambuf_iterator<char> last;
     string kernel_src_osem(it,last);
     ifs.close();*/
    
    ostringstream oss;
    oss<<"#define IMAGESIZE_X "<< g_nx<<endl<<endl;
    oss<<"#define IMAGESIZE_Y "<< g_nx<<endl<<endl;
    oss<<"#define IMAGESIZE_M "<< g_nx*g_nx<<endl<<endl;
    oss<<"#define PRJ_IMAGESIZE "<< g_px<<endl<<endl;
    oss<<"#define PRJ_ANGLESIZE "<< g_pa<<endl<<endl;
    oss<<"#define SS "<< g_ss<<endl<<endl;
    kernel_code+=oss.str();
    //cout << kernel_code<<endl;
    kernel_code += kernel_src_osem;
    //cout << kernel_code<<endl;
    size_t kernel_code_size = kernel_code.length();
    cl::Program::Sources source(1,make_pair(kernel_code.c_str(),kernel_code_size));
    cl::Program program(context, source,&ret);
    //kernel build
    ret=program.build();
    //cout<<ret<<endl;
    kernels[0].push_back(cl::Kernel::Kernel(program,"OSEM1", &ret));//0
    kernels[0].push_back(cl::Kernel::Kernel(program,"OSEM2", &ret));//1
    kernels[0].push_back(cl::Kernel::Kernel(program,"sinogramCorrection", &ret));//2
    
    return 0;
}


int OSEM_ocl(OCL_platform_device plat_dev_list, float *ang, int ss){
    
    // サブセットの順番を決定する
    int *sub;
    sub =  new int[ss];
    int k = 0;
    for (int i = 0; i < 32; i++) k += (ss >> i) & 1;
    if (k == 1){    //ssが2^nの場合
        int m1 = 0;
        sub[m1++] = 0;
        int i, m2;
        for (i = ss, m2 = 1; i > 1; i /= 2, m2 *= 2){
            for (int j = 0; j < m2; j++) sub[m1++] = sub[j] + i / 2;
        }
    }
    else {
        for (int i = 0; i < ss; i++) {
            sub[i] = i;
            //cout<<i<<":"<< sub[i] <<endl;
        }
    }
    
    
    //OpenCL Program
    vector<vector<cl::Kernel>> kernels;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        vector<cl::Kernel> kernels_plat;
        OSEM_programBuild(plat_dev_list.context(i), &kernels_plat);
        kernels.push_back(kernels_plat);
    }
    
    
    //display OCL device
    int t=0;
    vector<int> dN;
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
            dN.push_back(min(numParallel, (int)device_pram_size[0]));
            cout<<"Number of working compute unit: "<<dN[t]<<"/"<<device_pram_size[0]<<endl<<endl;
            t++;
        }
    }
    
    
    //angle buffers作成
    vector<cl::Buffer> angle_buffers;
    for (int i=0; i<plat_dev_list.contextsize(); i++){
        angle_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*g_pa, 0, NULL));
        plat_dev_list.queue(i,0).enqueueWriteBuffer(angle_buffers[i], CL_TRUE, 0, sizeof(cl_float)*g_pa, ang, NULL, NULL);
    }
    
    
    //start thread
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        input_th.push_back(thread(dummy));
        reconst_th.push_back(thread(dummy));
        output_th.push_back(thread(dummy));
    }
    for (int N = g_st-1; N < g_st-1 + g_num;) {
        for (int j=0; j<plat_dev_list.contextsize(); j++) {
            if (input_th[j].joinable()) {
                input_th[j].join();
                input_th[j] = thread(OSEM_input_thread,
                               plat_dev_list.queue(j,0),kernels[j],
                               angle_buffers[j],
                               sub/*_buffers[j]*/,N,min(N+dN[j]-1,g_st-2+g_num),g_it,j);
                
                N+=dN[j];
                if (N >= g_st-1 + g_num) break;
                
            } else this_thread::sleep_for(chrono::seconds(1));
        }
    }
    
    for (int j=0; j<plat_dev_list.contextsize(); j++) {
        input_th[j].join();
    }
    for (int j=0; j<plat_dev_list.contextsize(); j++) {
        reconst_th[j].join();
    }
    for (int j=0; j<plat_dev_list.contextsize(); j++) {
        output_th[j].join();
    }
    
    delete [] sub;
    delete [] ang;
    return 0;
}

int FBP_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
               cl::Buffer angle_buffer,vector<float*> imgs, vector<float*> prjs,
               int startN, int endN,int thread_id){
    
    cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
    cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
    string devicename = device.getInfo<CL_DEVICE_NAME>();
    size_t WorkGroupSize = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), IMAGE_SIZE_X);
    
    int dN = endN-startN+1;
#ifdef BATCHEXE
    int offsetN=startN;
#else
    int offsetN=0;
#endif
    
    //OpenCL memory objects
    cl::ImageFormat format(CL_R,CL_FLOAT);
    cl::Image2DArray reconst_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_nx,0,0,NULL,NULL);
    cl::Image2DArray prj_img(context,CL_MEM_READ_WRITE,format,dN,g_px,g_pa,0,0,NULL,NULL);
    cl::Image2DArray bprj_img(context,CL_MEM_READ_WRITE,format,dN,g_px,g_pa,0,0,NULL,NULL);
    
    //write memory objects to GPUs
    cl::size_t<3> origin;
    cl::size_t<3> region_img;
    cl::size_t<3> region_prj;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    region_img[0] = g_nx;
    region_img[1] = g_nx;
    region_img[2] = 1;
    region_prj[0] = g_px;
    region_prj[1] = g_pa;
    region_prj[2] = 1;
    
    
    for (int i=0; i<dN; i++) {
        origin[2] = i;
        command_queue.enqueueWriteImage(prj_img,CL_TRUE,origin,region_prj,sizeof(cl_float)*g_px,0,prjs[i+offsetN],NULL,NULL);
    }
    
    //NDrange settings
    const cl::NDRange global_item_size(WorkGroupSize*dN,1,1);
    const cl::NDRange local_item_size(WorkGroupSize,1,1);
    
    //kernel set Arguments
    kernel[0].setArg(0, prj_img);
    kernel[0].setArg(1, bprj_img);
    kernel[0].setArg(2, cl::Local(sizeof(cl_float)*g_zp)); //prz
    kernel[0].setArg(3, cl::Local(sizeof(cl_float2)*g_zp)); //xc1
    kernel[0].setArg(4, cl::Local(sizeof(cl_float2)*g_zp/2)); //W
    
    kernel[1].setArg(0, bprj_img);
    kernel[1].setArg(1, reconst_img);
    kernel[1].setArg(2, angle_buffer);
				

    command_queue.enqueueNDRangeKernel(kernel[0], NULL, global_item_size, local_item_size, NULL, NULL);
    command_queue.finish();
    
    command_queue.enqueueNDRangeKernel(kernel[1], NULL, global_item_size, local_item_size, NULL, NULL);
    command_queue.finish();
    

    for (int i=0; i<dN; i++) {
        origin[2] = i;
        command_queue.enqueueReadImage(reconst_img,CL_TRUE,origin,region_img,sizeof(float)*g_nx,0,imgs[i],NULL,NULL);
    }
    
    //output file
    output_th[thread_id].join();
    output_th[thread_id]=thread(output_thread,startN,endN,move(imgs));
    
#ifndef BATCHEXE
    for (int i=startN; i<=endN; i++) {
        delete [] prjs[i-startN];
    }
#endif
    
    return 0;
}

int FBP_input_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                     cl::Buffer angle_buffer,int startN, int endN,int thread_id){
    
    //read data
    vector<float*> imgs;
    vector<float*> prjs;
    for (int i=startN; i<=endN; i++) {
        prjs.push_back(new float[(unsigned long)g_px*g_pa]);
        imgs.push_back(new float[(unsigned long)g_nx*g_nx]);
    }
    
    for (int i=startN; i<=endN; i++) {
        ostringstream oss1;
        oss1<<base_f1<<setfill('0')<<setw(4)<<i+1<<tale_f1;
        string f1=oss1.str();
        read_data_input(g_d1,f1, prjs[i-startN], g_px*g_pa);
        first_image(g_f4, imgs[i-startN], g_nx*g_nx);
    }
    
    reconst_th[thread_id].join();
    reconst_th[thread_id]=thread(FBP_thread,command_queue,kernel,
                                 angle_buffer, move(imgs), move(prjs),
                                 startN, endN, thread_id);
    
    return 0;
}

int FBP_programBuild(cl::Context context,vector<cl::Kernel> *kernels){
    cl_int ret;
    
    string kernel_code="";
    //kernel program source
    /*ifstream ifs("./FBP.cl", ios::in);
     if(!ifs) {
     cerr << "   Failed to load kernel" << endl;
     return -1;
     }
     istreambuf_iterator<char> it(ifs);
     istreambuf_iterator<char> last;
     string kernel_src_fbp(it,last);
     ifs.close();*/
    
    //OpenCL Program
    ostringstream oss;
    oss<<"#define IMAGESIZE_X "<< g_nx<<endl<<endl;
    oss<<"#define IMAGESIZE_Y "<< g_nx<<endl<<endl;
    oss<<"#define IMAGESIZE_M "<< g_nx*g_nx<<endl<<endl;
    oss<<"#define PRJ_IMAGESIZE "<< g_px<<endl<<endl;
    oss<<"#define PRJ_ANGLESIZE "<< g_pa<<endl<<endl;
    oss<<"#define ZP_SIZE "<<g_zp<<endl<<endl;
    kernel_code+=oss.str();
    //cout << kernel_code<<endl;
    kernel_code += kernel_src_fbp;
    //cout << kernel_code<<endl;
    size_t kernel_code_size = kernel_code.length();
    cl::Program::Sources source(1,make_pair(kernel_code.c_str(),kernel_code_size));
    cl::Program program(context, source,&ret);
    //kernel build
    ret=program.build();
    //cout<<ret<<endl;
    kernels[0].push_back(cl::Kernel::Kernel(program,"FBP1", &ret));//0
    kernels[0].push_back(cl::Kernel::Kernel(program,"FBP2", &ret));//1
    kernels[0].push_back(cl::Kernel::Kernel(program,"sinogramCorrection", &ret));//2
    
    return 0;
}

int FBP_ocl(OCL_platform_device plat_dev_list, float *ang){
    
    //OpenCL Program
    vector<vector<cl::Kernel>> kernels;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        vector<cl::Kernel> kernels_plat;
        FBP_programBuild(plat_dev_list.context(i), &kernels_plat);
        kernels.push_back(kernels_plat);
    }
    
    //display OCL device
    int t=0;
    vector<int> dN;
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
            dN.push_back(1);
            size_t device_pram_size[3]={0,0,0};
            plat_dev_list.dev(i,j).getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &device_pram_size);
            cout<<"Number of working compute unit: "<<dN[t]<<"/"<<device_pram_size[0]<<endl<<endl;
            t++;
        }
    }
    
    
    //queueを通し番号に変換
    vector<cl::CommandQueue> queues;
    vector<int> cotextID_OfQueue;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        for (int j=0; j<plat_dev_list.queuesize(i); j++) {
            queues.push_back(plat_dev_list.queue(i,j));
            cotextID_OfQueue.push_back(i);
        }
    }
    
    
    
    //angle_buffers作成
    vector<cl::Buffer> angle_buffers;
    for (int i=0; i<plat_dev_list.contextsize(); i++){
        angle_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*g_pa, 0, NULL));
        plat_dev_list.queue(i,0).enqueueWriteBuffer(angle_buffers[i], CL_TRUE, 0, sizeof(cl_float)*g_pa, ang, NULL, NULL);
    }
    
    
    //thread start
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        input_th.push_back(thread(dummy));
        reconst_th.push_back(thread(dummy));
        output_th.push_back(thread(dummy));
    }
    for (int N = g_st-1; N < g_st-1 + g_num;) {
        for (int j=0; j<queues.size(); j++) {
            if (input_th[j].joinable()) {
                input_th[j].join();
                input_th[j] = thread(FBP_input_thread,
                                     queues[j],kernels[cotextID_OfQueue[j]],
                                     angle_buffers[cotextID_OfQueue[j]],
                                     N,min(N+dN[j]-1,g_st-2+g_num),j);
                
                N+=dN[j];
                if (N >= g_st-1 + g_num) break;
                
            } else this_thread::sleep_for(chrono::seconds(1));
        }
    }
    
    for (int j=0; j<plat_dev_list.contextsize(); j++) {
        input_th[j].join();
    }
    for (int j=0; j<plat_dev_list.contextsize(); j++) {
        reconst_th[j].join();
    }
    for (int j=0; j<plat_dev_list.contextsize(); j++) {
        output_th[j].join();
    }
    
    delete [] ang;
    return 0;
}

