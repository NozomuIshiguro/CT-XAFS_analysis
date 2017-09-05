//
//  GPU.cpp
//  CT_reconstruction
//
//  Created by Nozomu Ishiguro on 2015/06/21.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "CT_reconstruction.hpp"
#include "AART_cl.hpp"
#include "OSEM_cl.hpp"
#include "FBP_cl.hpp"
#include "FISTA_cl.hpp"
#include "reconstShare_cl.hpp"

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
            cout << "Processing by AART method";
            if(CSitBool) {
                cout << " with FISTA compressed sensing"<<endl<<endl;
                FISTA_ocl(plat_dev_list, g_ang, g_ss);
            }else {
                cout << endl << endl;
                AART_ocl(plat_dev_list, g_ang, g_ss);
            }
            break;
        case 2: //multiply (乗算型) algebraic reconstruction technique 法
            cout << "Processing by MART method";
            if(CSitBool) cout << " with FISTA compressed sensing"<<endl<<endl;
            else cout << endl << endl;
            OSEM_ocl(plat_dev_list, g_ang, g_pa); //OS-EMのg_ss=g_paと同等
            break;
        case 3: //add (加算型) simultaneous reconstruction technique 法
            cout << "Processing by ASIRT method";
            if(CSitBool) {
                cout << " with FISTA compressed sensing"<<endl<<endl;
                FISTA_ocl(plat_dev_list, g_ang, 1);
            }else {
                cout << endl << endl;
                AART_ocl(plat_dev_list, g_ang, 1); //AARTのg_ss=1と同等
            }
            break;
        case 4: //multiply(乗算型) simultaneous reconstruction technique 法
            cout << "Processing by MSIRT method";
            if(CSitBool) cout << " with FISTA compressed sensing"<<endl<<endl;
            else cout << endl << endl;
            OSEM_ocl(plat_dev_list, g_ang, 1);  //ML-EM法と同等?
            break;
        case 5: //maximum likelihood-expection maximumization (ML-EM)法
            cout << "Processing by ML-EM method";
            if(CSitBool) cout << " with FISTA compressed sensing"<<endl<<endl;
            else cout << endl << endl;
            OSEM_ocl(plat_dev_list, g_ang, 1);  //OS-EMのg_ss=1と同等
            break;
        case 6:  //ordered subset EM (OS-EM)法
            cout << "Processing by OS-EM method";
            if(CSitBool) cout << " with FISTA compressed sensing"<<endl<<endl;
            else cout << endl << endl;
            OSEM_ocl(plat_dev_list, g_ang, g_ss);
            break;
        case 7: //filter back-projection法
			cout << "Processing by FBP method" << endl << endl;
            FBP_ocl(plat_dev_list, g_ang);
            break;
        case 8: //FBP-OS-EM hybrid
            cout << "Processing by FBP-OE-SM hybrid method";
            if(CSitBool) cout << " with FISTA compressed sensing"<<endl<<endl;
            else cout << endl << endl;
            hybrid_ocl(plat_dev_list, g_ang, g_ss);
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
        oss2<<base_f2<<setfill('0')<<setw(4)<<i<<tale_f2;
        string f2=oss2.str();
        write_data_output(g_d2,f2,imgs[i-startN],g_ox*g_oy);
    }
    
    
    time(&t1);
    cout<<endl;
    for (int i=startN; i<=endN; i++) {
        cout<<"[Layer " << setfill('0')<<setw(4)<< i <<"/"<< g_st -1 + g_num<<"] ";
        cout<<(float)(t1-g_t0)<<"秒経過"<<endl;
        
        delete [] imgs[i-startN];
    }
	cout << endl;
    
    return 0;
}

int OSEM_execution(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                   cl::Buffer angle_buffer, int *sub,
                   cl::Image2DArray reconst_img, cl::Image2DArray prj_img,
                   int dN, int it, bool prjCorrection){
    
    try {
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
        string devicename = device.getInfo<CL_DEVICE_NAME>();
        size_t WorkGroupSize_n = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), g_nx);
		size_t WorkGroupSize_p = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), g_px);
        
        //OpenCL memory objects
        cl::ImageFormat format(CL_R,CL_FLOAT);
        cl::Image2DArray reconst_dest_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_ny,0,0,NULL,NULL);
        cl::Image2DArray rprj_img(context,CL_MEM_READ_WRITE,format,dN,g_px,g_pa,0,0,NULL,NULL);
        
        //write memory objects to GPUs
		cl::size_t<3> origin;
        cl::size_t<3> origin_img;
        cl::size_t<3> origin_img2;
        cl::size_t<3> region;
        cl::size_t<3> region2;
        cl::size_t<3> region_prj;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        origin_img[0] = (g_nx-g_ox)/2;
        origin_img[1] = (g_ny-g_oy)/2;
        origin_img[2] = 0;
        origin_img2[1] = (g_ny-g_oy)/2;
        origin_img2[2] = 0;
        region[0] = g_ox;
        region[1] = g_oy;
        region[2] = dN;
        region2[1] = g_oy;
        region2[2] = dN;
        region_prj[0] = g_nx;
        region_prj[1] = g_pa;
        region_prj[2] = dN;
        command_queue.enqueueCopyImage(reconst_img, reconst_dest_img, origin_img, origin_img, region, NULL,NULL);
        
        //NDrange settings
        const cl::NDRange global_item_size0(g_px,g_pa,dN);
        const cl::NDRange global_item_size1(g_px,g_pa/g_ss,dN);
        const cl::NDRange global_item_size2(g_ox,g_oy,dN);
        const cl::NDRange local_item_size_n(WorkGroupSize_n,1,1);
		const cl::NDRange local_item_size_p(WorkGroupSize_p, 1, 1);
        
        //sinogram correction
        if(prjCorrection&&correctionMode>0){
            command_queue.enqueueCopyImage(prj_img, rprj_img, origin, origin, region_prj, NULL,NULL);
            command_queue.finish();
            kernel[2].setArg(0, rprj_img);
            kernel[2].setArg(1, prj_img);
            kernel[2].setArg(2, angle_buffer);
            kernel[2].setArg(3, (cl_int)correctionMode);
            kernel[2].setArg(4, (cl_float)0.5f);
            kernel[2].setArg(5, (cl_float)0.5f);
            command_queue.enqueueNDRangeKernel(kernel[2], NULL, global_item_size0, local_item_size_p, NULL, NULL);
            command_queue.finish();
        }
        
		//projection
		kernel[0].setArg(0, reconst_img);
		kernel[0].setArg(1, prj_img);
		kernel[0].setArg(2, rprj_img);
		kernel[0].setArg(3, angle_buffer);
		//back projection
		kernel[1].setArg(0, reconst_img);
		kernel[1].setArg(1, reconst_dest_img);
		kernel[1].setArg(2, rprj_img);
		kernel[1].setArg(3, angle_buffer);
        
        for (int i=0; i<it; i++) {
            for (int k=0; k<g_ss; k++) {
                //extent reconst image
                for (int org=(g_nx-3*g_ox)/2; org>=-g_ox/2; org-=g_ox) {
                    origin_img2[0] = max(0,org);
                    region2[0] = org + g_ox - max(0, org);
                    command_queue.enqueueCopyImage(reconst_dest_img, reconst_img, origin_img, origin_img2, region2, NULL,NULL);
                }
                for (int org=(g_nx+g_ox)/2; org<g_nx; org+=g_ox) {
                    origin_img2[0] = org;
                    region2[0] = min(org+g_ox,g_nx) - org;
                    command_queue.enqueueCopyImage(reconst_dest_img, reconst_img, origin_img, origin_img2, region2, NULL,NULL);
                }
                
                
                //projection
                kernel[0].setArg(4, sub[k]);
                command_queue.enqueueNDRangeKernel(kernel[0], NULL, global_item_size1, local_item_size_p, NULL, NULL);
                command_queue.finish();
                
                //back projection
                kernel[1].setArg(4, sub[k]);
                command_queue.enqueueNDRangeKernel(kernel[1], NULL, global_item_size2, local_item_size_n, NULL, NULL);
                command_queue.finish();
                
                
                //update (copy) image
                command_queue.enqueueCopyImage(reconst_dest_img, reconst_img, origin_img, origin_img, region,NULL,NULL);
                command_queue.finish();
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


int FISTA_OSEM_execution(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                   cl::Buffer angle_buffer, int *sub,
                   cl::Image2DArray reconst_img, cl::Image2DArray prj_img,
                   int dN, int it, bool prjCorrection){
    
    try {
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
        string devicename = device.getInfo<CL_DEVICE_NAME>();
        size_t WorkGroupSize_n = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), g_nx);
        size_t WorkGroupSize_p = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), g_px);
        
        //OpenCL memory objects
        cl::ImageFormat format(CL_R,CL_FLOAT);
        cl::Image2DArray reconst_dest_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_ny,0,0,NULL,NULL);
        cl::Image2DArray rprj_img(context,CL_MEM_READ_WRITE,format,dN,g_px,g_pa,0,0,NULL,NULL);
        
        //write memory objects to GPUs
        cl::size_t<3> origin;
        cl::size_t<3> origin_img;
        cl::size_t<3> origin_img2;
        cl::size_t<3> region;
        cl::size_t<3> region2;
        cl::size_t<3> region_prj;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        origin_img[0] = (g_nx-g_ox)/2;
        origin_img[1] = (g_ny-g_oy)/2;
        origin_img[2] = 0;
        origin_img2[1] = (g_ny-g_oy)/2;
        origin_img2[2] = 0;
        region[0] = g_ox;
        region[1] = g_oy;
        region[2] = dN;
        region2[1] = g_oy;
        region2[2] = dN;
        region_prj[0] = g_nx;
        region_prj[1] = g_pa;
        region_prj[2] = dN;
        command_queue.enqueueCopyImage(reconst_img, reconst_dest_img, origin_img, origin_img, region, NULL,NULL);
        
        //NDrange settings
        const cl::NDRange global_item_size0(g_px,g_pa,dN);
        const cl::NDRange global_item_size1(g_px,g_pa/g_ss,dN);
        const cl::NDRange global_item_size2(g_ox,g_oy,dN);
        const cl::NDRange local_item_size_n(WorkGroupSize_n,1,1);
        const cl::NDRange local_item_size_p(WorkGroupSize_p, 1, 1);
        
        //sinogram correction
        if(prjCorrection&&correctionMode>0){
            command_queue.enqueueCopyImage(prj_img, rprj_img, origin, origin, region_prj, NULL,NULL);
            command_queue.finish();
            kernel[2].setArg(0, rprj_img);
            kernel[2].setArg(1, prj_img);
            kernel[2].setArg(2, angle_buffer);
            kernel[2].setArg(3, (cl_int)correctionMode);
            kernel[2].setArg(4, (cl_float)0.5f);
            kernel[2].setArg(5, (cl_float)0.5f);
            command_queue.enqueueNDRangeKernel(kernel[2], NULL, global_item_size0, local_item_size_p, NULL, NULL);
            command_queue.finish();
        }
        
        //projection
        kernel[0].setArg(0, reconst_img);
        kernel[0].setArg(1, prj_img);
        kernel[0].setArg(2, rprj_img);
        kernel[0].setArg(3, angle_buffer);
        //back projection
        kernel[1].setArg(0, reconst_img);
        kernel[1].setArg(1, reconst_dest_img);
        kernel[1].setArg(2, rprj_img);
        kernel[1].setArg(3, angle_buffer);
        
        //initialization for FISTA
        //x[0]=reconst_img, beta[0]=0, w[0]=x[0]=reconst_img
        cl::Image2DArray beta_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_ny,0,0,NULL,NULL);
        cl::Image2DArray beta_dest_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_ny,0,0,NULL,NULL);
        cl::Buffer L2norm_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*g_ss, 0, NULL);
        cl::Image2DArray reconst_v_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_ny,0,0,NULL,NULL);
        cl::Image2DArray reconst_w_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_ny,0,0,NULL,NULL);
        cl_float4 color = {0.0f,0.0f,0.0f,1.0f};
        command_queue.enqueueFillImage(beta_img, color, origin_img, region);
        command_queue.enqueueFillBuffer(L2norm_buffer, (cl_float)0.5f, 0, sizeof(cl_float)*g_ss);
        command_queue.enqueueCopyImage(reconst_img, reconst_w_img, origin_img, origin_img, region,NULL,NULL);
        //ISTA (update only for 0th cycle)
        kernel[3].setArg(0, reconst_img);  //as x[0] img -> w[t] img
        kernel[3].setArg(1, reconst_dest_img); //as x[1] img
        kernel[3].setArg(3, L2norm_buffer);
        //FISTA (update)
        kernel[4].setArg(0, reconst_img); //as x[t] img -> w[t] img
        kernel[4].setArg(1, reconst_v_img);
        kernel[4].setArg(2, beta_img); //as beta[t] img
        kernel[4].setArg(3, reconst_w_img); //as w[t+1] img
        kernel[4].setArg(4, reconst_dest_img); //as x[t+1] img
        kernel[4].setArg(5, beta_dest_img); //as beta[t+1] img
        kernel[4].setArg(7, L2norm_buffer);
        
        
        bool firstBool = true;
        for (int i=0; i<it; i++) {
            for (int k=0; k<g_ss; k++) {
                //extent reconst image
                for (int org=(g_nx-3*g_ox)/2; org>=-g_ox/2; org-=g_ox) {
                    origin_img2[0] = max(0,org);
                    region2[0] = org + g_ox - max(0, org);
                    command_queue.enqueueCopyImage(reconst_dest_img, reconst_img, origin_img, origin_img2, region2, NULL,NULL);
                }
                for (int org=(g_nx+g_ox)/2; org<g_nx; org+=g_ox) {
                    origin_img2[0] = org;
                    region2[0] = min(org+g_ox,g_nx) - org;
                    command_queue.enqueueCopyImage(reconst_dest_img, reconst_img, origin_img, origin_img2, region2, NULL,NULL);
                }
                
                
                //projection
                kernel[0].setArg(4, sub[k]);
                command_queue.enqueueNDRangeKernel(kernel[0], NULL, global_item_size1, local_item_size_p, NULL, NULL);
                command_queue.finish();
                
                //back projection
                kernel[1].setArg(4, sub[k]);
                command_queue.enqueueNDRangeKernel(kernel[1], NULL, global_item_size2, local_item_size_n, NULL, NULL);
                command_queue.finish();
                
                //compressed sensing-based iteration
                if (firstBool){
                    //ISTA (0th cycle update)
                    kernel[3].setArg(2, sub[k]);
                    command_queue.enqueueNDRangeKernel(kernel[3], NULL, global_item_size2, local_item_size_n, NULL, NULL);
                    
                    //after 0th cycle, iterlation preceeds with w_img(reconst_w_img), not with x_img(reconst_w_img)
                    firstBool=false;
                    kernel[3].setArg(0, reconst_w_img);
                    kernel[4].setArg(0, reconst_w_img);
                }else{
                    command_queue.enqueueCopyImage(reconst_dest_img, reconst_v_img, origin_img, origin_img, region,NULL,NULL);
                    
                    //FISTA (update)
                    kernel[4].setArg(6, sub[k]);
                    command_queue.enqueueNDRangeKernel(kernel[4], NULL, global_item_size2, local_item_size_n, NULL, NULL);
                    
                    //update(copy) b image
                    command_queue.enqueueCopyImage(beta_dest_img, beta_img, origin_img, origin_img, region, NULL, NULL);
                    command_queue.finish();
                }
                
                //update(copy) x image
                command_queue.enqueueCopyImage(reconst_dest_img, reconst_img, origin_img, origin_img, region, NULL, NULL);
                command_queue.finish();
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
    cl::Image2DArray reconst_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_ny,0,0,NULL,NULL);
    cl::Image2DArray prj_img(context,CL_MEM_READ_WRITE,format,dN,g_px,g_pa,0,0,NULL,NULL);
    
    //write memory objects to GPUs
    cl::size_t<3> origin_img;
    cl::size_t<3> origin_prj;
    cl::size_t<3> region_img;
    cl::size_t<3> region_prj;
    origin_img[0] = (g_nx-g_ox)/2;
    origin_img[1] = (g_ny-g_oy)/2;
    origin_prj[0] = 0;
    origin_prj[1] = 0;
    region_img[0] = g_ox;
    region_img[1] = g_oy;
    region_img[2] = 1;
    region_prj[0] = g_px;
    region_prj[1] = g_pa;
    region_prj[2] = 1;
    for (int i=0; i<dN; i++) {
        origin_img[2] = i;
        origin_prj[2] = i;
        command_queue.enqueueWriteImage(reconst_img, CL_TRUE, origin_img, region_img, 0,0,imgs[i],NULL,NULL);
        command_queue.enqueueWriteImage(prj_img,CL_TRUE,origin_prj,region_prj,0,0,prjs[i+offsetN],NULL,NULL);
    }
    
    
    //OSEM execution
    if (CSitBool) {
        FISTA_OSEM_execution(command_queue, kernel,angle_buffer,sub,reconst_img,prj_img,dN,it,true);
    }else{
        OSEM_execution(command_queue, kernel,angle_buffer,sub,reconst_img,prj_img,dN,it,true);
    }
    
    
    
    //read memory objects from GPUs
    for (int i=0; i<dN; i++) {
        origin_img[2] = i;
        command_queue.enqueueReadImage(reconst_img,CL_TRUE,origin_img,region_img,0,0,imgs[i],NULL,NULL);
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
        imgs.push_back(new float[(unsigned long)g_ox*g_oy]);
    }
    
    for (int i=startN; i<=endN; i++) {
        ostringstream oss1;
        oss1<<base_f1<<setfill('0')<<setw(4)<<i<<tale_f1;
        string f1=oss1.str();
        read_data_input(g_d1,f1, prjs[i-startN], g_px*g_pa);
        first_image(g_f4, imgs[i-startN], g_ox*g_oy);
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
    
    cl::Program::Sources source;
#if defined (OCL120)
	source.push_back(make_pair(kernel_src_osem.c_str(),kernel_src_osem.length()));
	source.push_back(make_pair(kernel_src_fista.c_str(), kernel_src_fista.length()));
	source.push_back(make_pair(kernel_src_reconstShare.c_str(),kernel_src_reconstShare.length()));
#else
	source.push_back(kernel_src_osem);
	source.push_back(kernel_src_fista);
    source.push_back(kernel_src_reconstShare);
#endif
    cl::Program program(context, source,&ret);
    ostringstream OSS;
    OSS << " -D IMAGESIZE_X=" << g_nx;
    OSS << " -D IMAGESIZE_Y=" << g_ny;
    OSS << " -D IMAGESIZE_M=" << g_nx*g_ny;
    OSS << " -D DEPTHSIZE=" << max(g_nx,g_ny);
    OSS << " -D PRJ_IMAGESIZE=" << g_px;
    OSS << " -D PRJ_ANGLESIZE=" << g_pa;
    OSS << " -D PRJ_IMAGESIZE_M=" << g_px*g_pa;
    OSS << " -D SS=" << g_ss;
    OSS << " -D LAMBDA_FISTA=" << CSlambda;
    string option = OSS.str();
    //kernel build
    ret=program.build(option.c_str());
    //cout<<ret<<endl;
    kernels[0].push_back(cl::Kernel::Kernel(program,"OSEM1", &ret));//0
    kernels[0].push_back(cl::Kernel::Kernel(program,"OSEM2", &ret));//1
    kernels[0].push_back(cl::Kernel::Kernel(program,"sinogramCorrection", &ret));//2
    kernels[0].push_back(cl::Kernel::Kernel(program,"ISTA", &ret));//3
    kernels[0].push_back(cl::Kernel::Kernel(program,"FISTA", &ret));//4
    
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
    for (int N = g_st; N < g_st + g_num;) {
        for (int j=0; j<plat_dev_list.contextsize(); j++) {
            if (input_th[j].joinable()) {
                input_th[j].join();
                input_th[j] = thread(OSEM_input_thread,
                               plat_dev_list.queue(j,0),kernels[j],
                               angle_buffers[j],
                               sub/*_buffers[j]*/,N,min(N+dN[j]-1,g_st-1+g_num),g_it,j);
                
                N+=dN[j];
                if (N >= g_st + g_num) break;
                
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

int FBP_execution(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                 cl::Buffer angle_buffer,
                 cl::Image2DArray reconst_img, cl::Image2DArray prj_img,
                 int dN, bool prjCorrection, unsigned int iter){
    
    try{
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
        string devicename = device.getInfo<CL_DEVICE_NAME>();
        size_t WorkGroupSize_p = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), g_px);
		size_t WorkGroupSize_n = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), g_nx);
		size_t WorkGroupSize_z = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), g_zp);
		size_t WorkGroupSize_zh= min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), g_zp/2);
        
        //float* xc;
        //xc = new float[g_zp*g_pa*dN*2];
        
        //buffer objects
        cl::ImageFormat format(CL_R,CL_FLOAT);
        cl::Image2DArray fprj_img(context,CL_MEM_READ_WRITE,format,dN,g_px,g_pa,0,0,NULL,NULL);
        cl::Buffer W_buff(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*g_zp, 0, NULL);
        cl::Buffer xc1_buff(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*g_zp*g_pa*dN, 0, NULL);
        cl::Buffer xc2_buff(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*g_zp*g_pa*dN, 0, NULL);
        
        
        cl::size_t<3> origin;
        cl::size_t<3> region_prj;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        region_prj[0] = g_px;
        region_prj[1] = g_pa;
        region_prj[2] = dN;
        
        //NDrange settings
        const cl::NDRange global_item_size0(g_zp,1,1);
        const cl::NDRange global_item_size1(g_px,g_pa,dN);
        const cl::NDRange global_item_size2(g_zp,g_pa,dN);
        const cl::NDRange global_item_size3(g_zp/2,g_pa,dN);
        const cl::NDRange global_item_size4(g_ox,g_oy,dN);
        const cl::NDRange local_item_size_p(WorkGroupSize_p,1,1);
		const cl::NDRange local_item_size_n(WorkGroupSize_n, 1, 1);
		const cl::NDRange local_item_size_z(WorkGroupSize_z, 1, 1);
		const cl::NDRange local_item_size_zh(WorkGroupSize_zh, 1, 1);
        
        
        //sinogram correction
        if(prjCorrection&&correctionMode>0){
            command_queue.enqueueCopyImage(prj_img, fprj_img, origin, origin, region_prj, NULL,NULL);
            command_queue.finish();
            kernel[8].setArg(0, fprj_img);
            kernel[8].setArg(1, prj_img);
            kernel[8].setArg(2, angle_buffer);
            kernel[8].setArg(3, (cl_int)correctionMode);
            kernel[8].setArg(4, (cl_float)0.5f);
            kernel[8].setArg(5, (cl_float)0.5f);
            command_queue.enqueueNDRangeKernel(kernel[8], NULL, global_item_size1, local_item_size_p, NULL, NULL);
            command_queue.finish();
        }
        
        //kernel set Arguments (spin factor)
        kernel[0].setArg(0, W_buff);
        //kernel set Arguments (zero padding)
        kernel[1].setArg(0, prj_img);
        kernel[1].setArg(1, xc1_buff);
        //kernel set Arguments (bit reverce)
        kernel[2].setArg(2, (cl_uint)iter);
        //kernel set Arguments (butterfly)
        kernel[3].setArg(1, W_buff); //W
        //kernel set Arguments (filter)
        kernel[4].setArg(0, xc2_buff); //xc_filter
        //kernel set Arguments (normalization)
        kernel[5].setArg(0, xc1_buff); //xc_src
        //kernel set Arguments (output image)
        kernel[6].setArg(0, xc1_buff);
        kernel[6].setArg(1, fprj_img);
        //kernel set Arguments (projection)
        kernel[7].setArg(0, fprj_img);
        kernel[7].setArg(1, reconst_img);
        kernel[7].setArg(2, angle_buffer);
        
        //create spin factor
        command_queue.enqueueNDRangeKernel(kernel[0], NULL, global_item_size0, local_item_size_z, NULL, NULL);
        command_queue.finish();
        
        //zero padding
        command_queue.enqueueFillBuffer(xc1_buff, (cl_float)0.0f, 0, sizeof(cl_float2)*g_zp*g_pa*dN,NULL,NULL);
        command_queue.enqueueNDRangeKernel(kernel[1], NULL, global_item_size1, local_item_size_p, NULL, NULL);
        command_queue.finish();
        //command_queue.enqueueReadBuffer(xc1_buff, CL_TRUE, 0, sizeof(cl_float2)*g_zp*g_pa*dN, xc);
        //write_data_output(g_d2,"zp.raw",xc,g_zp*g_pa*dN*2);
        
        //FFT
        //bit reverce
        kernel[2].setArg(0, xc1_buff); //xc_src
        kernel[2].setArg(1, xc2_buff); //xc_fft
        command_queue.enqueueNDRangeKernel(kernel[2], NULL, global_item_size2, local_item_size_z, NULL, NULL);
        command_queue.finish();
        //butterfly
        kernel[3].setArg(0, xc2_buff);
        kernel[3].setArg(2, (cl_uint)0x0); //flag
        for (unsigned int i=1; i<=iter; i++) {
            kernel[3].setArg(3, (cl_int)i); //iter
            command_queue.enqueueNDRangeKernel(kernel[3], NULL, global_item_size3, local_item_size_zh, NULL, NULL);
            command_queue.finish();
        }
        //command_queue.enqueueReadBuffer(xc2_buff, CL_TRUE, 0, sizeof(cl_float2)*g_zp*g_pa*dN, xc);
        //write_data_output(g_d2,"fft.raw",xc,g_zp*g_pa*dN*2);
        
        //filtering
        command_queue.enqueueNDRangeKernel(kernel[4], NULL, global_item_size2, local_item_size_z, NULL, NULL);
        command_queue.finish();
        //command_queue.enqueueReadBuffer(xc2_buff, CL_TRUE, 0, sizeof(cl_float2)*g_zp*g_pa*dN, xc);
        //write_data_output(g_d2,"filter.raw",xc,g_zp*g_pa*dN*2);
        
        //IFFT
        //bit reverce
        kernel[2].setArg(0, xc2_buff); //xc_src
        kernel[2].setArg(1, xc1_buff); //xc_fft
        command_queue.enqueueNDRangeKernel(kernel[2], NULL, global_item_size2, local_item_size_z, NULL, NULL);
        command_queue.finish();
        //butterfly
        kernel[3].setArg(0, xc1_buff);
        kernel[3].setArg(2, (cl_uint)0x80000000); //flag
        for (unsigned int i=1; i<=iter; i++) {
            kernel[3].setArg(3, (cl_int)i); //iter
            command_queue.enqueueNDRangeKernel(kernel[3], NULL, global_item_size3, local_item_size_zh, NULL, NULL);
            command_queue.finish();
        }
        //normaliztion
        command_queue.enqueueNDRangeKernel(kernel[5], NULL, global_item_size2, local_item_size_z, NULL, NULL);
        command_queue.finish();
        //command_queue.enqueueReadBuffer(xc1_buff, CL_TRUE, 0, sizeof(cl_float2)*g_zp*g_pa*dN, xc);
        //write_data_output(g_d2,"ifft.raw",xc,g_zp*g_pa*dN*2);
        
        //output filtered prj image
        command_queue.enqueueNDRangeKernel(kernel[6], NULL, global_item_size1, local_item_size_p, NULL, NULL);
        command_queue.finish();
        
        //back projection
        command_queue.enqueueNDRangeKernel(kernel[7], NULL, global_item_size4, local_item_size_n, NULL, NULL);
        command_queue.finish();
        
        
        //delete [] xc;
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(ret.err());
    }
    
    return 0;
}


int FBP_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
               cl::Buffer angle_buffer,vector<float*> imgs, vector<float*> prjs,
               int startN, int endN,int thread_id){
    
    try{
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        
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
        
        
        //write memory objects to GPUs
        cl::size_t<3> origin;
        cl::size_t<3> region_img;
        cl::size_t<3> region_prj;
        origin[0] = 0;
        origin[1] = 0;
        region_img[0] = g_ox;
        region_img[1] = g_oy;
        region_img[2] = 1;
        region_prj[0] = g_px;
        region_prj[1] = g_pa;
        region_prj[2] = 1;
        
        unsigned int iter = (unsigned int)(log(g_zp)/log(2));
        
        //input projection image to GPU
        for (int i=0; i<dN; i++) {
            origin[2] = i;
            command_queue.enqueueWriteImage(prj_img,CL_TRUE,origin,region_prj,sizeof(cl_float)*g_px,0,prjs[i+offsetN],NULL,NULL);
        }
        
        //execution
        FBP_execution(command_queue, kernel, angle_buffer,
                      reconst_img, prj_img, dN, true, iter);
        
        //read reconst image from GPU
        for (int i=0; i<dN; i++) {
            origin[2] = i;
            command_queue.enqueueReadImage(reconst_img,CL_TRUE,origin,region_img,sizeof(float)*g_ox,0,imgs[i],NULL,NULL);
        }
        
    }catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(ret.err());
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
        oss1<<base_f1<<setfill('0')<<setw(4)<<i<<tale_f1;
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
    
    //string kernel_code="";
    //kernel program source
    /*ifstream ifs("E:/Dropbox/CTprogram/CT_reconstruction/FBP.cl", ios::in);
     if(!ifs) {
     cerr << "   Failed to load kernel" << endl;
     return -1;
     }
     istreambuf_iterator<char> it(ifs);
     istreambuf_iterator<char> last;
     string kernel_src_fbp(it,last);
     ifs.close();*/
    
    //OpenCL Program
    cl::Program::Sources source;
#if defined (OCL120)
    source.push_back(make_pair(kernel_src_fbp.c_str(),kernel_src_fbp.length()));
    source.push_back(make_pair(kernel_src_reconstShare.c_str(),kernel_src_reconstShare.length()));
#else
    source.push_back(kernel_src_fbp);
    source.push_back(kernel_src_reconstShare);
#endif
    cl::Program program(context, source, &ret);
    ostringstream OSS;
    OSS << " -D IMAGESIZE_X=" << g_nx;
    OSS << " -D IMAGESIZE_Y=" << g_ny;
    OSS << " -D IMAGESIZE_M=" << g_nx*g_ny;
    OSS << " -D PRJ_IMAGESIZE=" << g_px;
    OSS << " -D PRJ_ANGLESIZE=" << g_pa;
    OSS << " -D ZP_SIZE=" << g_zp;
    OSS << " -D SS=" << 1;
    string option = OSS.str();
#ifdef DEBUG
    option += "-D DEBUG";
#endif
    string GPUvendor =  context.getInfo<CL_CONTEXT_DEVICES>()[0].getInfo<CL_DEVICE_VENDOR>();
    if(GPUvendor == "nvidia"){
        option += " -cl-nv-maxrregcount=64";
#ifdef DEBUG
        option += " -cl-nv-verbose -Werror";
#endif
    }
    else if (GPUvendor.find("NVIDIA Corporation")==0) {
        option += " -cl-nv-maxrregcount=64";
#ifdef DEBUG
        option += " -cl-nv-verbose -Werror";
#endif
    }
    else {
#ifdef DEBUG
        option += " -Werror";
#endif
    }
    //kernel build
    ret=program.build(option.c_str());
    //cout<<ret<<endl;
    kernels[0].push_back(cl::Kernel::Kernel(program,"spinFactor", &ret));//0
    kernels[0].push_back(cl::Kernel::Kernel(program,"zeroPadding", &ret));//1
    kernels[0].push_back(cl::Kernel::Kernel(program,"bitReverse", &ret));//2
    kernels[0].push_back(cl::Kernel::Kernel(program,"butterfly", &ret));//3
    kernels[0].push_back(cl::Kernel::Kernel(program,"filtering", &ret));//4
    kernels[0].push_back(cl::Kernel::Kernel(program,"normalization", &ret));//5
    kernels[0].push_back(cl::Kernel::Kernel(program,"outputImage", &ret));//6
    kernels[0].push_back(cl::Kernel::Kernel(program,"backProjectionFBP", &ret));//7
    kernels[0].push_back(cl::Kernel::Kernel(program,"sinogramCorrection", &ret));//8
    kernels[0].push_back(cl::Kernel::Kernel(program,"setThreshold", &ret));//9
    kernels[0].push_back(cl::Kernel::Kernel(program,"findMinimumX", &ret));//10
    kernels[0].push_back(cl::Kernel::Kernel(program,"findMinimumY", &ret));//11
    kernels[0].push_back(cl::Kernel::Kernel(program,"baseUp", &ret));//12
    
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
            size_t device_pram_size[3]={0,0,0};
            plat_dev_list.dev(i,j).getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &device_pram_size);
			dN.push_back(min(numParallel, (int)device_pram_size[0]));
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
    for (int N = g_st; N < g_st + g_num;) {
        for (int j=0; j<queues.size(); j++) {
            if (input_th[j].joinable()) {
                input_th[j].join();
                input_th[j] = thread(FBP_input_thread,
                                     queues[j],kernels[cotextID_OfQueue[j]],
                                     angle_buffers[cotextID_OfQueue[j]],
                                     N,min(N+dN[j]-1,g_st-1+g_num),j);
                
                N+=dN[j];
                if (N >= g_st + g_num) break;
                
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

int AART_execution(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                   cl::Buffer angle_buffer, int *sub,
                   cl::Image2DArray reconst_img, cl::Image2DArray prj_img,
                   int dN, int it){
    
    try {
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
        string devicename = device.getInfo<CL_DEVICE_NAME>();
        size_t WorkGroupSize_p = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), g_px);
		size_t WorkGroupSize_n = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), g_nx);
        
        //OpenCL memory objects
        cl::ImageFormat format(CL_R,CL_FLOAT);
        cl::Image2DArray reconst_dest_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_ny,0,0,NULL,NULL);
        cl::Image2DArray rprj_img(context,CL_MEM_READ_WRITE,format,dN,g_px,g_pa,0,0,NULL,NULL);
        
        //write memory objects to GPUs
        cl::size_t<3> origin;
        cl::size_t<3> origin_img;
        cl::size_t<3> origin_img2;
        cl::size_t<3> region;
        cl::size_t<3> region2;
        cl::size_t<3> region_prj;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        origin_img[0] = (g_nx-g_ox)/2;
        origin_img[1] = (g_ny-g_oy)/2;
        origin_img[2] = 0;
        origin_img2[1] = (g_ny-g_oy)/2;
        origin_img2[2] = 0;
        region[0] = g_ox;
        region[1] = g_oy;
        region[2] = dN;
        region2[1] = g_oy;
        region2[2] = dN;
        region_prj[0] = g_nx;
        region_prj[1] = g_pa;
        region_prj[2] = dN;
        command_queue.enqueueCopyImage(reconst_img, reconst_dest_img, origin_img, origin_img, region, NULL,NULL);
        
        //NDrange settings
        const cl::NDRange global_item_size0(g_px,g_pa,dN);
        const cl::NDRange global_item_size1(g_px,g_pa/g_ss,dN);
        const cl::NDRange global_item_size2(g_ox,g_oy,dN);
        const cl::NDRange local_item_size_p(WorkGroupSize_p,1,1);
		const cl::NDRange local_item_size_n(WorkGroupSize_n, 1, 1);
        
        //sinogram correction
        if(correctionMode>0){
            command_queue.enqueueCopyImage(prj_img, rprj_img, origin, origin, region_prj, NULL,NULL);
            command_queue.finish();
            kernel[2].setArg(0, rprj_img);
            kernel[2].setArg(1, prj_img);
            kernel[2].setArg(2, angle_buffer);
            kernel[2].setArg(3, (cl_int)correctionMode);
            kernel[2].setArg(4, (cl_float)0.5f);
            kernel[2].setArg(5, (cl_float)0.5f);
            command_queue.enqueueNDRangeKernel(kernel[2], NULL, global_item_size0, local_item_size_p, NULL, NULL);
            command_queue.finish();
        }
        
        //projection
        kernel[0].setArg(0, reconst_img);
        kernel[0].setArg(1, prj_img);
        kernel[0].setArg(2, rprj_img);
        kernel[0].setArg(3, angle_buffer);
        //back projection
        kernel[1].setArg(0, reconst_img);
        kernel[1].setArg(1, reconst_dest_img);
        kernel[1].setArg(2, rprj_img);
        kernel[1].setArg(3, angle_buffer);
        if (g_mode==3) {
            kernel[1].setArg(5, (cl_float)g_wt2);
        }else{
            kernel[1].setArg(5, (cl_float)g_wt1);
        }

		//(normal) back projection
		kernel[4].setArg(0, reconst_dest_img);
		kernel[4].setArg(1, prj_img);
		kernel[4].setArg(2, angle_buffer);
		kernel[4].setArg(3, sub[0]);
		command_queue.enqueueNDRangeKernel(kernel[4], NULL, global_item_size2, local_item_size_n, NULL, NULL);
		command_queue.finish();

		command_queue.enqueueCopyImage(reconst_dest_img, reconst_img, origin_img, origin_img, region, NULL, NULL);
		command_queue.finish();
        for (int i=0; i<it; i++) {
            for (int k=0; k<g_ss; k++) {
                //extent reconst image
                for (int org=(g_nx-3*g_ox)/2; org>=-g_ox/2; org-=g_ox) {
                    origin_img2[0] = max(0,org);
                    region2[0] = org + g_ox - max(0, org);
                    command_queue.enqueueCopyImage(reconst_dest_img, reconst_img, origin_img, origin_img2, region2, NULL,NULL);
                }
                for (int org=(g_nx+g_ox)/2; org<g_nx; org+=g_ox) {
                    origin_img2[0] = org;
                    region2[0] = min(org+g_ox,g_nx) - org;
                    command_queue.enqueueCopyImage(reconst_dest_img, reconst_img, origin_img, origin_img2, region2, NULL,NULL);
                }
                
                
                //projection
                kernel[0].setArg(4, (cl_int)sub[k]);
                command_queue.enqueueNDRangeKernel(kernel[0], NULL, global_item_size1, local_item_size_p, NULL, NULL);
                command_queue.finish();
                
                //back projection
                kernel[1].setArg(4, (cl_int)sub[k]);
                command_queue.enqueueNDRangeKernel(kernel[1], NULL, global_item_size2, local_item_size_n, NULL, NULL);
                command_queue.finish();
                
                command_queue.enqueueCopyImage(reconst_dest_img, reconst_img, origin_img, origin_img, region,NULL,NULL);
                command_queue.finish();
                
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

int AART_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
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
    cl::Image2DArray reconst_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_ny,0,0,NULL,NULL);
    cl::Image2DArray prj_img(context,CL_MEM_READ_WRITE,format,dN,g_px,g_pa,0,0,NULL,NULL);
    
    //write memory objects to GPUs
    cl::size_t<3> origin_img;
    cl::size_t<3> origin_prj;
    cl::size_t<3> region_img;
    cl::size_t<3> region_prj;
    origin_img[0] = (g_nx-g_ox)/2;
    origin_img[1] = (g_ny-g_oy)/2;
    origin_prj[0] = 0;
    origin_prj[1] = 0;
    region_img[0] = g_ox;
    region_img[1] = g_oy;
    region_img[2] = 1;
    region_prj[0] = g_px;
    region_prj[1] = g_pa;
    region_prj[2] = 1;
    for (int i=0; i<dN; i++) {
        origin_img[2] = i;
        origin_prj[2] = i;
        command_queue.enqueueWriteImage(reconst_img, CL_TRUE, origin_img, region_img, 0,0,imgs[i],NULL,NULL);
        command_queue.enqueueWriteImage(prj_img,CL_TRUE,origin_prj,region_prj,0,0,prjs[i+offsetN],NULL,NULL);
    }
    
    
    //AART execution
    AART_execution(command_queue, kernel,angle_buffer,sub,reconst_img,prj_img,dN,it);
    
    
    //read memory objects from GPUs
    for (int i=0; i<dN; i++) {
        origin_img[2] = i;
        command_queue.enqueueReadImage(reconst_img,CL_TRUE,origin_img,region_img,0,0,imgs[i],NULL,NULL);
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


int AART_input_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                      cl::Buffer angle_buffer, /*cl::Buffer*/int *sub/*_buffer*/,
                      int startN, int endN, int it, int thread_id){
    
    //read data
    vector<float*> imgs;
    vector<float*> prjs;
    for (int i=startN; i<=endN; i++) {
        prjs.push_back(new float[(unsigned long)g_px*g_pa]);
        imgs.push_back(new float[(unsigned long)g_ox*g_oy]);
    }
    
    for (int i=startN; i<=endN; i++) {
        ostringstream oss1;
        oss1<<base_f1<<setfill('0')<<setw(4)<<i<<tale_f1;
        string f1=oss1.str();
        read_data_input(g_d1,f1, prjs[i-startN], g_px*g_pa);
        first_image(g_f4, imgs[i-startN], g_ox*g_oy);
    }
    
    reconst_th[thread_id].join();
    reconst_th[thread_id]=thread(AART_thread,command_queue,kernel,
                                 angle_buffer, sub,
                                 move(imgs), move(prjs),
                                 startN, endN, it, thread_id);
    
    return 0;
}

int AART_programBuild(cl::Context context,vector<cl::Kernel> *kernels){
    cl_int ret;
    
    //kernel program source
    /*ifstream ifs("./AART.cl", ios::in);
     if(!ifs) {
     cerr << "   Failed to load kernel \n" << endl;
     return -1;
     }
     istreambuf_iterator<char> it(ifs);
     istreambuf_iterator<char> last;
     string kernel_src_osem(it,last);
     ifs.close();*/
    
    cl::Program::Sources source;
#if defined (OCL120)
    source.push_back(make_pair(kernel_src_aart.c_str(),kernel_src_aart.length()));
    source.push_back(make_pair(kernel_src_reconstShare.c_str(),kernel_src_reconstShare.length()));
#else
    source.push_back(kernel_src_aart);
    source.push_back(kernel_src_reconstShare);
#endif
    cl::Program program(context, source,&ret);
    
    ostringstream OSS;
    OSS << " -D IMAGESIZE_X=" << g_nx;
    OSS << " -D IMAGESIZE_Y=" << g_ny;
    OSS << " -D IMAGESIZE_M=" << g_nx*g_ny;
    OSS << " -D DEPTHSIZE=" << max(g_nx,g_ny);
    OSS << " -D PRJ_IMAGESIZE=" << g_px;
    OSS << " -D PRJ_ANGLESIZE=" << g_pa;
    OSS << " -D PRJ_IMAGESIZE_M=" << g_px*g_pa;
    OSS << " -D SS=" << g_ss;
    string option = OSS.str();
    //kernel build
    ret=program.build(option.c_str());
    //cout<<ret<<endl;
    kernels[0].push_back(cl::Kernel::Kernel(program,"AART1", &ret));//0
    kernels[0].push_back(cl::Kernel::Kernel(program,"AART2", &ret));//1
    kernels[0].push_back(cl::Kernel::Kernel(program,"sinogramCorrection", &ret));//2
    kernels[0].push_back(cl::Kernel::Kernel(program,"partialDerivativeOfGradiant", &ret));//3
	kernels[0].push_back(cl::Kernel::Kernel(program, "backProjection", &ret));//4
    
    return 0;
}


int AART_ocl(OCL_platform_device plat_dev_list, float *ang, int ss){
    
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
        AART_programBuild(plat_dev_list.context(i), &kernels_plat);
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
    for (int N = g_st; N < g_st + g_num;) {
        for (int j=0; j<plat_dev_list.contextsize(); j++) {
            if (input_th[j].joinable()) {
                input_th[j].join();
                input_th[j] = thread(AART_input_thread,
                                     plat_dev_list.queue(j,0),kernels[j],
                                     angle_buffers[j],
                                     sub/*_buffers[j]*/,N,min(N+dN[j]-1,g_st-1+g_num),g_it,j);
                
                N+=dN[j];
                if (N >= g_st + g_num) break;
                
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

int hybrid_thread(cl::CommandQueue command_queue,
                  vector<cl::Kernel> kernel_fbp,vector<cl::Kernel> kernel_osem,
                  cl::Buffer angle_buffer, int *sub,
                  vector<float*> imgs, vector<float*> prjs,
                  int startN, int endN, int it, int thread_id){
    
    try{
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
        string devicename = device.getInfo<CL_DEVICE_NAME>();
        size_t WorkGroupSize_n = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), g_nx);
        
        int dN = endN-startN+1;
#ifdef BATCHEXE
        int offsetN=startN;
#else
        int offsetN=0;
#endif
        
        //OpenCL memory objects
        cl::ImageFormat format(CL_R,CL_FLOAT);
        cl::Image2DArray reconst_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_nx,0,0,NULL,NULL);
        cl::Image2DArray reconst_dummy_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_nx,0,0,NULL,NULL);
        cl::Image2DArray prj_img(context,CL_MEM_READ_WRITE,format,dN,g_px,g_pa,0,0,NULL,NULL);
        
        
        //write memory objects to GPUs
        cl::size_t<3> origin;
        cl::size_t<3> origin_img;
        cl::size_t<3> region_img;
        cl::size_t<3> region_img2;
        cl::size_t<3> region_prj;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        origin_img[0] = (g_nx-g_ox)/2;
        origin_img[1] = (g_ny-g_oy)/2;
        origin_img[2] = 0;
        region_img[0] = g_ox;
        region_img[1] = g_oy;
        region_img[2] = 1;
        region_img2[0] = g_ox;
        region_img2[1] = g_oy;
        region_img2[2] = dN;
        region_prj[0] = g_px;
        region_prj[1] = g_pa;
        region_prj[2] = 1;
        
        unsigned int iter = (unsigned int)(log(g_zp)/log(2));
        
        //input projection image to GPU
        for (int i=0; i<dN; i++) {
            origin[2] = i;
            origin_img[2] = i;
            command_queue.enqueueWriteImage(prj_img,CL_TRUE,origin,region_prj,sizeof(cl_float)*g_px,0,prjs[i+offsetN],NULL,NULL);
            command_queue.enqueueWriteImage(reconst_img, CL_TRUE, origin_img, region_img, 0,0,imgs[i],NULL,NULL);
        }
        
        
        //fbp execution
        FBP_execution(command_queue, kernel_fbp, angle_buffer,
                      reconst_img, prj_img, dN, true, iter);
        
        int FBPcorrection=1;
        origin_img[2]=0;
        const cl::NDRange global_item_size0(g_nx,g_ny,dN);
        const cl::NDRange global_item_size1(WorkGroupSize_n,g_ny,dN);
        const cl::NDRange global_item_size1_1(WorkGroupSize_n,dN,1);
        const cl::NDRange local_item_size_n(WorkGroupSize_n,1,1);
		switch (FBPcorrection) {
		case 0:
			//set threshold
			command_queue.enqueueCopyImage(reconst_img, reconst_dummy_img, origin_img, origin_img, region_img2);
			kernel_fbp[9].setArg(0, reconst_dummy_img);
			kernel_fbp[9].setArg(1, reconst_img);
			kernel_fbp[9].setArg(2, (cl_float)1e-10f);
			command_queue.enqueueNDRangeKernel(kernel_fbp[9], NULL, global_item_size0, local_item_size_n, NULL, NULL);
			command_queue.finish();
			break;


		case 1:
			//base up
			command_queue.enqueueCopyImage(reconst_img, reconst_dummy_img, origin_img, origin_img, region_img2);
			cl::Buffer minimumY(context, CL_MEM_READ_WRITE, sizeof(cl_float)*g_nx*dN, 0, NULL);
			cl::Buffer minimum(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dN, 0, NULL);
			//findMinimumX
			kernel_fbp[10].setArg(0, reconst_dummy_img);
			kernel_fbp[10].setArg(1, cl::Local(sizeof(cl_float)*WorkGroupSize_n));
			kernel_fbp[10].setArg(2, minimumY);
			command_queue.enqueueNDRangeKernel(kernel_fbp[10], NULL, global_item_size1, local_item_size_n, NULL, NULL);
			command_queue.finish();

			//findMinimumY
			kernel_fbp[11].setArg(0, minimumY);
			kernel_fbp[11].setArg(1, cl::Local(sizeof(cl_float)*WorkGroupSize_n));
			kernel_fbp[11].setArg(2, minimum);
			command_queue.enqueueNDRangeKernel(kernel_fbp[11], NULL, global_item_size1_1, local_item_size_n, NULL, NULL);
			command_queue.finish();

			/*float* minimum_p;
			minimum_p = new float[dN];
			command_queue.enqueueReadBuffer(minimum, CL_TRUE, 0, dN, minimum_p, NULL, NULL);
			for (int l = 0; l < dN; l++) {
				cout << minimum_p[l]<<",";
			}
			cout << endl;*/

			//baseup
			kernel_fbp[12].setArg(0, reconst_dummy_img);
			kernel_fbp[12].setArg(1, reconst_img);
			kernel_fbp[12].setArg(2, minimum);
			kernel_fbp[12].setArg(3, (cl_int)baseupOrder);
			command_queue.enqueueNDRangeKernel(kernel_fbp[12], NULL, global_item_size0, local_item_size_n, NULL, NULL);
			command_queue.finish();
			break;
		}
        
        //OSEM execution
        if (CSitBool) {
            FISTA_OSEM_execution(command_queue, kernel_osem, angle_buffer,sub,reconst_img,prj_img,dN,it,false);
        }else{
            OSEM_execution(command_queue, kernel_osem, angle_buffer,sub,reconst_img,prj_img,dN,it,false);
        }
        
        //read reconst image from GPU
        for (int i=0; i<dN; i++) {
            origin_img[2] = i;
            command_queue.enqueueReadImage(reconst_img,CL_TRUE,origin_img,region_img,sizeof(float)*g_nx,0,imgs[i],NULL,NULL);
        }
        
    }catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(ret.err());
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

int hybrid_input_thread(cl::CommandQueue command_queue,
                        vector<cl::Kernel> kernel_fbp,vector<cl::Kernel> kernel_osem,
                        cl::Buffer angle_buffer, /*cl::Buffer*/int *sub/*_buffer*/,
                        int startN, int endN, int it, int thread_id){
    
    //read data
    vector<float*> imgs;
    vector<float*> prjs;
    for (int i=startN; i<=endN; i++) {
        prjs.push_back(new float[(unsigned long)g_px*g_pa]);
        imgs.push_back(new float[(unsigned long)g_ox*g_oy]);
    }
    
    for (int i=startN; i<=endN; i++) {
        ostringstream oss1;
        oss1<<base_f1<<setfill('0')<<setw(4)<<i<<tale_f1;
        string f1=oss1.str();
        read_data_input(g_d1,f1, prjs[i-startN], g_px*g_pa);
        first_image(g_f4, imgs[i-startN], g_ox*g_oy);
    }
    
    reconst_th[thread_id].join();
    reconst_th[thread_id]=thread(hybrid_thread,command_queue,
                                 kernel_fbp,kernel_osem,
                                 angle_buffer, sub,
                                 move(imgs), move(prjs),
                                 startN, endN, it, thread_id);
    
    return 0;
}

int hybrid_ocl(OCL_platform_device plat_dev_list, float *ang, int ss){
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
    vector<vector<cl::Kernel>> kernels_fbp;
    vector<vector<cl::Kernel>> kernels_osem;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        vector<cl::Kernel> kernels_plat_fbp;
        FBP_programBuild(plat_dev_list.context(i), &kernels_plat_fbp);
        kernels_fbp.push_back(kernels_plat_fbp);
        vector<cl::Kernel> kernels_plat_osem;
        OSEM_programBuild(plat_dev_list.context(i), &kernels_plat_osem);
        kernels_osem.push_back(kernels_plat_osem);
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
	delete[] ang;

    
    //thread start
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        input_th.push_back(thread(dummy));
        reconst_th.push_back(thread(dummy));
        output_th.push_back(thread(dummy));
    }
    for (int N = g_st; N < g_st + g_num;) {
        for (int j=0; j<queues.size(); j++) {
            if (input_th[j].joinable()) {
                input_th[j].join();
                input_th[j] = thread(hybrid_input_thread,queues[j],
                                     kernels_fbp[cotextID_OfQueue[j]],
                                     kernels_osem[cotextID_OfQueue[j]],
                                     angle_buffers[cotextID_OfQueue[j]],
                                     sub,N,min(N+dN[j]-1,g_st-1+g_num),g_it,j);
                
                N+=dN[j];
                if (N >= g_st + g_num) break;
                
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
    
    
    return 0;
}

int FISTA_execution(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                   cl::Buffer angle_buffer,cl::Buffer L2norm_buffer,
                   int *sub, cl::Image2DArray reconst_img, cl::Image2DArray prj_img,
                   int dN, int it, bool prjCorrection){
    
    try {
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
        string devicename = device.getInfo<CL_DEVICE_NAME>();
        size_t WorkGroupSize = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), g_ox);
        
        //OpenCL memory objects
        cl::ImageFormat format(CL_R,CL_FLOAT);
        cl::Image2DArray reconst_v_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_ny,0,0,NULL,NULL);
        cl::Image2DArray reconst_w_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_ny,0,0,NULL,NULL);
        cl::Image2DArray beta_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_ny,0,0,NULL,NULL);
        cl::Image2DArray reconst_dest_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_ny,0,0,NULL,NULL);
        cl::Image2DArray beta_dest_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_ny,0,0,NULL,NULL);
        cl::Image2DArray vprj_img(context,CL_MEM_READ_WRITE,format,dN,g_px,g_pa,0,0,NULL,NULL);
        
        //write memory objects to GPUs
        cl::size_t<3> origin;
        cl::size_t<3> origin_img;
        cl::size_t<3> origin_img2;
        cl::size_t<3> region;
        cl::size_t<3> region2;
        cl::size_t<3> region_prj;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        origin_img[0] = (g_nx-g_ox)/2;
        origin_img[1] = (g_ny-g_oy)/2;
        origin_img[2] = 0;
        origin_img2[1] = (g_ny-g_oy)/2;
        origin_img2[2] = 0;
        region[0] = g_ox;
        region[1] = g_oy;
        region[2] = dN;
        region2[1] = g_oy;
        region2[2] = dN;
        region_prj[0] = g_nx;
        region_prj[1] = g_pa;
        region_prj[2] = dN;
        cl_float4 color = {0.0f,0.0f,0.0f,1.0f};
        command_queue.enqueueCopyImage(reconst_img, reconst_w_img, origin_img, origin_img, region, NULL,NULL);
        command_queue.enqueueFillImage(beta_img, color, origin_img, region);
        
        //NDrange settings
        const cl::NDRange global_item_size0(g_px,g_pa,dN);
        const cl::NDRange global_item_size1(g_px,g_pa/g_ss,dN);
        const cl::NDRange global_item_size2(g_ox,g_oy,dN);
        const cl::NDRange local_item_size(WorkGroupSize,1,1);
        
        //sinogram correction
        if(prjCorrection&&correctionMode>0){
            command_queue.enqueueCopyImage(prj_img, vprj_img, origin, origin, region_prj, NULL,NULL);
            command_queue.finish();
            kernel[9].setArg(0, vprj_img);
            kernel[9].setArg(1, prj_img);
            kernel[9].setArg(2, angle_buffer);
            kernel[9].setArg(3, (cl_int)correctionMode);
            kernel[9].setArg(4, (cl_float)0.5f);
            kernel[9].setArg(5, (cl_float)0.5f);
            command_queue.enqueueNDRangeKernel(kernel[8], NULL, global_item_size0, local_item_size, NULL, NULL);
            command_queue.finish();
        }
        
		/*float *image;
		image = new float[IMAGE_SIZE_M*dN];*/

        //set arguments
        //AART1 (projection)
        kernel[5].setArg(0, reconst_img); //as x[0] img -> w[t] img
        kernel[5].setArg(1, prj_img);
        kernel[5].setArg(2, vprj_img);
        kernel[5].setArg(3, angle_buffer);
        //FISTA1 (back projection)
        kernel[6].setArg(0, reconst_img); //as x[0] img -> w[t] img
        kernel[6].setArg(1, reconst_v_img);
        kernel[6].setArg(2, vprj_img);
        kernel[6].setArg(3, angle_buffer);
        kernel[6].setArg(4, L2norm_buffer);
        kernel[6].setArg(6, (cl_float)g_wt1);
        //ISTA (update for 0th cycle)
        kernel[7].setArg(0, reconst_v_img);
        kernel[7].setArg(1, reconst_dest_img); //as x[1] img
        kernel[7].setArg(3, L2norm_buffer);
        //FISTA (update)
        kernel[8].setArg(0, reconst_img); //as x[t] img -> w[t] img
        kernel[8].setArg(1, reconst_v_img);
        kernel[8].setArg(2, beta_img); //as b[t] img
        kernel[8].setArg(3, reconst_w_img); //as w[t+1] img
        kernel[8].setArg(4, beta_dest_img); //as x[t+1] img
        kernel[8].setArg(5, beta_dest_img); //as beta[t+1] img
        kernel[8].setArg(7, L2norm_buffer);
		
        
        //normal back projection (estimate initial x image by simple back projection)
		kernel[10].setArg(0, reconst_img);
		kernel[10].setArg(1, prj_img);
		kernel[10].setArg(2, angle_buffer);
		kernel[10].setArg(3, sub[0]);
		command_queue.enqueueNDRangeKernel(kernel[10], NULL, global_item_size2, local_item_size, NULL, NULL);
		command_queue.finish();
        command_queue.enqueueCopyImage(reconst_img, reconst_w_img, origin_img, origin_img, region, NULL, NULL);
        command_queue.finish();
		
        
        bool firstBool=true;
        for (int i=0; i<it; i++) {
            for (int k=0; k<g_ss; k++) {
                //extent reconst image
                for (int org=(g_nx-3*g_ox)/2; org>=-g_ox/2; org-=g_ox) {
                    origin_img2[0] = max(0,org);
                    region2[0] = org + g_ox - max(0, org);
                    command_queue.enqueueCopyImage(reconst_dest_img, reconst_img, origin_img, origin_img2, region2, NULL,NULL);
                }
                for (int org=(g_nx+g_ox)/2; org<g_nx; org+=g_ox) {
                    origin_img2[0] = org;
                    region2[0] = min(org+g_ox,g_nx) - org;
                    command_queue.enqueueCopyImage(reconst_dest_img, reconst_img, origin_img, origin_img2, region2, NULL,NULL);
                }
                
                //AART1 (projection)
                kernel[5].setArg(4, sub[k]);
                command_queue.enqueueNDRangeKernel(kernel[5], NULL, global_item_size1, local_item_size, NULL, NULL);
                command_queue.finish();
                
                //FISTA back projection
                kernel[6].setArg(5, sub[k]);
                command_queue.enqueueNDRangeKernel(kernel[6], NULL, global_item_size2, local_item_size, NULL, NULL);
                command_queue.finish();
                
                if (firstBool) {
                    //ISTA (update for 0th cycle)
                    kernel[7].setArg(2, sub[k]);
                    command_queue.enqueueNDRangeKernel(kernel[7], NULL, global_item_size2, local_item_size, NULL, NULL);
                    
                    //after 0th cycle, iterlation preceeds with w_img (reconst_w_img), not with x_img (reconst_img)
                    firstBool=false;
                    kernel[5].setArg(0, reconst_w_img);
                    kernel[6].setArg(0, reconst_w_img);
                }else{
                    //FISTA (update)
                    kernel[8].setArg(6, sub[k]);
                    command_queue.enqueueNDRangeKernel(kernel[8], NULL, global_item_size2, local_item_size, NULL, NULL);
                    
                    //update(copy) of beta image
                    command_queue.enqueueCopyImage(beta_dest_img, beta_img, origin_img, origin_img, region, NULL, NULL);
                    command_queue.finish();
                }
                //update(copy) of x image (reconst_img)
                command_queue.enqueueCopyImage(reconst_dest_img, reconst_img, origin_img, origin_img, region, NULL, NULL);
                command_queue.finish();
                
            }
        }
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
    
    return 0;
}


int FISTA_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                cl::Buffer angle_buffer, cl::Buffer L2norm_buffer,
                int *sub, vector<float*> imgs, vector<float*> prjs,
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
    cl::Image2DArray reconst_img(context,CL_MEM_READ_WRITE,format,dN,g_nx,g_ny,0,0,NULL,NULL);
    cl::Image2DArray prj_img(context,CL_MEM_READ_WRITE,format,dN,g_px,g_pa,0,0,NULL,NULL);
    
    //write memory objects to GPUs
    cl::size_t<3> origin_img;
    cl::size_t<3> origin_prj;
    cl::size_t<3> region_img;
    cl::size_t<3> region_prj;
    origin_img[0] = (g_nx-g_ox)/2;
    origin_img[1] = (g_ny-g_oy)/2;
    origin_prj[0] = 0;
    origin_prj[1] = 0;
    region_img[0] = g_ox;
    region_img[1] = g_oy;
    region_img[2] = 1;
    region_prj[0] = g_px;
    region_prj[1] = g_pa;
    region_prj[2] = 1;
    for (int i=0; i<dN; i++) {
        origin_img[2] = i;
        origin_prj[2] = i;
        command_queue.enqueueWriteImage(reconst_img, CL_TRUE, origin_img, region_img, 0,0,imgs[i],NULL,NULL);
        command_queue.enqueueWriteImage(prj_img,CL_TRUE,origin_prj,region_prj,0,0,prjs[i+offsetN],NULL,NULL);
    }
    
    
    //ISTA execution
    FISTA_execution(command_queue, kernel,angle_buffer,L2norm_buffer,sub,reconst_img,prj_img,dN,it,true);
    
    
    //read memory objects from GPUs
    for (int i=0; i<dN; i++) {
        origin_img[2] = i;
        command_queue.enqueueReadImage(reconst_img,CL_TRUE,origin_img,region_img,0,0,imgs[i],NULL,NULL);
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

int FISTA_input_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                      cl::Buffer angle_buffer, cl::Buffer L2norm_buffer,
                      int *sub, int startN, int endN, int it, int thread_id){
    
    //read data
    vector<float*> imgs;
    vector<float*> prjs;
    for (int i=startN; i<=endN; i++) {
        prjs.push_back(new float[(unsigned long)g_px*g_pa]);
        imgs.push_back(new float[(unsigned long)g_ox*g_oy]);
    }
    
    for (int i=startN; i<=endN; i++) {
        ostringstream oss1;
        oss1<<base_f1<<setfill('0')<<setw(4)<<i<<tale_f1;
        string f1=oss1.str();
        read_data_input(g_d1,f1, prjs[i-startN], g_px*g_pa);
        first_image(g_f4, imgs[i-startN], g_ox*g_oy);
    }
    
    reconst_th[thread_id].join();
    reconst_th[thread_id]=thread(FISTA_thread,command_queue,kernel,
                                 angle_buffer, L2norm_buffer,
                                 sub, move(imgs), move(prjs),
                                 startN, endN, it, thread_id);
    
    return 0;
}

int FISTA_programBuild(cl::Context context,vector<cl::Kernel> *kernels){
    cl_int ret;
    
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
    
    cl::Program::Sources source;
#if defined (OCL120)
    //source.push_back(make_pair(kernel_src_osem.c_str(),kernel_src_osem.length()));
    source.push_back(make_pair(kernel_src_aart.c_str(),kernel_src_aart.length()));
    source.push_back(make_pair(kernel_src_fista.c_str(),kernel_src_fista.length()));
    source.push_back(make_pair(kernel_src_reconstShare.c_str(),kernel_src_reconstShare.length()));
#else
    //source.push_back(kernel_src_osem);
    source.push_back(kernel_src_aart);
    source.push_back(kernel_src_fista);
    source.push_back(kernel_src_reconstShare);
#endif
    cl::Program program(context, source,&ret);
    ostringstream OSS;
    OSS << " -D IMAGESIZE_X=" << g_nx;
    OSS << " -D IMAGESIZE_Y=" << g_ny;
    OSS << " -D IMAGESIZE_M=" << g_nx*g_ny;
    OSS << " -D DEPTHSIZE=" << max(g_nx,g_ny);
    OSS << " -D PRJ_IMAGESIZE=" << g_px;
    OSS << " -D PRJ_ANGLESIZE=" << g_pa;
    OSS << " -D PRJ_IMAGESIZE_M=" << g_px*g_pa;
    OSS << " -D SS=" << g_ss;
    OSS << " -D LAMBDA_FISTA=" << CSlambda;
    string option = OSS.str();
    //kernel build
    ret=program.build(option.c_str());
    //cout<<ret<<endl;
    kernels[0].push_back(cl::Kernel::Kernel(program,"powerIter1", &ret));//0
    kernels[0].push_back(cl::Kernel::Kernel(program,"powerIter2", &ret));//1
    kernels[0].push_back(cl::Kernel::Kernel(program,"powerIter3", &ret));//2
    kernels[0].push_back(cl::Kernel::Kernel(program,"imageL2AbsX", &ret));//3
    kernels[0].push_back(cl::Kernel::Kernel(program,"imageL2AbsY", &ret));//4
    kernels[0].push_back(cl::Kernel::Kernel(program,"AART1", &ret));//5
    kernels[0].push_back(cl::Kernel::Kernel(program,"FISTAbackProjection", &ret));//6
    kernels[0].push_back(cl::Kernel::Kernel(program,"ISTA", &ret));//7
    kernels[0].push_back(cl::Kernel::Kernel(program,"FISTA", &ret));//8
    kernels[0].push_back(cl::Kernel::Kernel(program,"sinogramCorrection", &ret));//9
	kernels[0].push_back(cl::Kernel::Kernel(program,"backProjection", &ret));//10
    
    return 0;
}

int powerIteration(cl::CommandQueue command_queue, vector<cl::Kernel> kernels,
                   cl::Buffer angle_buffer, cl::Buffer L2norm_buffer){
    
    try{
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        cl::Device device = command_queue.getInfo<CL_QUEUE_DEVICE>();
        string devicename = device.getInfo<CL_DEVICE_NAME>();
        size_t WorkGroupSize = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), g_ox);
        
        
        cl::size_t<3> origin;
        cl::size_t<3> region;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        region[0] = g_ox;
        region[1] = g_oy;
        region[2] = g_ss;
        cl_float4 color = {1.0f,0.0f,0.0f,0.0f};
        cl::ImageFormat format(CL_R,CL_FLOAT);
        
        const cl::NDRange global_item_size1(g_px,g_pa/g_ss,g_ss);
        const cl::NDRange global_item_size2(g_ox,g_oy,g_ss);
        const cl::NDRange local_item_size(WorkGroupSize,1,1);
        const cl::NDRange global_item_size3(WorkGroupSize,g_oy,g_ss);
        const cl::NDRange global_item_size4(WorkGroupSize,g_ss,1);
        
        cl::Image2DArray reconst_img(context,CL_MEM_READ_WRITE,format,g_ss,g_nx,g_ny,0,0,NULL,NULL);
        cl::Image2DArray reconst_cnd_img(context,CL_MEM_READ_WRITE,format,g_ss,g_nx,g_ny,0,0,NULL,NULL);
        cl::Image2DArray prj_img(context,CL_MEM_READ_WRITE,format,g_ss,g_px,g_pa,0,0,NULL,NULL);
        cl::Buffer L2normY(context, CL_MEM_READ_WRITE, sizeof(cl_float)*g_ny*g_ss, 0, NULL);
        
        command_queue.enqueueFillImage(reconst_img, color, origin, region);
        command_queue.finish();
        
        //power iteration 1
        kernels[0].setArg(0,reconst_img);
        kernels[0].setArg(1,prj_img);
        kernels[0].setArg(2,angle_buffer);
        //power iteration 2
        kernels[1].setArg(0,reconst_cnd_img);
        kernels[1].setArg(1,prj_img);
        kernels[1].setArg(2,angle_buffer);
        //power iteration 3
        kernels[2].setArg(0,reconst_cnd_img);
        kernels[2].setArg(1,reconst_img);
        kernels[2].setArg(2,L2norm_buffer);
        //L2-norm 1
        kernels[3].setArg(0,reconst_cnd_img);
        kernels[3].setArg(1,cl::Local(sizeof(cl_float)*WorkGroupSize));
        kernels[3].setArg(2,L2normY);
        //L2-norm 2
        kernels[4].setArg(0,L2normY);
        kernels[4].setArg(1,cl::Local(sizeof(cl_float)*WorkGroupSize));
        kernels[4].setArg(2,L2norm_buffer);
        
        /*ofstream ofs("E:/eval_list.txt", ios::out | ios::trunc);
        float *image;
        image = new float[IMAGE_SIZE_M*g_ss];
        float *L2;
        L2 = new float[g_ss];*/
        
        for (int trial=0; trial<30; trial++) {
            
			//power iteration 1
            command_queue.enqueueNDRangeKernel(kernels[0], NULL, global_item_size1,local_item_size, NULL, NULL);
            command_queue.finish();
            
            //power iteration 2
            command_queue.enqueueNDRangeKernel(kernels[1], NULL, global_item_size2,local_item_size, NULL, NULL);
            command_queue.finish();
            
            //L2-norm 1
            command_queue.enqueueNDRangeKernel(kernels[3], NULL, global_item_size3,local_item_size, NULL, NULL);
            command_queue.finish();
            
            //L2-norm 2
            command_queue.enqueueNDRangeKernel(kernels[4], NULL, global_item_size4,local_item_size, NULL, NULL);
            command_queue.finish();
            
            //power iteration 3
            command_queue.enqueueNDRangeKernel(kernels[2], NULL, global_item_size2,local_item_size, NULL, NULL);
            command_queue.finish();
            
           /* command_queue.enqueueReadBuffer(L2norm_buffer, CL_TRUE, 0, sizeof(cl_float)*g_ss, L2);
            cout << trial<<" ";
            ofs << trial << " ";
            for (int j=0; j<g_ss; j++) {
                cout<<L2[j] << " ";
                ofs<<L2[j] << " ";
            }
            cout << endl;
            ofs << endl;*/
        }
        
        /*command_queue.enqueueReadImage(reconst_img, CL_TRUE, origin, region, 0, 0, image);
        write_data_output("E:", "evectorImg.raw", image, IMAGE_SIZE_M);
        ofs.close();*/

    }catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(ret.err());
        
    }
    
    return 0;
}

int FISTA_ocl(OCL_platform_device plat_dev_list, float *ang, int ss){
    
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
        FISTA_programBuild(plat_dev_list.context(i), &kernels_plat);
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
    vector<cl::Buffer> L2norm_buffers;
    for (int i=0; i<plat_dev_list.contextsize(); i++){
        
        angle_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*g_pa, 0, NULL));
        plat_dev_list.queue(i,0).enqueueWriteBuffer(angle_buffers[i], CL_TRUE, 0, sizeof(cl_float)*g_pa, ang, NULL, NULL);
        
        
        //tAAの最大固有値の探索
        //input_thを間借り
        L2norm_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float)*g_ss, 0, NULL));
        input_th.push_back(thread(powerIteration,
                                  plat_dev_list.queue(i,0),kernels[i],
                                  angle_buffers[i],L2norm_buffers[i]));
    }
    
    
    //start thread
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        //input_th.push_back(thread(dummy));
        reconst_th.push_back(thread(dummy));
        output_th.push_back(thread(dummy));
    }
    for (int N = g_st; N < g_st + g_num;) {
        for (int j=0; j<plat_dev_list.contextsize(); j++) {
            if (input_th[j].joinable()) {
                input_th[j].join();
                input_th[j] = thread(FISTA_input_thread,
                                     plat_dev_list.queue(j,0),kernels[j],
                                     angle_buffers[j],L2norm_buffers[j],
                                     sub,N,min(N+dN[j]-1,g_st-1+g_num),g_it,j);
                
                N+=dN[j];
                if (N >= g_st + g_num) break;
                
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
