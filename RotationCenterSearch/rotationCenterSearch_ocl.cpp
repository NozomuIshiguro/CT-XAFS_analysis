//
//  rotationCenterSearch_ocl.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2016/02/16.
//  Copyright © 2016年 Nozomu Ishiguro. All rights reserved.
//

#include "CTXAFS.hpp"
#include "reslice.hpp"
#include "CT_reconstruction.hpp"
#include "rotCenterSerch_cl.hpp"

vector<thread> input_th, output_th, imageReg_th, reconst_th, reslice_th, output_th1, output_th2;;

extern int g_ss;

int his_data_input(OCL_platform_device plat_dev_list,
                   vector<cl::Kernel> kernels, string fileName_base, input_parameter inp,
                   vector<float*> mt_target_img){
    cl::Context context = plat_dev_list.context(0);
    cl::CommandQueue queue = plat_dev_list.queue(0, 0);
    cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
    int imageSizeX = inp.getImageSizeX();
    int imageSizeY = inp.getImageSizeY();
    int imageSizeM = inp.getImageSizeM();
    int maxWorkSize = (int)min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), imageSizeX);
    
    int targetEnergyNo=inp.getTargetEnergyNo();
    int startAngleNo = inp.getStartAngleNo();
    int EndAngleNo = inp.getEndAngleNo();
    const int dA=EndAngleNo-startAngleNo+1;
    const int di=inp.getNumParallel();
    int scanN=inp.getScanN();
    
    // dark data input
    cout << "Reading dark file...";
    unsigned short *dark_img;
    dark_img = new unsigned short[(imageSizeM+32)*scanN];
    string fileName_dark = fileName_base+ "dark.his";
    readHisFile_stream(fileName_dark,1,scanN,dark_img,imageSizeM);
	cout << "done." << endl << endl;
    

    // I0 target data input
	cout << "Reading I0 file...";
    unsigned short *I0_img_target;
    I0_img_target = new unsigned short[(imageSizeM+32)*scanN];
    string fileName_I0_target;
    fileName_I0_target =  EnumTagString(targetEnergyNo,fileName_base,"_I0.his");
    readHisFile_stream(fileName_I0_target,1,scanN,I0_img_target,imageSizeM);
	cout << "done." << endl << endl;
    
   
    //target It data input
	cout << "Reading It file...";
    unsigned short *It_img_target;
    It_img_target = new unsigned short[(imageSizeM+32)*(int64_t)dA];
    string fileName_It_target = EnumTagString(targetEnergyNo,fileName_base,".his");
    readHisFile_stream(fileName_It_target,startAngleNo,EndAngleNo,It_img_target,imageSizeM);
	cout << "done." << endl << endl;
    
    cl::Buffer dark_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
    cl::Buffer I0_target_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM, 0, NULL);
    cl::Buffer mt_target_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imageSizeM*di, 0, NULL);
    cl::ImageFormat format(CL_RG,CL_FLOAT);
    cl::Image2DArray mt_target_image(context, CL_MEM_READ_WRITE,format,di,imageSizeX,imageSizeY,0,0,NULL,NULL);
    cl::Image2DArray mt_target_outputImg(context, CL_MEM_READ_WRITE,format,di,imageSizeX,imageSizeY,0,0,NULL,NULL);
    
    
    //transfer and merge rawhis data to dark_buffer
	cout << "Merging dark...";
    mergeRawhisBuffers(queue, kernels[0], cl::NDRange(imageSizeX,imageSizeY,1),cl::NDRange(maxWorkSize,1,1), dark_img, dark_buffer,scanN,imageSizeM);
	cout << "done." << endl << endl;
    
    //transfer and merge rawhis data to I0_target_buffer
	cout << "Merging I0...";
    mergeRawhisBuffers(queue, kernels[0],cl::NDRange(imageSizeX,imageSizeY,1),cl::NDRange(maxWorkSize,1,1), I0_img_target, I0_target_buffer,scanN,imageSizeM);
	cout << "done." << endl << endl;

    //transfer and merge rawhis data to It_target_buffer
	cout << "Convering to mt...";
	mask msk(inp);
    for (int i=0; i<dA; i+=di) {
		int step = min(di,dA-i);
        mt_conversion(queue,kernels[1],dark_buffer,I0_target_buffer,
                      mt_target_buffer,mt_target_image,mt_target_outputImg,
                      cl::NDRange(imageSizeX,imageSizeY,step),cl::NDRange(maxWorkSize,1,1),
                      cl::NDRange(0,0,0),It_img_target+(imageSizeM+32)*(int64_t)i,
                      step,msk,false,imageSizeM);
		for (int j = 0; j < step; j++) {
			queue.enqueueReadBuffer(mt_target_buffer, CL_FALSE, sizeof(cl_float)*imageSizeM*j, sizeof(cl_float)*imageSizeM, mt_target_img[i + j], NULL, NULL);
		}
        queue.finish();
    }
	cout << "done." << endl << endl;
    

    delete[] dark_img;
    delete[] I0_img_target;
    delete[] It_img_target;
    return 0;
}


int rotCenterShift(OCL_platform_device plat_dev_list,cl::Kernel kernel,
                   vector<float*> prj_img_vec,vector<float*> rotCntShft_img_vec,
                   input_parameter inp){
    
    cl::Context context = plat_dev_list.context(0);
    cl::CommandQueue queue = plat_dev_list.queue(0, 0);
    string devicename = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>();
    size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    int imageSizeX = inp.getImageSizeX();
    int num_angle=inp.getEndAngleNo()-inp.getStartAngleNo()+1;
    const cl::NDRange local_item_size(maxWorkGroupSize,1,1);
    const cl::NDRange global_item_size(imageSizeX,num_angle,1);
    
    cl::ImageFormat format(CL_R, CL_FLOAT);
    cl::Image2D prj_input_img(context,CL_MEM_READ_WRITE,format,imageSizeX,num_angle,0,NULL, NULL);
    cl::Image2D prj_output_img(context,CL_MEM_WRITE_ONLY,format,imageSizeX,num_angle,0, NULL, NULL);
    
    cl::size_t<3> origin;
    cl::size_t<3> region;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;
	region[0] = imageSizeX;
	region[1] = num_angle;
	region[2] = 1;
    
    queue.enqueueWriteImage(prj_input_img, CL_TRUE, origin, region, imageSizeX*sizeof(float), 0, prj_img_vec[inp.getLayerN()-1],NULL,NULL);
	kernel.setArg(0, prj_input_img);
	kernel.setArg(1, prj_output_img);
    for (int i=0; i<inp.getRotCenterShiftN(); i++) {
        float rotCenterShift = inp.getRotCenterShiftStart()+inp.getRotCenterShiftStep()*i;
		//cout << rotCenterShift << endl;
        kernel.setArg(2, (cl_float)rotCenterShift);
        queue.enqueueNDRangeKernel(kernel, NULL, global_item_size, local_item_size, NULL, NULL);
        queue.finish();
        
        queue.enqueueReadImage(prj_output_img, CL_TRUE, origin, region, imageSizeX*sizeof(float), 0, rotCntShft_img_vec[i] ,NULL,NULL);
        queue.finish();
    }
    
    return 0;
}

int rotCntSrc_programBuild(cl::Context context,vector<cl::Kernel> *kernels){
    cl_int ret;
    
    //OpenCL Program
    //kernel program source
    /*ifstream ifs("./rotCenterSearch.cl", ios::in);
     if(!ifs) {
     cerr << "   Failed to load kernel" << endl;
     return -1;
     }
     istreambuf_iterator<char> it(ifs);
     istreambuf_iterator<char> last;
     string kernel_src(it,last);
     ifs.close();*/
    //cout << kernel_code<<endl;
#if defined (OCL120)
    cl::Program::Sources source(1,std::make_pair(kernel_src_rcs.c_str(),kernel_src_rcs.length()));
#else
    cl::Program::Sources source(1,kernel_src_rcs);
#endif
    cl::Program program(context, source,&ret);
    //kernel build
    ret=program.build();
    //cout<<ret<<endl;
    kernels[0].push_back(cl::Kernel::Kernel(program,"rotCenterShift", &ret));//0
    kernels[0].push_back(cl::Kernel::Kernel(program,"imgAVG", &ret));//1
    kernels[0].push_back(cl::Kernel::Kernel(program,"imgSTDEV", &ret));//2
    kernels[0].push_back(cl::Kernel::Kernel(program,"setMask", &ret));//3
    kernels[0].push_back(cl::Kernel::Kernel(program,"imgFocusIndex", &ret));//4
    
    return 0;
}

int imgFocusIndex_estimation(cl::Context context,cl::CommandQueue queue,
                        cl::Kernel kernel_Findex,cl::Kernel kernel_mask,
                        cl::Image2DArray reconst_img, float* Findex,
                        int dN, int offsetN, input_parameter inp,float ang_ini,float ang_fin){
    
    int imageSizeX = inp.getImageSizeX();
    
    cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
    string devicename = device.getInfo<CL_DEVICE_NAME>();
    size_t WorkGroupSize = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), imageSizeX);
    
    cl::ImageFormat format(CL_R, CL_FLOAT);
    cl::Buffer imgFindex_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dN, 0, NULL);
    cl::Image2DArray mask_img(context, CL_MEM_READ_WRITE, format, dN, g_nx, g_nx, 0, 0, NULL, NULL);
    
    
    
    //kernel dimension declaration
    const cl::NDRange global_item_size(WorkGroupSize*dN,1,1);
    const cl::NDRange local_item_size(WorkGroupSize,1,1);
    
    
    //kernel set arg
    kernel_mask.setArg(0, mask_img);
    kernel_mask.setArg(1, (cl_int)offsetN);
    kernel_mask.setArg(2, (cl_float)inp.getRotCenterShiftStart());
    kernel_mask.setArg(3, (cl_float)inp.getRotCenterShiftStep());
    kernel_mask.setArg(4, (cl_float)ang_ini);
    kernel_mask.setArg(5, (cl_float)ang_fin);
    queue.enqueueNDRangeKernel(kernel_mask, NULL, global_item_size, local_item_size, NULL, NULL);
    
    kernel_Findex.setArg(0, reconst_img);
    kernel_Findex.setArg(1, mask_img);
    kernel_Findex.setArg(2, cl::Local(sizeof(cl_float2)*WorkGroupSize));
    kernel_Findex.setArg(3, imgFindex_buff);
    queue.enqueueNDRangeKernel(kernel_Findex, NULL, global_item_size, local_item_size, NULL, NULL);
    
    
    //read stdev data from GPU
    queue.enqueueReadBuffer(imgFindex_buff, CL_TRUE, 0, sizeof(cl_float)*dN, &Findex[offsetN], NULL, NULL);
    queue.finish();
    
    return 0;
}


int imgSTDEV_estimation(cl::Context context,cl::CommandQueue queue,
                        cl::Kernel kernel_avg,cl::Kernel kernel_stdev,cl::Kernel kernel_mask,
                        cl::Image2DArray reconst_img, float* stdev,
                        int dN, int offsetN, input_parameter inp,float ang_ini,float ang_fin){
    
    int imageSizeX = inp.getImageSizeX();
    
    cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
    string devicename = device.getInfo<CL_DEVICE_NAME>();
    size_t WorkGroupSize = min((int)device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(), imageSizeX);
    
    cl::ImageFormat format(CL_R, CL_FLOAT);
    cl::Buffer imgAVG_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dN, 0, NULL);
    cl::Buffer imgSTDEV_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*dN, 0, NULL);
    cl::Image2DArray mask_img(context, CL_MEM_READ_WRITE, format, dN, g_nx, g_nx, 0, 0, NULL, NULL);
    
    
    
    //kernel dimension declaration
    const cl::NDRange global_item_size(WorkGroupSize*dN,1,1);
    const cl::NDRange local_item_size(WorkGroupSize,1,1);
    
    
    //kernet set arg
    kernel_mask.setArg(0, mask_img);
    kernel_mask.setArg(1, (cl_int)offsetN);
    kernel_mask.setArg(2, (cl_float)inp.getRotCenterShiftStart());
    kernel_mask.setArg(3, (cl_float)inp.getRotCenterShiftStep());
    kernel_mask.setArg(4, (cl_float)ang_ini);
    kernel_mask.setArg(5, (cl_float)ang_fin);
    queue.enqueueNDRangeKernel(kernel_mask, NULL, global_item_size, local_item_size, NULL, NULL);
    
    kernel_avg.setArg(0, reconst_img);
    kernel_avg.setArg(1, mask_img);
    kernel_avg.setArg(2, cl::Local(sizeof(cl_float2)*WorkGroupSize));
    kernel_avg.setArg(3, imgAVG_buff);
	queue.enqueueNDRangeKernel(kernel_avg, NULL, global_item_size, local_item_size, NULL, NULL);
    
    kernel_stdev.setArg(0, reconst_img);
    kernel_stdev.setArg(1, mask_img);
    kernel_stdev.setArg(2, cl::Local(sizeof(cl_float2)*WorkGroupSize));
    kernel_stdev.setArg(3, imgAVG_buff);
    kernel_stdev.setArg(4, imgSTDEV_buff);
	queue.enqueueNDRangeKernel(kernel_stdev, NULL, global_item_size, local_item_size, NULL, NULL);
    
    
    //read stdev data from GPU
    queue.enqueueReadBuffer(imgSTDEV_buff, CL_TRUE, 0, sizeof(cl_float)*dN, &stdev[offsetN], NULL, NULL);
    queue.finish();
    
    return 0;
}

int reconstRotCntShift_thread(cl::CommandQueue queue, vector<cl::Kernel> kernels,
	cl::Buffer angle_buffer, cl::Image2DArray reconst_img,cl::Image2DArray prj_img, 
	vector<float*> reconst_img_vec, vector<float*> prj_img_vec,
	int *sub, float* Findex, int offsetN, int di,input_parameter inp,float ang_ini,float ang_fin) {

	cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

	//origin and region of image objects
	cl::size_t<3> origin;
	cl::size_t<3> region_img;
	cl::size_t<3> region_prj;
	origin[0] = (g_nx - g_ox) / 2;
	origin[1] = (g_ny - g_oy) / 2;
	region_img[0] = g_ox;
	region_img[1] = g_oy;
	region_img[2] = 1;
	region_prj[0] = g_px;
	region_prj[1] = g_pa;
	region_prj[2] = 1;

	//write memory objects to GPUs
	cl_float4 ini={(cl_float)g_ini,0.0f,0.0f,0.0f};
	for (int i = 0; i<di; i++) {
		origin[2] = i;
		queue.enqueueFillImage(reconst_img,ini, origin, region_img, NULL, NULL);
		//queue.enqueueWriteImage(reconst_img, CL_TRUE, origin, region_img, sizeof(cl_float)*g_nx, 0, reconst_img_vec[i+offsetN], NULL, NULL);
		queue.enqueueWriteImage(prj_img, CL_TRUE, origin, region_prj, sizeof(cl_float)*g_px, 0, prj_img_vec[i+offsetN], NULL, NULL);
	}
	//OSEM_execution(queue, kernels, angle_buffer, sub, reconst_img, prj_img, di, offsetN, 1);
	OSEM_execution(queue, kernels, angle_buffer, sub, reconst_img, prj_img, di, 1, true);
	//imgSTDEV_estimation(context, queue, kernels[5], kernels[6], kernels[7], reconst_img, stdev, di, offsetN,inp,ang_ini,ang_fin);
    imgFocusIndex_estimation(context, queue, kernels[9], kernels[8], reconst_img, Findex, di, offsetN,inp,ang_ini,ang_fin);

	//read memory objects from GPUs
	for (int i = 0; i<di; i++) {
		origin[2] = i;
		queue.enqueueReadImage(reconst_img, CL_TRUE, origin, region_img, sizeof(cl_float)*g_nx, 0, reconst_img_vec[i + offsetN], NULL, NULL);
	}


	return 0;
}

//dummy thread
static int dummy() {
	return 0;
}

int reconstRotCntShift(OCL_platform_device plat_dev_list,vector<vector<cl::Kernel>> kernels,
                       vector<float*> reconst_img_vec,vector<float*> prj_img_vec,
                       input_parameter inp, int ss, float *ang, float* stdev){
	
    const int di=inp.getNumParallel();
    const int dN=inp.getRotCenterShiftN();
    
    
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
    
    float ang_ini = ang[inp.getStartAngleNo()-1];
    float ang_fin = ang[inp.getEndAngleNo()-1];
    
    
    //buffer・image objects作成
	cl::ImageFormat format(CL_R, CL_FLOAT);
	vector<cl::Context> contexts;
	vector<cl::CommandQueue> queues;
	vector<cl::Buffer> angle_buffers;
	vector<cl::Image2DArray> reconst_imgs;
	vector<cl::Image2DArray> prj_imgs;
	for (int i = 0; i < plat_dev_list.contextsize(); i++) {
		contexts.push_back(plat_dev_list.context(i));
		queues.push_back(plat_dev_list.queue(i, 0));

		angle_buffers.push_back(cl::Buffer(contexts[i], CL_MEM_READ_ONLY, sizeof(cl_float)*g_pa, 0, NULL));
		queues[i].enqueueWriteBuffer(angle_buffers[i], CL_TRUE, 0, sizeof(cl_float)*g_pa, ang, NULL, NULL);
		
		reconst_imgs.push_back(cl::Image2DArray(contexts[i], CL_MEM_READ_WRITE, format, di, g_nx, g_nx, 0, 0, NULL, NULL));
		prj_imgs.push_back(cl::Image2DArray(contexts[i], CL_MEM_READ_WRITE, format, di, g_px, g_pa, 0, 0, NULL, NULL));
	}
    
    
    //reconstruction
    /*for (int i=0; i<dN; i+=di) {
		reconstRotCntShift_thread(queues[0], kernels[0], angle_buffers[0], reconst_imgs[0], prj_imgs[0], reconst_img_vec, prj_img_vec, sub, stdev, i, di);

		for (int j = i; j < i + di; j++) {
			cout << inp.getRotCenterShiftStart() + inp.getRotCenterShiftStep()*j << " " << stdev[j] << endl;
		}
    }*/
	//start thread
	for (int i = 0; i<plat_dev_list.contextsize(); i++) {
		reconst_th.push_back(thread(dummy));
	}
	for (int i = 0; i < dN;) {
		for (int j = 0; j<plat_dev_list.contextsize(); j++) {
			if (reconst_th[j].joinable()) {
				int step = min(di,dN-i);
				reconst_th[j].join();
				reconst_th[j] = thread(reconstRotCntShift_thread, queues[j], kernels[j], angle_buffers[j], reconst_imgs[j], prj_imgs[j], reconst_img_vec, prj_img_vec, sub, stdev, i, step, inp, ang_ini, ang_fin);
				i += di;
				if (i >= dN) break;
			}
			else {
				this_thread::sleep_for(chrono::milliseconds(10));
			}

			if (i >= dN) break;
		}
	}
	for (int i = 0; i<plat_dev_list.contextsize(); i++) {
		reconst_th[i].join();
	}
	/*for (int i = 0; i < dN; i++) {
		cout << i+1 <<": "<< inp.getRotCenterShiftStart() + inp.getRotCenterShiftStep()*i << " " << stdev[i] << endl;
	}*/
    
    
    return 0;
}

int rotationCenterSearch(string fileName_base, input_parameter inp, float *ang){
    
    OCL_platform_device plat_dev_list(g_devList,true);
    
    int num_angle=inp.getEndAngleNo()-inp.getStartAngleNo()+1;
    int imageSizeX = inp.getImageSizeX();
    int imageSizeY = inp.getImageSizeY();
    int imageSizeM = inp.getImageSizeM();
    
    //program build (mt conversion, reslice)
    cl_int ret;
    regMode regmode(0);
    cl::Program program=regmode.buildImageRegProgram(plat_dev_list.context(0),imageSizeX,imageSizeY);
    vector<cl::Kernel> kernels;
    kernels.push_back(cl::Kernel(program,"merge_rawhisdata", &ret));//0
    kernels.push_back(cl::Kernel(program,"mt_conversion", &ret));//1
    reslice_programBuild(plat_dev_list.context(0),&kernels,inp.getStartAngleNo(),inp.getEndAngleNo(),inp);//reslice:2,xprojection:3,zcorrection:4
    
    
    //program build (reconst)
	vector<vector<cl::Kernel>> kernels_reconst;
	for (int i = 0; i < plat_dev_list.contextsize(); i++) {
		vector<cl::Kernel> kernels_plat;
		OSEM_programBuild(plat_dev_list.context(i), &kernels_plat);
        //OSEM1:0,OSEM2:1,OSEM3:2,partialDerivativeOfGradiant:3,FISTA2:4
		rotCntSrc_programBuild(plat_dev_list.context(i), &kernels_plat);
        //rotCenterShift:5,imgAVG:6,imgSTDEV:7,setMask:8,imgFocusIndex:9

		kernels_reconst.push_back(kernels_plat);
	}
   
    
    
    //input his file
    vector<float*> mt_img_vec;
    for (int i=0; i<num_angle; i++) {
        mt_img_vec.push_back(new float[imageSizeM]);
    }
    his_data_input(plat_dev_list,kernels,fileName_base,inp,mt_img_vec);
	/*for (int i = 0; i < num_angle; i++) {
		string fileName_output = inp.getOutputDir() + "/" + AnumTagString(i + 1, inp.getOutputFileBase(), ".raw");
		outputRawFile_stream(fileName_output, mt_img_vec[i], IMAGE_SIZE_M);
	}*/
    
    //reslice
	cout << "Reslicing...";
    vector<float*> prj_img_vec;
    for (int i=0; i<imageSizeY; i++) {
        prj_img_vec.push_back(new float[imageSizeX*num_angle]);
    }
    reslice_mtImg(plat_dev_list,kernels[2],mt_img_vec,prj_img_vec,inp);
    //delete input mt_vec
    for (int i=0; i<num_angle; i++) {
        delete [] mt_img_vec[i];
    }
	/*for (int i = 0; i < IMAGE_SIZE_Y; i++) {
		string fileName_output = inp.getOutputDir() + "/" + AnumTagString(i + 1, inp.getOutputFileBase(), ".raw");
		outputRawFile_stream(fileName_output, prj_img_vec[i], (size_t)(IMAGE_SIZE_X*num_angle));
	}*/
	cout << "done." << endl << endl;

    
    //rotation center shift
	cout << "Shifting rotation center...";
    vector<float*> prjshift_img_vec;
    for (int i=0; i<inp.getRotCenterShiftN(); i++) {
        prjshift_img_vec.push_back(new float[imageSizeX*num_angle]);
    }
    rotCenterShift(plat_dev_list,kernels_reconst[0][5],prj_img_vec,prjshift_img_vec,inp);
    for (int i=0; i<imageSizeY; i++) {
        delete [] prj_img_vec[i];
    }
	/*for (int i = 0; i < inp.getRotCenterShiftN(); i++) {
		string fileName_output = inp.getOutputDir() + "/" + AnumTagString(i + 1, inp.getOutputFileBase(), ".raw");
		outputRawFile_stream(fileName_output, prjshift_img_vec[i], (size_t)(IMAGE_SIZE_X*num_angle));
	}*/
	cout << "done." << endl << endl;

    
    //reconstruction & stvev
	cout << "Reconstructing CT images...";
    vector<float*> reconst_img_vec;
    for (int i=0; i<inp.getRotCenterShiftN(); i++) {
        reconst_img_vec.push_back(new float[imageSizeM]);
    }
    float* Findex;
    Findex = new float[inp.getRotCenterShiftN()];
    reconstRotCntShift(plat_dev_list,kernels_reconst,reconst_img_vec,prjshift_img_vec,inp,g_ss,ang,Findex);
	cout << "done." << endl << endl;
	for (int i=0; i<inp.getRotCenterShiftN(); i++) {
        delete [] prjshift_img_vec[i];
    }
	
    

    //output
    cout<<"Output...";
	float min = FLT_MAX;
	int min_i = 0;
    for (int i=0; i<inp.getRotCenterShiftN(); i++) {
		string fileName_output = inp.getOutputDir() + "/" + AnumTagString(i + 1, inp.getOutputFileBase(), ".raw");
		outputRawFile_stream(fileName_output, reconst_img_vec[i], imageSizeM);

		if (min > Findex[i]) {
			min = Findex[i];
			min_i = i;
		}
    }
	for (int i = 0; i<inp.getRotCenterShiftN(); i++) {
		delete[] reconst_img_vec[i];
	}
	cout << "done." << endl << endl;

	cout << "Result" << endl;
	string fileName_result= inp.getOutputDir() + "/result.dat";
	ofstream ofs(fileName_result, ios::out | ios::trunc);
	ofs << "Rotation center shift\tFocus Index\tfilename"<<endl;
	for (int i = 0; i < inp.getRotCenterShiftN(); i++) {
		cout << inp.getRotCenterShiftStart() + inp.getRotCenterShiftStep()*i << " " << Findex[i] << "	";
		cout <<"("<< AnumTagString(i + 1, inp.getOutputFileBase(), ".raw)") << endl;
		ofs << inp.getRotCenterShiftStart() + inp.getRotCenterShiftStep()*i << "\t";
		ofs << Findex[i] << "\t";
		ofs << AnumTagString(i + 1, inp.getOutputFileBase(), ".raw") << endl;
	}
	ofs.close();
	cout << endl;

	float rotCntShift = inp.getRotCenterShiftStart() + min_i*inp.getRotCenterShiftStep();
	cout << "Estimated rotation center shift is " << rotCntShift;
	cout << "("+ AnumTagString(min_i + 1, inp.getOutputFileBase(), ".raw)") << endl << endl;
	
    return 0;
}

