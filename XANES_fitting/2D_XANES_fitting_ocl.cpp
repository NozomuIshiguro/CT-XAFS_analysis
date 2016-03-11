//
//  2D_XANES_fitting_ocl.cpp
//  XANES_fitting
//
//  Created by Nozomu Ishiguro on 2015/01/07.
//  Copyright (c) 2014-2015 Nozomu Ishiguro. All rights reserved.
//

#include "XANES_fitting.hpp"
#include "XANES_fit_cl.hpp"


string kernel_preprocessor_nums(float E0, int num_energy, size_t param_size){
    ostringstream OSS;
    
    OSS<<"#define E0 "<<E0<<endl<<endl;
    
    OSS<<"#define ENERGY_NUM "<< num_energy <<endl<<endl;
    
    OSS<<"#define PARA_NUM "<< param_size <<endl<<endl;
    
    OSS<<"#define PARA_NUM_SQ "<< param_size*param_size <<endl<<endl;
    
    return OSS.str();
}

int fitresult_output_thread(fitting_eq fit_eq, int AngleNo,string output_dir,
                            vector<float*> result_outputs){
    
    ostringstream oss;
    int t=0;
	for (int i=0; i<fit_eq.ParaSize(); i++) {
		char buffer;
        buffer=fit_eq.freefix_para()[i];
        if (atoi(&buffer)==1) {
			string fileName_output= output_dir+ "/"+fit_eq.param_name(i)+ AnumTagString(AngleNo,"/i", ".raw");
			oss << "output file: " << fileName_output << "\n";
            outputRawFile_stream(fileName_output,result_outputs[t],IMAGE_SIZE_M);
            t++;
        }
    }
    oss << endl;
    cout << oss.str();
    for (int i=0; i<fit_eq.freeParaSize(); i++) {
        delete [] result_outputs[i];
    }
    return 0;
}

int XANES_fit_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernels,
					fitting_eq fiteq,int AngleNo, int thread_id,
                    input_parameter inp,cl::Buffer energy_buff, vector<float*> mt_vec,int64_t offset)
{
    try {
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        string devicename = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>();
        size_t maxWorkGroupSize = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        
        int fittingStartEnergyNo = inp.getFittingStartEnergyNo();
        int fittingEndEnergyNo = inp.getFittingEndEnergyNo();
        int startEnergyNo=inp.getStartEnergyNo();
        int num_energy = fittingEndEnergyNo - fittingStartEnergyNo + 1;
        int paramsize = (int)fiteq.freeParaSize();
        string output_dir=inp.getFittingOutputDir();
        
        const cl::NDRange local_item_size(maxWorkGroupSize,1,1);
        const cl::NDRange global_item_size(IMAGE_SIZE_X,IMAGE_SIZE_Y,1);
		cout << "Processing angle No. " << AngleNo << "..." << endl << endl;
		//cout << "     global worksize :" << IMAGE_SIZE_X << ", " << IMAGE_SIZE_Y << ", 1" << endl;
		//cout << "     local worksize :" << maxWorkGroupSize << ", 1, 1" << endl << endl;
        
        
        cl::Buffer para_lowerLimit_buff(context, CL_MEM_READ_ONLY, sizeof(cl_float)*paramsize, 0, NULL);
        cl::Buffer para_upperLimit_buff(context, CL_MEM_READ_ONLY, sizeof(cl_float)*paramsize, 0, NULL);
        cl::Buffer lambda_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        cl::Buffer rho_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        cl::Buffer nyu_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        cl::Buffer mask_buff(context, CL_MEM_READ_WRITE, sizeof(cl_char)*IMAGE_SIZE_M, 0, NULL);
        cl::Buffer results_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*paramsize, 0, NULL);
        cl::Buffer results_cnd_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*paramsize, 0, NULL);
        cl::ImageFormat format(CL_R, CL_FLOAT);
		cl::Image2DArray mt_img(context, CL_MEM_READ_ONLY, format, num_energy,IMAGE_SIZE_X, IMAGE_SIZE_Y, 0, 0, NULL, NULL);

        
        command_queue.enqueueWriteBuffer(para_lowerLimit_buff, CL_TRUE, 0, sizeof(cl_float)*paramsize, fiteq.freepara_lowerlimit,NULL,NULL);
        command_queue.enqueueWriteBuffer(para_upperLimit_buff, CL_TRUE, 0, sizeof(cl_float)*paramsize, fiteq.freepara_upperlimit,NULL,NULL);
        command_queue.enqueueFillBuffer(lambda_buff, (cl_float)inp.getLambda_t_fit(), 0, sizeof(cl_float)*IMAGE_SIZE_M,NULL,NULL);
        command_queue.enqueueFillBuffer(nyu_buff, (cl_float)2.0f, 0, sizeof(cl_float)*IMAGE_SIZE_M,NULL,NULL);
        for (int i=0; i<paramsize; i++) {
            command_queue.enqueueFillBuffer(results_img, (cl_float)fiteq.freefit_para()[i], sizeof(cl_float)*IMAGE_SIZE_M*i, sizeof(cl_float)*IMAGE_SIZE_M,NULL,NULL);
        }
        
        
        vector<float*> results_vec;
        for (int i=0; i<paramsize; i++) {
            results_vec.push_back(new float[IMAGE_SIZE_M]);
        }
        
        
		cl::size_t<3> origin;
		cl::size_t<3> region;
		origin[0] = 0;
		origin[1] = 0;
		region[0] = IMAGE_SIZE_X;
		region[1] = IMAGE_SIZE_Y;
		region[2] = 1;


		//write buffer
		for (int i = fittingStartEnergyNo; i<=fittingEndEnergyNo; i++) {
			origin[2] = i-fittingStartEnergyNo;
            command_queue.enqueueWriteImage(mt_img, CL_TRUE, origin, region, IMAGE_SIZE_X*sizeof(float), 0, mt_vec[i-startEnergyNo]+offset, NULL, NULL);
		}
        
        
        //XANES fitting
        //set kernel (XANES_fitting)
        kernels[0].setArg(0, mt_img);
        kernels[0].setArg(1, results_img);
        kernels[0].setArg(2, results_cnd_img);
        kernels[0].setArg(3, energy_buff);
        kernels[0].setArg(4, lambda_buff);
        kernels[0].setArg(5, rho_buff);
        kernels[0].setArg(6, para_lowerLimit_buff);
        kernels[0].setArg(7, para_upperLimit_buff);
        //set kernel (update results)
        kernels[1].setArg(0, results_img);
        kernels[1].setArg(1, results_cnd_img);
        kernels[1].setArg(2, lambda_buff);
        kernels[1].setArg(3, rho_buff);
        kernels[1].setArg(4, nyu_buff);
        for (int trial=0; trial<inp.getNumTrial_fit(); trial++) {
            command_queue.enqueueNDRangeKernel(kernels[0], NULL, global_item_size, local_item_size, NULL, NULL);
            command_queue.finish();
            
            command_queue.enqueueNDRangeKernel(kernels[1], NULL, global_item_size, local_item_size, NULL, NULL);
            command_queue.finish();
        }
        
        
        //set mask
        kernels[2].setArg(0, mt_img);
        kernels[2].setArg(1, mask_buff);
        command_queue.enqueueNDRangeKernel(kernels[2], NULL, global_item_size, local_item_size, NULL, NULL);
        command_queue.finish();
		
        
        //apply threshold
        kernels[3].setArg(0, results_img);
        kernels[3].setArg(1, mask_buff);
        kernels[3].setArg(2, para_lowerLimit_buff);
        kernels[3].setArg(3, para_upperLimit_buff);
        command_queue.enqueueNDRangeKernel(kernels[3], NULL, global_item_size, local_item_size, NULL, NULL);
        command_queue.finish();
		
        
        //read results image from buffer
		for (int i = 0; i<paramsize; i++) {
            command_queue.enqueueReadBuffer(results_img, CL_TRUE, sizeof(cl_float)*IMAGE_SIZE_M*i, sizeof(cl_float)*IMAGE_SIZE_M, results_vec[i]);
        }
        
        
        //delete mt data
#ifndef XANES_FIT
		for (int i = 0; i<num_energy; i++) {
			delete[] mt_vec[i];
		}
#endif
        
        //output thread
        output_th_fit[thread_id].join();
        output_th_fit[thread_id]=thread(fitresult_output_thread, fiteq, AngleNo,
                                     output_dir,move(results_vec));

        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
    
    return 0;
}


int data_input_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernels,
						fitting_eq fiteq, int AngleNo, int thread_id,
						input_parameter inp, cl::Buffer energy_buff){
    
    
    int startEnergyNo = inp.getFittingStartEnergyNo();
    int endEnergyNo = inp.getFittingEndEnergyNo();
    string fileName_base = inp.getFittingFileBase();
    string input_dir=inp.getInputDir();
    vector<float*> mt_vec;
	vector<string> filepath_input;
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
		mt_vec.push_back(new float[IMAGE_SIZE_M]);
		filepath_input.push_back(input_dir + EnumTagString(i,"/",fileName_base) + AnumTagString(AngleNo,"",".raw"));
    }
    

    //input mt data
	//m1.lock();
    int num_energy = endEnergyNo -startEnergyNo +1;
    for (int i=0; i<num_energy; i++) {
        readRawFile(filepath_input[i],mt_vec[i]);
    }
	//m1.unlock();
    
    fitting_th[thread_id].join();
    fitting_th[thread_id] = thread(XANES_fit_thread,
                                   command_queue,kernels,
								   fiteq, AngleNo, thread_id,
                                   inp,energy_buff,move(mt_vec),0);
    
    
    return 0;
}

//dummy thread
static int dummy(){
    return 0;
}


int XANES_fit_ocl(fitting_eq fiteq, input_parameter inp,
                  OCL_platform_device plat_dev_list)
{
    cl_int ret;
    float startEnergy=inp.getStartEnergy();
    float endEnergy=inp.getEndEnergy();
    float E0=inp.getE0();
    int startAngleNo=inp.getStartAngleNo();
    int endAngleNo=inp.getEndAngleNo();
    
    
    //create output dir
    for (int i=0; i<fiteq.ParaSize(); i++) {
        char buffer=fiteq.freefix_para()[i];
        if (atoi(&buffer)==1) {
            string fileName_output = inp.getOutputDir() + "/"+fiteq.param_name(i);
            MKDIR(fileName_output.c_str());
        }
    }
    
    
    //energy file input & processing
    ifstream energy_ifs(inp.getEnergyFilePath(),ios::in);
    vector<float> energy;
    int i=0, startEnergyNo=0, endEnergyNo=0;
    do {
        float a;
        energy_ifs>>a;
		if (energy_ifs.eof()) break;
        cout<<i<<": "<<a;
        if ((a>=startEnergy)&(a<=endEnergy)) {
            energy.push_back(a-E0);
            cout<<" <- fitting range";
            endEnergyNo = i;
        } else if(a<startEnergy) {
            startEnergyNo = i+1;
        }
        cout<<endl;
        i++;
	} while (!energy_ifs.eof());
    int num_energy=endEnergyNo-startEnergyNo+1;
    inp.setFittingStartEnergyNo(startEnergyNo);
    inp.setFittingEndEnergyNo(endEnergyNo);
    ostringstream oss;
    oss << startEnergyNo<<"-"<<endEnergyNo;
    inp.setEnergyNoRange(oss.str());
    oss.flush();
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
    vector<vector<cl::Kernel>> kernels;
    string kernel_code="";
    kernel_code += kernel_preprocessor_nums(E0,num_energy,fiteq.freeParaSize());
    kernel_code += fiteq.preprocessor_str();
    kernel_code += kernel_fit_src;
    //cout << kernel_code<<endl;
    size_t kernel_code_size = kernel_code.length();
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        
        cl::Program::Sources source(1,std::make_pair(kernel_code.c_str(),kernel_code_size));
        cl::Program program(plat_dev_list.context(i), source,&ret);
        //kernel build
        string option = "-cl-nv-maxrregcount=64";
        //option += " -cl-nv-verbose -Werror";
        ret = program.build(option.c_str());
        //string logstr=program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0]);
        //cout << logstr << endl;
        //cout<<ret<<endl;
        vector<cl::Kernel> kernels_plat;
        kernels_plat.push_back(cl::Kernel::Kernel(program,"XANES_fitting", &ret));//0
        kernels_plat.push_back(cl::Kernel::Kernel(program,"updateResults", &ret));//1
        kernels_plat.push_back(cl::Kernel::Kernel(program,"setMask", &ret));//2
        kernels_plat.push_back(cl::Kernel::Kernel(program,"applyThreshold", &ret));//3
        
        kernels.push_back(kernels_plat);
    }
	//cout << kernels[0][0].getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(plat_dev_list.dev(0, 0)) << endl;

    
    //display OCL device
    vector<int> dA;
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
			cout << "CL DEVICE NAME: " << device_pram << endl << endl;
            t++;
        }
    }

    
    //energy_buffers create (per context)
    vector<cl::Buffer> energy_buffers;
    for (int i=0; i<plat_dev_list.contextsize(); i++){
        energy_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*num_energy, 0, NULL));
		plat_dev_list.queue(i,0).enqueueWriteBuffer(energy_buffers[i], CL_TRUE, 0, sizeof(cl_float)*num_energy, &energy[0], NULL, NULL);
    }
   

    //start thread
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        input_th.push_back(thread(dummy));
        fitting_th.push_back(thread(dummy));
        output_th_fit.push_back(thread(dummy));
    }
    for (int i=startAngleNo; i<=endAngleNo;) {
        for (int j=0; j<plat_dev_list.contextsize(); j++) {
            if (input_th[j].joinable()) {
                input_th[j].join();
                input_th[j] = thread(data_input_thread,
                                     plat_dev_list.queue(j, 0),kernels[j],
                                     fiteq,i,j,inp,energy_buffers[j]);
                i++;
                if (i > endAngleNo) break;
            } else{
                this_thread::sleep_for(chrono::milliseconds(100));
            }
        }
        if (i > endAngleNo) break;
    }
    
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        input_th[j].join();
    }
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        fitting_th[j].join();
    }
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        output_th_fit[j].join();
    }
    

    return 0;
}