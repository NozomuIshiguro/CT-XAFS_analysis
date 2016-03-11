//
//  2D_XANES_fitting_ocl.c
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/01/07.
//  Copyright (c) 2014-2015 Nozomu Ishiguro. All rights reserved.
//

#include "XANES_fitting.hpp"
#include "atan_lor_linear_fitting.hpp"
//#include "XANES_fit_cl.hpp"


string kernel_preprocessor_nums(float E0, int num_energy, size_t param_size){
    ostringstream OSS;
    
    OSS<<"#define E0 "<<E0<<endl<<endl;
    
    OSS<<"#define ENERGY_NUM "<< num_energy <<endl<<endl;
    
    OSS<<"#define PARA_NUM "<< param_size <<endl<<endl;
    
    return OSS.str();
}

int fitresult_output_thread(fitting_eq fit_eq,
                            int startAngleNo, int EndAngleNo,string output_dir,
                            vector<vector<float*>> result_outputs, int waittime){
    
    this_thread::sleep_for(chrono::seconds(waittime));
    ostringstream oss;
    for (int j=startAngleNo; j<=EndAngleNo; j++) {
        int t=0;
        for (int i=0; i<fit_eq.ParaSize(); i++) {
            char buffer;
            buffer=fit_eq.freefix_para()[i];
            if (atoi(&buffer)==1) {
                string fileName_output= output_dir+ "/"+fit_eq.param_name(i)+ AnumTagString(j,"/i", ".raw");
                oss << "output file: " << fileName_output << "\n";
                outputRawFile_stream(fileName_output,result_outputs[j-startAngleNo][t],IMAGE_SIZE_M);
                t++;
            }
        }
    }
    oss << endl;
    cout << oss.str();
    for (int j=startAngleNo; j<=EndAngleNo; j++) {
        for (int i=0; i<fit_eq.freeParaSize(); i++) {
            delete [] result_outputs[j-startAngleNo][i];
        }
    }
    return 0;
}

int XANES_fit_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                    fitting_eq fiteq,
                    int startAngleNo, int EndAngleNo,
                    int startEnergyNo, int endEnergyNo, cl::Buffer energy_buff,
                    string input_dir, string output_dir,string fileName_base,bool last)
{
    try {
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        string devicename = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>();
        size_t maxWorkGroupSize = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        
        
        int num_energy = endEnergyNo - startEnergyNo + 1;
        int paramsize = (int)fiteq.freeParaSize();
        
        const cl::NDRange local_item_size(maxWorkGroupSize,1,1);
        const cl::NDRange global_item_size(IMAGE_SIZE_X,IMAGE_SIZE_Y,1);
		if (startAngleNo == EndAngleNo) cout << "Processing angle No. " << startAngleNo << "..." << endl;
		else cout << "Processing angle No. " << startAngleNo <<"-"<<EndAngleNo<< "..." << endl;
		cout << "     global worksize :" << IMAGE_SIZE_X << ", " << IMAGE_SIZE_Y << ",1" << endl << endl;;
        
        
        cl::Buffer para_lowerLimit_buff(context, CL_MEM_READ_ONLY, sizeof(cl_float)*paramsize, 0, NULL);
        cl::Buffer para_upperLimit_buff(context, CL_MEM_READ_ONLY, sizeof(cl_float)*paramsize, 0, NULL);
        cl::Buffer lambda_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        cl::Buffer nyu_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
		cl::ImageFormat format(CL_R, CL_FLOAT);
		cl::Image2DArray mt_img(context, CL_MEM_READ_ONLY, format, num_energy,IMAGE_SIZE_X, IMAGE_SIZE_Y, 0, 0, NULL, NULL);
		cl::Image2DArray fp_img(context, CL_MEM_READ_WRITE, format, paramsize, IMAGE_SIZE_X, IMAGE_SIZE_Y, 0, 0, NULL, NULL);
        cl::Image2DArray fp_dest_img(context, CL_MEM_READ_WRITE, format, paramsize, IMAGE_SIZE_X, IMAGE_SIZE_Y, 0, 0, NULL, NULL);
        vector<vector<float*>> mt_vec;
        vector<vector<float*>> results_vec;
        
        
        cl::size_t<3> origin;
        cl::size_t<3> region;
        origin[0] = 0;
        origin[1] = 0;
        region[0] = IMAGE_SIZE_X;
        region[1] = IMAGE_SIZE_Y;
        region[2] = 1;
        
        //Initialize fitting parameter
        command_queue.enqueueWriteBuffer(para_lowerLimit_buff, CL_TRUE, 0, sizeof(cl_float)*paramsize, fiteq.freepara_lowerlimit,NULL,NULL);
        command_queue.enqueueWriteBuffer(para_upperLimit_buff, CL_TRUE, 0, sizeof(cl_float)*paramsize, fiteq.freepara_upperlimit,NULL,NULL);
        for (int i=0; i<paramsize; i++) {
            cl_float4 fillcolor = {fiteq.freefit_para()[i],0,0,1};
            origin[2] = i;
            command_queue.enqueueFillImage(fp_img, fillcolor, origin, region);
        }
        command_queue.enqueueFillBuffer(lambda_buff, (cl_float)(0.2f), 0, sizeof(float)*IMAGE_SIZE_M);
        
        
        for (int j=startAngleNo; j<=EndAngleNo; j++) {
            vector<float*> mt_vec_atAng;
            vector<float*> results_vec_atAng;
            for (int i=0; i<num_energy; i++) {
                mt_vec_atAng.push_back(new float[IMAGE_SIZE_M]);
            }
            
            for (int i=0; i<paramsize; i++) {
                results_vec_atAng.push_back(new float[IMAGE_SIZE_M]);
            }
            mt_vec.push_back(move(mt_vec_atAng));
            results_vec.push_back(move(results_vec_atAng));
        }
        
        
        //input mt data
        time_t start_t,readfinish_t;
        time(&start_t);
        for (int j=startAngleNo; j<=EndAngleNo; j++) {
            for (int i=0; i<num_energy; i++) {
                string filepath_input = input_dir+EnumTagString(i+startEnergyNo,"/",fileName_base)+AnumTagString(j, "", ".raw");
                //cout<<filepath_input;
                readRawFile(filepath_input,mt_vec[j-startAngleNo][i]);
            }
        }
        time(&readfinish_t);
        

        for (int j=startAngleNo; j<=EndAngleNo; j++) {
			//write buffer
			for (int i = 0; i<num_energy; i++) {
				origin[2] = i;
				command_queue.enqueueWriteImage(mt_img, CL_TRUE, origin, region, IMAGE_SIZE_X*sizeof(float), 0, mt_vec[j - startAngleNo][i], NULL, NULL);
			}
            
            for (int i=0; i<20; i++) {
                //XANES fitting
                kernel[0].setArg(0, mt_img);
                kernel[0].setArg(1, fp_img);
                kernel[0].setArg(2, fp_dest_img);
                kernel[0].setArg(3, energy_buff);
                kernel[0].setArg(4, lambda_buff);
                kernel[0].setArg(5, nyu_buff);
                command_queue.enqueueNDRangeKernel(kernel[0], NULL, global_item_size, local_item_size, NULL, NULL);
                command_queue.finish();
                
                origin[2] = 0;
                region[2] = paramsize;
                command_queue.enqueueCopyImage(fp_dest_img, fp_img, origin, origin, region);
                command_queue.finish();
            }
			
            
            //output result image with threashold
            kernel[1].setArg(1, fp_img);
            kernel[1].setArg(2, fp_dest_img);
            kernel[1].setArg(3, para_lowerLimit_buff);
            kernel[1].setArg(4, para_upperLimit_buff);
            command_queue.enqueueNDRangeKernel(kernel[1], NULL, global_item_size, local_item_size, NULL, NULL);
            command_queue.finish();
            
            //read result image
            region[2] = 1;
			for (int i = 0; i<paramsize; i++) {
				origin[2] = i;
				command_queue.enqueueReadImage(fp_dest_img, CL_TRUE, origin, region, IMAGE_SIZE_X*sizeof(float), 0, results_vec[j - startAngleNo][i], NULL, NULL);
			}
        }

		for (int j = startAngleNo; j <= EndAngleNo; j++) {
			for (int i = 0; i<num_energy; i++) {
				delete[] mt_vec[j - startAngleNo][i];
			}
		}
        
        int delta_t;
        if (last) {
            delta_t =0;
        } else {
            delta_t = 1;//difftime(readfinish_t,start_t)*1.25;
        }
        thread th_output(fitresult_output_thread, fiteq,
                         startAngleNo,EndAngleNo,
                         output_dir,move(results_vec),delta_t);
        if(last) th_output.join();
        else th_output.detach();

        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
    
    return 0;
}



int XANES_fit_ocl(fitting_eq fiteq, input_parameter inp,
                  OCL_platform_device plat_dev_list,string fileName_base)
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
    cout << "energy num for fitting: "<<num_energy<<endl<<endl;
    
    
    //kernel program source
    ifstream ifs("./XANES_fit.cl", ios::in);
    if(!ifs) {
        cerr << "   Failed to load kernel \n" << endl;
        return -1;
    }
    istreambuf_iterator<char> it(ifs);
    istreambuf_iterator<char> last;
    string kernel_src(it,last);
    ifs.close();
    
    
    //OpenCL Program
    vector<vector<cl::Kernel>> kernels;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        //OpenCL Program
        string kernel_code="";
        kernel_code += kernel_preprocessor_nums(E0,num_energy,fiteq.freeParaSize());
        kernel_code += fiteq.preprocessor_str();
        ostringstream oss;
        oss<<"#define NUM_TRIAL "<<20<<endl<<endl;
        oss<<"#define LAMBDA "<<0.001f<<endl<<endl;
        kernel_code += oss.str();
        //cout << kernel_code<<endl;
        kernel_code += kernel_src;
        //cout << kernel_code<<endl;
        size_t kernel_code_size = kernel_code.length();
        cl::Program::Sources source(1,std::make_pair(kernel_code.c_str(),kernel_code_size));
        cl::Program program(plat_dev_list.context(i), source,&ret);
        //kernel build
        ret=program.build();
        //cout<<ret<<endl;
        vector<cl::Kernel> kernels_plat;
        kernels_plat.push_back(cl::Kernel::Kernel(program,"XANES_fitting", &ret));//0
        kernels_plat.push_back(cl::Kernel::Kernel(program,"outputFitParaImage", &ret));//1
        
        kernels.push_back(kernels_plat);
    }
    
    
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
            cout << "CL DEVICE NAME: "<< device_pram<<endl;
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
    
    
    //energy_buffers作成(context毎)
    vector<cl::Buffer> energy_buffers;
    for (int i=0; i<plat_dev_list.contextsize(); i++){
        energy_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*num_energy, 0, NULL));
		plat_dev_list.queue(i,0).enqueueWriteBuffer(energy_buffers[i], CL_TRUE, 0, sizeof(cl_float)*num_energy, &energy[0], NULL, NULL);
    }
    
    
    //start thread
    vector<thread> th;
    for (int i=startAngleNo; i<=endAngleNo;) {
        for (int j=0; j<plat_dev_list.contextsize(); j++) {
            if (th.size()<plat_dev_list.contextsize()) {
                bool last = (i>endAngleNo-plat_dev_list.contextsize());
                th.push_back(thread(XANES_fit_thread,
                                    plat_dev_list.queue(j, 0),kernels[j],
                                    fiteq,
                                    i,i,startEnergyNo,endEnergyNo,
                                    energy_buffers[j],
                                    inp.getInputDir(),inp.getOutputDir(),fileName_base,last));
                i++;
                if (i > endAngleNo) break;
                else continue;
            }else if (th[j].joinable()) {
                bool last = (i>endAngleNo-plat_dev_list.contextsize());
                th[j].join();
                th[j] = thread(XANES_fit_thread,
                               plat_dev_list.queue(j, 0),kernels[j],
                               fiteq,
                               i,i,startEnergyNo,endEnergyNo,
                               energy_buffers[j],
                               inp.getInputDir(),inp.getOutputDir(),fileName_base,last);
                i++;
                if (i > endAngleNo) break;
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