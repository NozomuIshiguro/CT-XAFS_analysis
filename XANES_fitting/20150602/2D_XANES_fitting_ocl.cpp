//
//  2D_XANES_fitting_ocl.c
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/01/07.
//  Copyright (c) 2014-2015 Nozomu Ishiguro. All rights reserved.
//

#include "XANES_fitting.hpp"
#include "atan_lor_linear_fitting.hpp"
#include "XANES_fit_cl.hpp"
float VRAM_safe_percent=0.75;

class Device_object_para{
    size_t max_parameter_size;
    size_t GlobalMemSize;
    size_t totalBufferSize;
public:
    Device_object_para(size_t numE,size_t num_ffp, cl::Device device);
    size_t num_energy;
    size_t num_Ebuffer;
    size_t num_freefittingpara;
    size_t num_energyInBuffer;
    size_t num_buffShift;
};

Device_object_para::Device_object_para(size_t numE,size_t num_ffp, cl::Device device){
    num_energy = numE;
    num_Ebuffer = numE;
    num_freefittingpara =num_ffp;
    num_energyInBuffer=1;
    num_buffShift = IMAGE_SIZE_M;
    
    max_parameter_size=device.getInfo<CL_DEVICE_MAX_PARAMETER_SIZE>();
    cout<<"Max number of kernel object:"<<max_parameter_size<<endl;
    while (num_Ebuffer+num_freefittingpara+4>max_parameter_size/8) {
        num_Ebuffer = (num_Ebuffer%2==1)? num_Ebuffer/2+1:num_Ebuffer/2;
        num_energyInBuffer++;
    }
    cout<<"Number of buffer:"<<num_Ebuffer<<endl;
    GlobalMemSize = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
    totalBufferSize=(1+IMAGE_SIZE_M)*sizeof(float)*(num_energy+num_freefittingpara);
    
    cout<<"Global memory (VRAM) size: "<<GlobalMemSize<<" bytes"<<endl;
    cout<<"VRAM safty proportion: "<<VRAM_safe_percent*100<<"%"<<endl;
    while (totalBufferSize>GlobalMemSize*VRAM_safe_percent) {
        num_buffShift /= 2;
        totalBufferSize = (1+num_buffShift)*sizeof(float)*(num_energy+num_freefittingpara);
    }
    cout<<"Energies in buffer:"<<num_energyInBuffer<<endl<<endl;
    //cout<<"num_shift:"<<IMAGE_SIZE_X<<"x"<<num_buffShift/IMAGE_SIZE_X<<endl<<endl;
}

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
                outputRawFile_stream(fileName_output,result_outputs[j-startAngleNo][t]);
                t++;
            }
        }
    }
    oss << "\n";
    cout << oss.str();
    for (int j=startAngleNo; j<=EndAngleNo; j++) {
        for (int i=0; i<fit_eq.freeParaSize(); i++) {
            delete [] result_outputs[j-startAngleNo][i];
        }
    }
    return 0;
}

int XANES_fit_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                    Device_object_para d_objP, fitting_eq fiteq,
                    int startAngleNo, int EndAngleNo,
                    int startEnergyNo, int endEnergyNo, cl::Buffer energy_buff,
                    string input_dir, string output_dir,string fileName_base,bool last)
{
    try {
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        string devicename = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>();
        size_t maxWorkGroupSize = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        
        
        int num_energy = endEnergyNo - startEnergyNo + 1;
        int num_buffer = (int)d_objP.num_Ebuffer;
        size_t num_buffShift =d_objP.num_buffShift;
        size_t num_EinB= d_objP.num_energyInBuffer;
        int pramsize = (int)fiteq.freeParaSize();
        
        size_t global_item_size_y=num_buffShift/IMAGE_SIZE_X;
        const cl::NDRange local_item_size(maxWorkGroupSize,1,1);
        const cl::NDRange global_item_size(IMAGE_SIZE_X,global_item_size_y,1);
		if (startAngleNo == EndAngleNo) cout << "Processing angle No. " << startAngleNo << "..." << endl;
		else cout << "Processing angle No. " << startAngleNo <<"-"<<EndAngleNo<< "..." << endl;
        cout<<"     global worksize :"<<IMAGE_SIZE_X<<", "<<global_item_size_y<<",1"<<endl<<endl;;
        
        
        cl::Buffer fitting_para_buff(context, CL_MEM_READ_ONLY, sizeof(cl_float)*pramsize, 0, NULL);
        cl::Buffer para_lowerLimit_buff(context, CL_MEM_READ_ONLY, sizeof(cl_float)*pramsize, 0, NULL);
        cl::Buffer para_upperLimit_buff(context, CL_MEM_READ_ONLY, sizeof(cl_float)*pramsize, 0, NULL);
        vector<cl::Buffer> mt_buffs;
        vector<cl::Buffer> results_buffs;
        vector<vector<float*>> mt_imgs;
        vector<vector<float*>> results_imgs;
        
        command_queue.enqueueWriteBuffer(fitting_para_buff, CL_TRUE, 0, sizeof(cl_float)*pramsize, fiteq.freefit_para(),NULL,NULL);
        command_queue.enqueueWriteBuffer(para_lowerLimit_buff, CL_TRUE, 0, sizeof(cl_float)*pramsize, fiteq.freepara_lowerlimit,NULL,NULL);
        command_queue.enqueueWriteBuffer(para_upperLimit_buff, CL_TRUE, 0, sizeof(cl_float)*pramsize, fiteq.freepara_upperlimit,NULL,NULL);
        
        for (int j=startAngleNo; j<=EndAngleNo; j++) {
            vector<float*> mt_imgs_atAng;
            vector<float*> results_imgs_atAng;
            for (int i=0; i<num_energy; i++) {
                mt_imgs_atAng.push_back(new float[IMAGE_SIZE_M]);
            }
            
            for (int i=0; i<pramsize; i++) {
                results_imgs_atAng.push_back(new float[IMAGE_SIZE_M]);
            }
            mt_imgs.push_back(move(mt_imgs_atAng));
            results_imgs.push_back(move(results_imgs_atAng));
        }
        
        
        //mt_img buffers
        for (int i=0; i<num_buffer; i++) {
            mt_buffs.push_back(cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(cl_float)*num_EinB*num_buffShift, 0, NULL));
        }
        // results buffers
        for (int i=0; i<pramsize; i++) {
            results_buffs.push_back(cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float)*num_buffShift, 0, NULL));
        }
        
        
        
        //input mt data
        time_t start_t,readfinish_t;
        time(&start_t);
        for (int j=startAngleNo; j<=EndAngleNo; j++) {
            for (int i=0; i<num_energy; i++) {
                string filepath_input = input_dir+EnumTagString(i+startEnergyNo,"/",fileName_base)+AnumTagString(j, "", ".raw");
                //cout<<filepath_input;
                readRawFile(filepath_input,mt_imgs[j-startAngleNo][i]);
            }
        }
        time(&readfinish_t);
        
        for (int j=startAngleNo; j<=EndAngleNo; j++) {
            for (int n=0; n<IMAGE_SIZE_M; n+=num_buffShift) {
                
                //write buffer
                for (int i=0; i<num_buffer; i++) {
                    for (int k=0; k<num_EinB; k++) {
                        if (i*num_EinB+k==num_energy) break;
                        //cout <<"energy:"<<i*num_EinB+k<<",buffer:"<<i<<endl;
                        command_queue.enqueueWriteBuffer(mt_buffs[i], CL_TRUE, sizeof(cl_float)*num_buffShift*k, sizeof(cl_float)*num_buffShift, mt_imgs[j-startAngleNo][i*num_EinB+k]+n,NULL,NULL);
                    }
                }
                
                
                
                //XANES fitting
                for (int i=0; i<num_buffer; i++) {
                    kernel[0].setArg(i, mt_buffs[i]);
                }
                //kernel[0].setArg(num_buffer, cl::Local(sizeof(cl_float*)*num_energy));
                for (int i=0; i<pramsize; i++) {
                    kernel[0].setArg(i+num_buffer, results_buffs[i]);
                }
                //kernel[0].setArg(num_buffer+pramsize+1,cl::Local(sizeof(cl_float*)*num_energy));
                kernel[0].setArg(num_buffer+pramsize, energy_buff);
                kernel[0].setArg(num_buffer+pramsize+1, fitting_para_buff);
                kernel[0].setArg(num_buffer+pramsize+2, para_lowerLimit_buff);
                kernel[0].setArg(num_buffer+pramsize+3, para_upperLimit_buff);
                command_queue.enqueueNDRangeKernel(kernel[0], NULL, global_item_size, local_item_size, NULL, NULL);
                command_queue.finish();
                
                for (int i=0; i<pramsize; i++) {
                    command_queue.enqueueReadBuffer(results_buffs[i], CL_TRUE, 0, sizeof(cl_float)*num_buffShift, results_imgs[j-startAngleNo][i]+n, NULL, NULL);
                }
            }
        }

		for (int j = startAngleNo; j <= EndAngleNo; j++) {
			for (int i = 0; i<num_energy; i++) {
				delete[] mt_imgs[j - startAngleNo][i];
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
                         output_dir,move(results_imgs),delta_t);
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
    
    vector<cl::Program> programs;
    vector<vector<cl::Kernel>> kernels;
    vector<Device_object_para> d_objP;
    
    //energy file input & processing
    ifstream energy_ifs(inp.getEnergyFilePath(),ios::in);
    vector<float> energy;
    int i=0, startEnergyNo=0, endEnergyNo=0;
    while (!energy_ifs.eof()) {
        float a;
        energy_ifs>>a;
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
    }
    int num_energy=endEnergyNo-startEnergyNo+1;
    cout << "energy num for fitting: "<<num_energy<<endl<<endl;
    
    
    //kernel program source
    /*ifstream ifs("./XANES_fit_headerless.cl", ios::in);
    if(!ifs) {
        cerr << "   Failed to load kernel \n" << endl;
        return -1;
    }
    istreambuf_iterator<char> it(ifs);
    istreambuf_iterator<char> last;
    string kernel_src(it,last);
    ifs.close();*/
    

    for (int i=0; i<plat_dev_list.size(); i++) {
        
        cout<<"Device No. "<<i+1<<"\n";
            
        string platform_param;
        plat_dev_list.plat(i).getInfo(CL_PLATFORM_NAME, &platform_param);
        cout << "CL PLATFORM NAME: "<< platform_param<<"\n";
        plat_dev_list.plat(i).getInfo(CL_PLATFORM_VERSION, &platform_param);
        cout << "   "<<platform_param<<"\n";
        string device_pram;
        plat_dev_list.dev(i).getInfo(CL_DEVICE_NAME, &device_pram);
        cout << "CL DEVICE NAME: "<< device_pram<<"\n";
        cout << "\n";
        
        //estimation of num_buffer, num_shift
        d_objP.push_back(Device_object_para(num_energy,fiteq.freeParaSize(), plat_dev_list.dev(i)));
        
        //OpenCL Program
        string kernel_code="";
        kernel_code += kernel_preprocessor_def("MT", "__global float *",
                                                   "mt","mt_p",
                                               d_objP[i].num_energy,d_objP[i].num_Ebuffer,
                                               d_objP[i].num_buffShift);
        kernel_code += kernel_preprocessor_def("FIT_RESULTS", "__global float *",
                                                "fit_results_img","fit_results_p",
                                                d_objP[i].num_freefittingpara,
                                               d_objP[i].num_freefittingpara,0);
        kernel_code += kernel_preprocessor_nums(E0,num_energy,fiteq.freeParaSize());
        kernel_code += kernel_preprocessor_read("MT_DATA","mt",d_objP[i].num_energy,
                                                d_objP[i].num_Ebuffer,
                                                d_objP[i].num_buffShift);
        kernel_code += kernel_preprocessor_write("FIT_RESULTS", "fit_results_img",
                                                 d_objP[i].num_freefittingpara,
                                                 d_objP[i].num_freefittingpara,0);
        kernel_code += fiteq.preprocessor_str();
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
        kernels.push_back(kernels_plat);
        
    }
    
    
    
    vector<cl::Buffer> energy_buffers;
    for (int i=0; i<plat_dev_list.size(); i++){
        energy_buffers.push_back(cl::Buffer(plat_dev_list.queue(i).getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(cl_float)*num_energy, (float*)&energy[0], NULL));
    }
    
    
    vector<thread> th;
    for (int i=startAngleNo; i<=endAngleNo;) {
        for (int j=0; j<plat_dev_list.size(); j++) {
            if (th.size()<plat_dev_list.size()) {
                bool last = (i>endAngleNo-plat_dev_list.size());
                th.push_back(thread(XANES_fit_thread,
                                    plat_dev_list.queue(j),kernels[j],
                                    d_objP[j],fiteq,
                                    i,i,startEnergyNo,endEnergyNo,
                                    energy_buffers[j],
                                    inp.getInputDir(),inp.getOutputDir(),fileName_base,last));
                i++;
                if (i > endAngleNo) break;
                else continue;
            }else if (th[j].joinable()) {
                bool last = (i>endAngleNo-plat_dev_list.size());
                th[j].join();
                th[j] = thread(XANES_fit_thread,
                               plat_dev_list.queue(j),kernels[j],
                               d_objP[j],fiteq,
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
    
    for (int j=0; j<plat_dev_list.size(); j++) {
        if (th[j].joinable()) th[j].join();
    }
    

    return 0;
}