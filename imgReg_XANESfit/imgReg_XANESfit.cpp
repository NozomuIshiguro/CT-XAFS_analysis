//
//  imgReg_XANESfit.cpp
//  imgReg_XANESfit
//
//  Created by Nozomu Ishiguro on 2017/09/28.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#include "imgReg_XANESfit.hpp"

int data_input_thread(cl::CommandQueue command_queue, cl::Program program,
                      fitting_eq fiteq, int AngleNo, int thread_id,input_parameter inp,
                      cl::Buffer energy, cl::Buffer C_matrix_buff, cl::Buffer D_vector_buff,
                      cl::Buffer C2_vector_buff,cl::Buffer freeFix_buff, cl::Image1DArray refSpectra,
                      cl::Buffer funcMode_buff, int processImageSizeY){
    
    
    int startEnergyNo = inp.getFittingStartEnergyNo();
    int endEnergyNo = inp.getFittingEndEnergyNo();
    string fileName_base = inp.getFittingFileBase();
    string input_dir=inp.getInputDir();
    float* mt_pnt;
    string filepath_input;
    const int imgSizeM = inp.getImageSizeM();
    int num_energy = endEnergyNo -startEnergyNo +1;
    mt_pnt = new float[imgSizeM*num_energy];
    filepath_input = input_dir + AnumTagString(AngleNo,"/",".raw");
    readRawFile(filepath_input,mt_pnt,imgSizeM*num_energy);
    
    fitting_th[thread_id].join();
    fitting_th[thread_id] = thread(XANES_fit_thread,command_queue, program,
                                   fiteq,AngleNo,thread_id,inp,
                                   energy, C_matrix_buff, D_vector_buff, C2_vector_buff,
                                   freeFix_buff,refSpectra,funcMode_buff,
                                   move(mt_pnt),0,processImageSizeY);
    
    
    return 0;
}

//dummy thread
static int dummy(){
    return 0;
}

int imgReg_XANES_fit_ocl(fitting_eq fiteq, input_parameter inp,
                         OCL_platform_device plat_dev_list)
{
    float startEnergy=inp.getStartEnergy();
    float endEnergy=inp.getEndEnergy();
    float E0=inp.getE0();
    int startAngleNo=inp.getStartAngleNo();
    int endAngleNo=inp.getEndAngleNo();
    
    
    //create output dir
    for (int i=0; i<fiteq.ParaSize(); i++) {
        char buffer=fiteq.freefix_para()[i];
        if (atoi(&buffer)==1) {
            string fileName_output = inp.getFittingOutputDir() + "/"+fiteq.param_name(i);
            MKDIR(fileName_output.c_str());
        }
    }
    
    
    //energy file input & processing
    ifstream energy_ifs(inp.getEnergyFilePath(),ios::in);
    if (!energy_ifs.is_open()) {
        cout<<"energy file not found."<<endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(-1);
    }
    vector<float> energy;
    int i=0, startEnergyNo=0, endEnergyNo=0;
    do {
        string str;
        str = ifs_getline(&energy_ifs) ;
        if (energy_ifs.eof()) break;
        
        istringstream iss(str);
        string a;
        iss >> a;
        float aa;
        try {
            aa = stof(a);
        }catch(invalid_argument ret){ //ヘッダータグが存在する場合に入力エラーになる際への対応
            continue;
        }
        cout<<i<<": "<<aa;
        if ((aa>=startEnergy)&(aa<=endEnergy)) {
            energy.push_back(aa-E0);
            cout<<" <- fitting range";
            endEnergyNo = i;
        } else if(aa<startEnergy) {
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
    /*ifstream ifs("E:/Dropbox/CTprogram/XANES_fitting/XANES_fit.cl", ios::in);
     if(!ifs) {
     cerr << "   Failed to load kernel \n" << endl;
     return -1;
     }
     istreambuf_iterator<char> it(ifs);
     istreambuf_iterator<char> last;
     string kernel_code(it,last);
     ifs.close();*/
    
    
    //OpenCL Program
    vector<cl::Program> programs;
    vector<int> processImageSizeY;
    int imageSizeX = inp.getImageSizeX();
    int imageSizeY = inp.getImageSizeY();
    try {
        for (int i = 0; i<plat_dev_list.contextsize(); i++) {
            processImageSizeY.push_back(XANESGPUmemoryControl(imageSizeX,imageSizeY,num_energy,fiteq, plat_dev_list.queue(i,0)));
            cl::Program::Sources source;
#if defined (OCL120)
            source.push_back(std::make_pair(kernel_fit_src.c_str(), kernel_fit_src.length()));
            source.push_back(std::make_pair(kernel_LM_src.c_str(), kernel_LM_src.length()));
#else
            source.push_back(kernel_fit_src);
            source.push_back(kernel_LM_src);
#endif
            programs.push_back(cl::Program(plat_dev_list.context(i), source));
            //kernel build
            string option = "";
#ifdef DEBUG
            option += "-D DEBUG -Werror ";
#endif
            option += kernel_preprocessor_nums(fiteq, inp, processImageSizeY[i]);
            string GPUvendor = plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0].getInfo<CL_DEVICE_VENDOR>();
            if (GPUvendor == "nvidia") {
                option += " -cl-nv-maxrregcount=64 -cl-nv-verbose";
            }
            else if (GPUvendor.find("NVIDIA Corporation") == 0) {
                option += " -cl-nv-maxrregcount=64";
            }
            //cout << option << endl;
            programs[i].build(option.c_str());
#ifdef DEBUG
            string logstr = programs[i].getBuildInfo<CL_PROGRAM_BUILD_LOG>(plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0]);
            cout << logstr << endl;
#endif
        }
    }
    catch (const cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
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
            cout << "CL DEVICE NAME: " << device_pram << endl << endl;
            t++;
        }
    }
    
    
    //energy_buffers create (per context)
    vector<cl::Buffer> energy_buffers;
    vector<cl::Buffer> C_matrix_buffers;
    vector<cl::Buffer> D_vector_buffers;
    vector<cl::Buffer> C2_vector_buffers;
    vector<cl::Buffer> freeFix_buffers;
    vector<cl::Buffer> funcMode_buffers;
    cl::ImageFormat format(CL_R, CL_FLOAT);
    vector<cl::Image1DArray> refSpectra;
    int paramsize = (int)fiteq.ParaSize();
    int contrainsize = (int)fiteq.contrain_size;
    const cl::NDRange local_item_size(1,1,1);
    const cl::NDRange global_item_size(1,1,1);
    const cl::NDRange global_item_size2(contrainsize,1,1);
    for (int i=0; i<plat_dev_list.contextsize(); i++){
        energy_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*num_energy, 0, NULL));
        C_matrix_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*paramsize*contrainsize, 0, NULL));
        D_vector_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*contrainsize, 0, NULL));
        C2_vector_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float)*contrainsize, 0, NULL));
        freeFix_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_char)*paramsize, 0, NULL));
        funcMode_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_int)*fiteq.numFunc, 0, NULL));
        refSpectra.push_back(cl::Image1DArray(plat_dev_list.context(i), CL_MEM_READ_WRITE,format,fiteq.numLCF,IMAGE_SIZE_E, 0, NULL));
        
        
        //write buffers
        plat_dev_list.queue(i,0).enqueueWriteBuffer(energy_buffers[i], CL_TRUE, 0, sizeof(cl_float)*num_energy, &energy[0], NULL, NULL);
        for (int j = 0; j < contrainsize; j++) {
            plat_dev_list.queue(i, 0).enqueueWriteBuffer(C_matrix_buffers[i], CL_TRUE, sizeof(cl_float)*j*paramsize, sizeof(cl_float)*paramsize, &(fiteq.C_matrix[j][0]), NULL, NULL);
        }
        plat_dev_list.queue(i,0).enqueueWriteBuffer(D_vector_buffers[i], CL_TRUE, 0, sizeof(cl_float)*contrainsize, &fiteq.D_vector[0], NULL, NULL);
        plat_dev_list.queue(i,0).enqueueWriteBuffer(freeFix_buffers[i], CL_TRUE, 0, sizeof(cl_char)*paramsize, fiteq.freefix_para(), NULL, NULL);
        plat_dev_list.queue(i,0).enqueueWriteBuffer(funcMode_buffers[i], CL_TRUE, 0, sizeof(cl_int)*fiteq.numFunc, &(fiteq.funcmode[0]), NULL, NULL);
        
        //write buffers for LCF
        cl::Kernel kernel_LCFstd(programs[i],"redimension_refSpecta");
        cl::size_t<3> origin;
        cl::size_t<3> region;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        region[1] = 1;
        region[2] = 1;
        for (int j=0; j<fiteq.numLCF; j++) {
            cl::Image1D refSpectraRaw(plat_dev_list.context(i), CL_MEM_READ_ONLY,format, fiteq.LCFstd_size[j], 0, NULL);
            cl::Buffer refSpectraRawE(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*fiteq.LCFstd_size[j], 0, NULL);
            
            region[0] = fiteq.LCFstd_size[j];
            plat_dev_list.queue(i,0).enqueueWriteImage(refSpectraRaw,CL_TRUE,origin,region,0,0,&(fiteq.LCFstd_mt[j][0]));
            plat_dev_list.queue(i,0).enqueueWriteBuffer(refSpectraRawE, CL_TRUE,0,sizeof(cl_float)*fiteq.LCFstd_size[j], &(fiteq.LCFstd_E[j][0]));
            
            kernel_LCFstd.setArg(0, refSpectra[i]);
            kernel_LCFstd.setArg(1, refSpectraRaw);
            kernel_LCFstd.setArg(2, refSpectraRawE);
            kernel_LCFstd.setArg(3, (cl_int)fiteq.LCFstd_size[j]);
            kernel_LCFstd.setArg(4, (cl_int)j);
            
            plat_dev_list.queue(i,0).enqueueNDRangeKernel(kernel_LCFstd, NULL, global_item_size, local_item_size, NULL, NULL);
            plat_dev_list.queue(i,0).finish();
        }
        
        
        //estimate C2
        cl::Kernel kernel_contrain0(programs[i],"contrain_0");
        kernel_contrain0.setArg(0, C_matrix_buffers[i]);
        kernel_contrain0.setArg(1, C2_vector_buffers[i]);
        plat_dev_list.queue(i,0).enqueueNDRangeKernel(kernel_contrain0, NULL, global_item_size2, local_item_size, NULL, NULL);
        plat_dev_list.queue(i,0).finish();
    }
    
    /*float *spectra;
     spectra = new float[IMAGE_SIZE_E*2];
     cl::size_t<3> origin;
     cl::size_t<3> region;
     origin[0] = 0;
     origin[1] = 0;
     origin[2] = 0;
     region[0] = IMAGE_SIZE_E;
     region[1] = 2;
     region[2] = 1;
     plat_dev_list.queue(0, 0).enqueueReadImage(refSpectra[0], CL_TRUE, origin, region, 0, 0, spectra);
     string output_dir = inp.getFittingOutputDir();
     string fileName_output = output_dir + AnumTagString(1, "/sp", ".raw");
     outputRawFile_stream(fileName_output, spectra, IMAGE_SIZE_E*2);*/
    
    
    
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
                                     plat_dev_list.queue(j,0),programs[j],
                                     fiteq,i,j,inp,energy_buffers[j], C_matrix_buffers[j],
                                     D_vector_buffers[j],C2_vector_buffers[j],
                                     freeFix_buffers[j],refSpectra[j],funcMode_buffers[j],
                                     processImageSizeY[j]);
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
