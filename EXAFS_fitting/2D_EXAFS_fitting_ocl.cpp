//
//  2D_EXAFS_fitting.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2017/08/04.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#include "EXAFS.hpp"
#include "EXAFS_fit_cl.hpp"
#include "3D_FFT_cl.hpp"
#include "LevenbergMarquardt_cl.hpp"

int EXAFSfitresult_output_thread(int AngleNo, string output_dir, vector<string> paraName,
                                 vector<float*> result_outputs, int imageSizeM){
    
    ostringstream oss;
    for (int i=0; i<paraName.size(); i++) {
        string fileName_output= output_dir+ "/"+paraName[i]+ AnumTagString(AngleNo,"/i", ".raw");
        oss << "output file: " << fileName_output << endl;
        outputRawFile_stream(fileName_output,result_outputs[i],imageSizeM);
    }
    oss << endl;
    cout << oss.str();
    for (int i=0; i<paraName.size(); i++) {
        delete [] result_outputs[i];
    }
    return 0;
}



int EXAFS_fit_thread(cl::CommandQueue queue, cl::Program program,
                     int AngleNo, int thread_id, input_parameter inp,vector<string> paraName,
                     vector<shellObjects> shObjs, cl::Buffer w_factor, float* edgeJ,
                     vector<float*> chi_vec, vector<int> processImageSizeY){
    
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();
    const int imgSizeX = inp.getImageSizeX();
    const int imgSizeM = inp.getImageSizeM();
    const int procImgSizeY = processImageSizeY[0];
    const int FFTprocImgSizeY = processImageSizeY[1];
    const int procImgSizeM = imgSizeX*procImgSizeY;
    string output_dir=inp.getFittingOutputDir();
    const int fittingmode = inp.getEXAFSfittingMode();
    const int numTrial = inp.getNumTrial();
    const float lambda = inp.getLambda_t_fit();
    const float ini_S02 = inp.getIniS02();
    const bool S02freeFix = inp.getS02freeFix();
    int kw = inp.get_kw();
    float kstart = inp.get_kstart();
    float kend   = inp.get_kend();
    float Rstart = inp.get_Rstart();
    float Rend   = inp.get_Rend();
    float qstart = inp.get_qstart();
    float qend   = inp.get_qend();
    int ksize = (int)ceil((min(float(kend+WIN_DK),(float)MAX_KQ)-max(float(kstart-WIN_DK),0.0f))/KGRID) + 1;
    int Rsize = (int)ceil((min(float(Rend+WIN_DR),(float)MAX_R)-max(float(Rstart-WIN_DR),0.0f))/RGRID) + 1;
    int qsize = (int)ceil((min(float(qend+WIN_DK),(float)MAX_KQ)-max(float(qstart-WIN_DK),0.0f))/KGRID) + 1;
    
    
    //create fitting results img pointer
    vector<float*> results_vec;
    //S02 img
    results_vec.push_back(new float[imgSizeM]);
    for (int i=0; i<shObjs.size(); i++) {
        //CN img
        results_vec.push_back(new float[imgSizeM]);
        //bond distance img
        results_vec.push_back(new float[imgSizeM]);
        //dE0 img
        results_vec.push_back(new float[imgSizeM]);
        //ss img
        results_vec.push_back(new float[imgSizeM]);
    }
    
    
    //S02 img buffer;
    cl::Buffer S02(context,CL_MEM_READ_WRITE,sizeof(cl_float)*procImgSizeM,0,NULL);
    cl::Buffer Data_buff;
    cl::Buffer edgeJ_buff(context,CL_MEM_READ_WRITE,sizeof(cl_float)*procImgSizeM,0,NULL);
    for (int imgOffset=0; imgOffset<imgSizeM; imgOffset+=procImgSizeM) {
        //reset parameters
        queue.enqueueFillBuffer(S02, (cl_float)ini_S02, 0, sizeof(cl_float)*procImgSizeM);
        queue.enqueueWriteBuffer(edgeJ_buff, CL_TRUE, 0, sizeof(cl_float)*procImgSizeM, &edgeJ[imgOffset]);
        for (int sh=0; sh<shObjs.size(); sh++) {
            shObjs[sh].inputIniCN(inp.getEXAFSiniPara()[sh][0], edgeJ_buff);
            shObjs[sh].inputIniR(inp.getEXAFSiniPara()[sh][1]);
            shObjs[sh].inputInidE0(inp.getEXAFSiniPara()[sh][2]);
            shObjs[sh].inputIniss(inp.getEXAFSiniPara()[sh][3]);
        }
        
        
		//cl_float2 *FFTchidata;
		//FFTchidata = new cl_float2[procImgSizeM*Rsize];
        switch (fittingmode) {
            case 0: //k-fit
                Data_buff=cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*ksize*procImgSizeM,0,NULL);
                //data input
                ChiData_k(queue,program,Data_buff,move(chi_vec),kw, kstart, kend, imgSizeX, procImgSizeY, true, imgOffset);
                //fitting
                EXAFS_kFit(queue,program,Data_buff,S02,shObjs,kw,kstart,kend,imgSizeX,procImgSizeY, S02freeFix,numTrial,lambda);
                break;
                
            case 1: //R-fit
                Data_buff=cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*Rsize*procImgSizeM,0,NULL);
                //data input
                ChiData_R(queue,program,Data_buff,move(chi_vec),w_factor,kw,kstart,kend,Rstart,Rend,imgSizeX, procImgSizeY, FFTprocImgSizeY, true, imgOffset);
				/*queue.enqueueReadBuffer(Data_buff, CL_TRUE, 0, sizeof(cl_float2)*procImgSizeM*Rsize, FFTchidata);
				for (int i = 0; i < Rsize; i++) {
					cout << FFTchidata[i*procImgSizeM+ procImgSizeM/2+96].x << "\t" << FFTchidata[i*procImgSizeM + procImgSizeM / 2+96 ].y << endl;
				}*/
				//fitting
                EXAFS_RFit(queue,program,w_factor,Data_buff,S02,shObjs,kw,kstart,kend,Rstart,Rend, imgSizeX, procImgSizeY, FFTprocImgSizeY,S02freeFix,numTrial,lambda);
                break;
            
            case 2: //q-fit
                Data_buff=cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*qsize*procImgSizeM,0,NULL);
                //data input
                ChiData_q(queue,program,Data_buff,move(chi_vec),w_factor,kw,kstart,kend,Rstart,Rend, qstart,qend, imgSizeX, procImgSizeY, FFTprocImgSizeY, true, imgOffset);
                //fitting
                EXAFS_qFit(queue,program,w_factor,Data_buff,S02,shObjs,kw,kstart,kend,Rstart,Rend, qstart,qend,imgSizeX, procImgSizeY, FFTprocImgSizeY,S02freeFix,numTrial,lambda);
                break;
                
            default:
                break;
        }
        
        
        //read fitting results
        int t=1;
        //S02
        queue.enqueueReadBuffer(S02, CL_TRUE, 0, sizeof(float)*procImgSizeM, &results_vec[0][imgOffset]);
        //others
        for (int i=0; i<shObjs.size(); i++) {
            shObjs[i].readParaImage(&results_vec[t++][imgOffset], 1);
            shObjs[i].readParaImage(&results_vec[t++][imgOffset], 2);
            shObjs[i].readParaImage(&results_vec[t++][imgOffset], 3);
            shObjs[i].readParaImage(&results_vec[t++][imgOffset], 4);
        }
    }
    
    
    //delete chi data
    /*for (int i = 0; i<chi_vec.size(); i++) {
        delete[] chi_vec[i];
    }*/
	delete[] edgeJ;
    
    //output thread
    output_th_fit[thread_id].join();
    output_th_fit[thread_id]=thread(EXAFSfitresult_output_thread, AngleNo,
                                    output_dir, paraName, move(results_vec),imgSizeM);
    
    
    return 0;
}



int data_input_thread(cl::CommandQueue command_queue, cl::Program program,
                      int AngleNo, int thread_id, input_parameter inp,vector<string> paraName,
                      vector<shellObjects> shObjs, cl::Buffer w_factor,
                      vector<int> processImageSizeY){
    
    
    int startkNo = 1;
    int endkNo = (int)ceil((inp.get_kend()+WIN_DK)/KGRID)+1;
    int num_chi = endkNo -startkNo +1;
    const int imgSizeM = inp.getImageSizeM();
    
    string fileName_base = inp.getFittingFileBase();
	string fileName_base_ej = inp.getEJFileBase();
    string input_dir=inp.getInputDir();
    string input_dir_ej=inp.getInputDir_EJ();
    string filepath_input=input_dir+"/"+AnumTagString(AngleNo,fileName_base,".raw");
    string filepath_input_ej=input_dir_ej+"/"+AnumTagString(AngleNo,fileName_base_ej,".raw");

    
    
    //input chi data
    vector<float*> chi_vec;
    for (int i=0; i<num_chi; i++) {
        chi_vec.push_back(new float[imgSizeM]);
        readRawFile_offset(filepath_input,chi_vec[i],(int64_t)i*imgSizeM*sizeof(float), (int64_t)imgSizeM*sizeof(float));
    }

    
    //input edge jump data
    float* edgeJ;
    edgeJ = new float[imgSizeM];
    if (input_dir_ej.length()>0) {
        readRawFile_offset(filepath_input_ej,edgeJ,0,imgSizeM*sizeof(float));
    }else{
        for (int i=0; i<imgSizeM; i++) {
            edgeJ[i]=1.0f;
        }
    }
    
    
    fitting_th[thread_id].join();
    fitting_th[thread_id] = thread(EXAFS_fit_thread,command_queue,program,AngleNo,thread_id,inp,paraName, shObjs, w_factor, move(edgeJ),move(chi_vec),processImageSizeY);
    
    return 0;
}



//dummy thread
static int dummy(){
    return 0;
}



int EXAFS_fit_ocl(input_parameter inp, OCL_platform_device plat_dev_list,vector<FEFF_shell> shell)
{
    cl_int ret;
    int startAngleNo=inp.getStartAngleNo();
    int endAngleNo=inp.getEndAngleNo();
    const int imgSizeX = inp.getImageSizeX();
    const int imgSizeY = inp.getImageSizeY();
    float kstart= inp.get_kstart();
    float kend  = inp.get_kend();
    float Rstart= inp.get_Rstart();
    float Rend  = inp.get_Rend();
    float qstart= inp.get_qstart();
    float qend  = inp.get_qend();
    int ksize = (int)ceil((min(float(kend+WIN_DK),(float)MAX_KQ)-max(float(kstart-WIN_DK),0.0f))/KGRID) + 1;
    int Rsize = (int)ceil((min(float(Rend+WIN_DR),(float)MAX_R)-max(float(Rstart-WIN_DR),0.0f))/RGRID) + 1;
    int qsize = (int)ceil((min(float(qend+WIN_DK),(float)MAX_KQ)-max(float(qstart-WIN_DK),0.0f))/KGRID) + 1;
    
    int num_shell = (int)shell.size();
    int num_fpara = inp.getS02freeFix() ? 1:0;
    vector<string> shellName = inp.getShellName();
    vector<string> paraName;
    paraName.push_back("S02");
    for (int sh=0; sh<num_shell; sh++) {
        paraName.push_back(shellName[sh]+"_CN");
        paraName.push_back(shellName[sh]+"_R");
        paraName.push_back(shellName[sh]+"_dE0");
        paraName.push_back(shellName[sh]+"_ss");
        num_fpara += inp.getEXAFSfreeFixPara()[sh][0] ? 1:0;//CN
        num_fpara += inp.getEXAFSfreeFixPara()[sh][1] ? 1:0;//dR
        num_fpara += inp.getEXAFSfreeFixPara()[sh][2] ? 1:0;//dE0
        num_fpara += inp.getEXAFSfreeFixPara()[sh][3] ? 1:0;//ss
    }
    vector<vector<bool>> freefixpara = inp.getEXAFSfreeFixPara();
    
    //create output dir
    for (int i=0; i<paraName.size(); i++) {
        string fileName_output = inp.getFittingOutputDir() + "/"+paraName[i];
        MKDIR(fileName_output.c_str());
    }
    
    
    vector<vector<int>> processImageSizeYs;
    
    
    //OpenCL Program
    vector<cl::Program> programs;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        cl::Program::Sources source;
#if defined (OCL120)
        source.push_back(make_pair(kernel_EXAFSfit_src.c_str(),kernel_EXAFSfit_src.length()));
        source.push_back(make_pair(kernel_3D_FFT_src.c_str(),kernel_3D_FFT_src.length()));
		source.push_back(make_pair(kernel_LM_src.c_str(), kernel_LM_src.length()));
#else
        source.push_back(kernel_EXAFSfit_src);
        source.push_back(kernel_3D_FFT_src);
		source.push_back(kernel_LM_src);
#endif
        programs.push_back(cl::Program(plat_dev_list.context(i), source,&ret));
        //kernel build
        ostringstream oss;
        oss << "-D FFT_SIZE="<<FFT_SIZE<<" ";
        oss << "-D PARA_NUM="<<num_fpara<<" ";
        oss << "-D PARA_NUM_SQ="<<num_fpara*num_fpara<<" ";
        string option =oss.str();
#ifdef DEBUG
        option += "-D DEBUG -Werror";
#endif
        string GPUvendor =  plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0].getInfo<CL_DEVICE_VENDOR>();
        if(GPUvendor == "nvidia"){
            option += " -cl-nv-maxrregcount=64 -cl-nv-verbose";
        }
        else if (GPUvendor.find("NVIDIA Corporation")==0) {
            option += " -cl-nv-maxrregcount=64";
        }
        ret = programs[i].build(option.c_str());
#ifdef DEBUG
        string logstr=programs[i].getBuildInfo<CL_PROGRAM_BUILD_LOG>(plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0]);
        cout << logstr << endl;
#endif
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
    vector<cl::Buffer> w_factors;
    vector<vector<shellObjects>> shObjs;
    for (int i=0; i<plat_dev_list.contextsize(); i++){
        w_factors.push_back(cl::Buffer(plat_dev_list.context(i),CL_MEM_READ_WRITE,sizeof(cl_float2)*FFT_SIZE/2, 0, NULL));
        createSpinFactor(w_factors[i], plat_dev_list.queue(i,0), programs[i]);
        
        vector<shellObjects> shObj_atP;
        for (int sh=0; sh<num_shell; sh++) {
            shObj_atP.push_back(shellObjects(plat_dev_list.queue(i,0), programs[i], shell[sh], imgSizeX, imgSizeY));
        }
        shObjs.push_back(shObj_atP);
        
        
        //set free/fix
        for (int sh=0; sh<num_shell; sh++) {
            shObjs[i][sh].setFreeFixPara("CN", freefixpara[sh][0]);
            shObjs[i][sh].setFreeFixPara("dR", freefixpara[sh][1]);
            shObjs[i][sh].setFreeFixPara("dE0", freefixpara[sh][2]);
            shObjs[i][sh].setFreeFixPara("ss", freefixpara[sh][3]);
        }
        
        processImageSizeYs.push_back(GPUmemoryControl(imgSizeX,imgSizeY,ksize,Rsize,qsize,num_shell,num_fpara,shObjs[i],plat_dev_list.queue(i,0)));
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
                input_th[j] = thread(data_input_thread, plat_dev_list.queue(j,0), programs[j],
                                                       i, j, inp, paraName,shObjs[j],w_factors[j],
                                                       processImageSizeYs[j]);
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
