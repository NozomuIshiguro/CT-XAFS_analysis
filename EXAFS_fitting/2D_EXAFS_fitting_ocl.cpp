//
//  2D_EXAFS_fitting.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2017/08/04.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#include "EXAFS.hpp"
#include "EXAFS_fit_cl.hpp"
#include "threeD_FFT_cl.hpp"
#include "LevenbergMarquardt_cl.hpp"

int outputFittingcondition(input_parameter inp, string filepath){
    ofstream ofs(filepath,ios::out|ios::trunc);
    
    //get date & time
    time_t t = time(nullptr);
    tm lt;
    LOCALTIME(&t,&lt);
    stringstream s;
    s<<"20"<<lt.tm_year-100<<"/"<<lt.tm_mon+1<<"/"<<lt.tm_mday;
    s<<" ";
    s<<lt.tm_hour<<":"<<lt.tm_min<<":"<<lt.tm_sec;
    
    ofs<<"EXAFS fitting executed at "<< s.str() << endl<<endl;
    ofs<< "fitting mode: ";
    switch (inp.getEXAFSfittingMode()) {
        case 0: //kspace
            ofs <<"k-space"<<endl<<endl;
            break;
        
        case 1: //Rspace
            ofs <<"R-space"<<endl<<endl;
            break;
            
        case 2: //qspace
            ofs <<"q-space"<<endl<<endl;
            break;
    }
    
    ofs<<"k-weight: "<<inp.get_kw()<<endl<<endl;
    
    switch (inp.getEXAFSfittingMode()) {
        case 0: //kspace
            ofs <<"k range: "<< inp.get_kstart() << " - "<< inp.get_kend() << endl;
            break;
            
        case 1: //Rspace
            ofs <<"k range: "<< inp.get_kstart() << " - "<< inp.get_kend() << endl;
            ofs <<"R range: "<< inp.get_Rstart() << " - "<< inp.get_Rend() << endl;
            break;
            
        case 2: //qspace
            ofs <<"k range: "<< inp.get_kstart() << " - "<< inp.get_kend() << endl;
            ofs <<"R range: "<< inp.get_Rstart() << " - "<< inp.get_Rend() << endl;
            ofs <<"k range: "<< inp.get_qstart() << " - "<< inp.get_qend() << endl;
            break;
    }
    ofs << endl;
    
    for (int i=0; i<inp.getShellNum(); i++) {
        ofs<<"Shell "<<i+1<<": "<<inp.getShellName()[i]<<endl;
        ofs<<"\t"<<inp.getFeffxxxxdatPath()[i]<<endl;
    }
    ofs<<endl;
    
    ofs<<"Free paramater: ";
    bool ini=false;
    if (inp.getS02freeFix()) {
        ofs<<"S02";
        ini=true;
    }
    for (int i=0; i<inp.getShellNum(); i++) {
        if (inp.getEXAFSfreeFixPara()[i][0]) {
            if(ini) ofs<<",";
            ofs<<"CN_"<<inp.getShellName()[i];
            ini=true;
        }
        if (inp.getEXAFSfreeFixPara()[i][1]) {
            if(ini) ofs<<",";
            ofs<<"R_"<<inp.getShellName()[i];
            ini=true;
        }
        if (inp.getEXAFSfreeFixPara()[i][2]) {
            if(ini) ofs<<",";
            ofs<<"dE0_"<<inp.getShellName()[i];
            ini=true;
        }
        if (inp.getEXAFSfreeFixPara()[i][3]) {
            if(ini) ofs<<",";
            ofs<<"ss_"<<inp.getShellName()[i];
            ini=true;
        }
    }
    ofs<<endl<<endl;
    
    
    return 0;
}


int correctBondDistanceContrain(vector<vector<float>> *C_matrix, vector<float> *D_vector,
                                int fpnum, float Reff){
    
    int contrainSize = (int)(*D_vector).size();
    for (int i=0; i<contrainSize; i++) {
        (*D_vector)[i] -= Reff*(*C_matrix)[i][fpnum];
    }
    
    return 0;
}


int EXAFSfitresult_output_thread(int AngleNo, string output_dir, vector<string> paraName,
                                 vector<float*> result_outputs, float* Rfactor,
                                 vector<float*> chiFitData, int imageSizeM, bool chifitout){
    
    ostringstream oss;
    for (int i=0; i<paraName.size(); i++) {
        string fileName_output= output_dir+ "/"+paraName[i]+ AnumTagString(AngleNo,"/i", ".raw");
        oss << "output file: " << fileName_output << endl;
        outputRawFile_stream(fileName_output,result_outputs[i],imageSizeM);
    }
    string fileName_output= output_dir+ "/Rfactor"+ AnumTagString(AngleNo,"/i", ".raw");
    oss << "output file: " << fileName_output << endl;
    outputRawFile_stream(fileName_output,Rfactor,imageSizeM);
    
    if (chifitout) {
        string fileName_output= output_dir+ "/chiFit" + AnumTagString(AngleNo,"/chiFit", ".raw");
        outputRawFile_stream_batch(fileName_output, chiFitData, imageSizeM);
        oss << "output file: " << fileName_output << endl;
        
        for (int i=0; i<chiFitData.size(); i++) {
            delete [] chiFitData[i];
        }
    }
    
    oss << endl;
    cout << oss.str();
    for (int i=0; i<paraName.size(); i++) {
        delete [] result_outputs[i];
    }
    delete [] Rfactor;
    return 0;
}



int EXAFS_fit_thread(cl::CommandQueue queue, cl::Program program,
                     int AngleNo, int thread_id, input_parameter inp,vector<string> paraName,
                     vector<shellObjects> shObjs, cl::Buffer w_factor, float* edgeJ,
                     vector<float*> chi_vec, vector<int> processImageSizeY, int contrainSize,
                     cl::Buffer C_matrix_buff, cl::Buffer D_vector_buff, cl::Buffer C2_vector_buff,
                     cl::Buffer CSlambda_buff){
    
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
    bool CSbool = inp.getCSbool();
    
    
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
    float* Rfactor;
    Rfactor = new float[imgSizeM];
    
    bool chiFitOut=inp.getChiFitOutBool();;
    vector<float*> chiFitData;
    int startkNo = 0;
    int endkNo = 20/KGRID;
    int num_chi = endkNo -startkNo +1;
    if (chiFitOut) {
        for (int i=0; i<num_chi; i++) {
            chiFitData.push_back(new float[imgSizeM]);
        }
    }

    //S02 img buffer;
    cl::Buffer S02(context,CL_MEM_READ_WRITE,sizeof(cl_float)*procImgSizeM,0,NULL);
    cl::Buffer Data_buff;
    cl::Buffer edgeJ_buff(context,CL_MEM_READ_WRITE,sizeof(cl_float)*procImgSizeM,0,NULL);
    cl::Buffer Rfactor_buff(context,CL_MEM_READ_WRITE,sizeof(cl_float)*procImgSizeM,0,NULL);
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
        
        
        switch (fittingmode) {
            case 0: //k-fit
                Data_buff=cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*ksize*procImgSizeM,0,NULL);
                //data input
                ChiData_k(queue,program,Data_buff,move(chi_vec),kw, kstart, kend, imgSizeX, procImgSizeY, true, imgOffset);
                break;
                
            case 1: //R-fit
                Data_buff=cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*Rsize*procImgSizeM,0,NULL);
                //data input
                ChiData_R(queue,program,Data_buff,move(chi_vec),w_factor,kw,kstart,kend,Rstart,Rend,imgSizeX, procImgSizeY, FFTprocImgSizeY, true, imgOffset);
                break;
            
            case 2: //q-fit
                Data_buff=cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float2)*qsize*procImgSizeM,0,NULL);
                //data input
                ChiData_q(queue,program,Data_buff,move(chi_vec),w_factor,kw,kstart,kend,Rstart,Rend, qstart,qend, imgSizeX, procImgSizeY, FFTprocImgSizeY, true, imgOffset);
                break;
        }
        //fitting
        EXAFS_Fit(queue,program,w_factor,Data_buff,S02,Rfactor_buff,shObjs,kw,kstart,kend,Rstart,Rend,qstart,qend,imgSizeX,procImgSizeY, FFTprocImgSizeY,fittingmode,S02freeFix,numTrial,lambda,contrainSize,edgeJ_buff,C_matrix_buff,D_vector_buff,C2_vector_buff,CSbool,CSlambda_buff);
        
        
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
        //Rfactor
        queue.enqueueReadBuffer(Rfactor_buff, CL_TRUE, 0, sizeof(float)*procImgSizeM, &Rfactor[imgOffset]);
        
        //output chiFit
        if (chiFitOut) {
            cl::Buffer chiFit_buff(context,CL_MEM_READ_WRITE,sizeof(cl_float)*procImgSizeM*num_chi,0,NULL);
            
            outputFit_r(queue, program, chiFit_buff, S02, shObjs, 0, 20, imgSizeX, procImgSizeY, 0);
            
            for (int i=0; i<num_chi; i++) {
                queue.enqueueReadBuffer(chiFit_buff, CL_TRUE, sizeof(cl_float)*procImgSizeM*i, sizeof(cl_float)*procImgSizeM, &chiFitData[i][imgOffset]);
            }
        }
    }
    
	delete[] edgeJ;
    
    
    
    //output thread
    output_th_fit[thread_id].join();
    output_th_fit[thread_id]=thread(EXAFSfitresult_output_thread, AngleNo,output_dir, paraName,
                                    move(results_vec),move(Rfactor),move(chiFitData),
                                    imgSizeM,chiFitOut);
    
    
    return 0;
}



int data_input_thread(cl::CommandQueue command_queue, cl::Program program,
                      int AngleNo, int thread_id, input_parameter inp,vector<string> paraName,
                      vector<shellObjects> shObjs, cl::Buffer w_factor, vector<int> processImageSizeY,
                      int contrainSize, cl::Buffer C_matrix_buff, cl::Buffer D_vector_buff,
                      cl::Buffer C2_vector_buff, cl::Buffer CSlambda_buff){
    
    
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
    fitting_th[thread_id] = thread(EXAFS_fit_thread,command_queue,program,AngleNo,thread_id,inp,
                                   paraName, shObjs, w_factor, move(edgeJ),
                                   move(chi_vec),processImageSizeY, contrainSize,
                                   C_matrix_buff, D_vector_buff, C2_vector_buff, CSlambda_buff);
    
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
    vector<vector<bool>> freefixpara = inp.getEXAFSfreeFixPara();
    vector<string> paraName;
    paraName.push_back("S02");
    for (int sh=0; sh<num_shell; sh++) {
        paraName.push_back("CN_"+shellName[sh]);
        paraName.push_back("R_"+shellName[sh]);
        paraName.push_back("dE0_"+shellName[sh]);
        paraName.push_back("ss_"+shellName[sh]);
        num_fpara += freefixpara[sh][0] ? 1:0;//CN
        num_fpara += freefixpara[sh][1] ? 1:0;//dR
        num_fpara += freefixpara[sh][2] ? 1:0;//dE0
        num_fpara += freefixpara[sh][3] ? 1:0;//ss
    }
    vector<string> fparaName;
    vector<float> CSlambda;
    if (inp.getS02freeFix()) {
        fparaName.push_back(paraName[0]);
        CSlambda.push_back(inp.getCSlambda()[0]);
    }
    for (int sh=0; sh<num_shell; sh++) {
        for (int pn=0; pn<4; pn++) {
            if (freefixpara[sh][pn]) {
                fparaName.push_back(paraName[pn+1+4*sh]);
                CSlambda.push_back(inp.getCSlambda()[pn+1+4*sh]);
            }
        }
    }
    //contrain setting
    vector<string> contrain_eqs=inp.getContrainEqs();
    vector<vector<float>> C_matrix;
    vector<float> D_vector;
    int contrainSize=createContrainMatrix(contrain_eqs,fparaName,&C_matrix,&D_vector,0);
    int fpn=0;
    if (inp.getS02freeFix()) fpn++;
    for (int sh=0; sh<num_shell; sh++) {
        if(freefixpara[sh][0]) fpn++; //CN
        if(freefixpara[sh][1]) {//dR
            correctBondDistanceContrain(&C_matrix, &D_vector, fpn, shell[sh].getReff());
            fpn++;
        }
        if(freefixpara[sh][2]) fpn++; //dE0
        if(freefixpara[sh][3]) fpn++; //ss
    }
    cout<<endl<<"C_matrix | D_vector"<<endl;
    for (int i=0; i<contrainSize; i++) {
        for (int j=0; j<num_fpara-1; j++) {
            cout << C_matrix[i][j] <<" ";
        }
        cout<< C_matrix[i][num_fpara-1] <<" | "<< D_vector[i] << endl;
    }
    cout<<endl;
    
    
    //create output dir
    for (int i=0; i<paraName.size(); i++) {
        string fileName_output = inp.getFittingOutputDir() + "/"+paraName[i];
        MKDIR(fileName_output.c_str());
    }
    string fileName_output = inp.getFittingOutputDir() + "/Rfactor";
    MKDIR(fileName_output.c_str());
    bool chiFitout=inp.getChiFitOutBool();
    if (chiFitout) {
        string fileName_output = inp.getFittingOutputDir() + "/chiFit";
        MKDIR(fileName_output.c_str());
    }
    

    
    
    //OpenCL Program
    vector<cl::Program> programs;
    vector<vector<int>> processImageSizeYs;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        processImageSizeYs.push_back(GPUmemoryControl(imgSizeX,imgSizeY,ksize,Rsize,qsize,num_shell,num_fpara,num_shell,plat_dev_list.queue(i,0)));
        
        cl::Program::Sources source;
#if defined (OCL120)
        source.push_back(make_pair(kernel_EXAFSfit_src.c_str(),kernel_EXAFSfit_src.length()));
        source.push_back(make_pair(kernel_threeD_FFT_src.c_str(),kernel_threeD_FFT_src.length()));
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
        oss << "-D CONTRAIN_NUM="<<contrainSize<<" ";
        oss << "-D IMAGESIZE_X="<<imgSizeX<<" ";
        oss << "-D IMAGESIZE_Y="<<processImageSizeYs[i][0]<<" ";
        oss << "-D IMAGESIZE_M="<<imgSizeX*processImageSizeYs[i][0]<<" ";
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
    vector<cl::Buffer> C_mats, D_vecs, C2_vecs;
    vector<cl::Buffer> CSlambda_buff;
    const cl::NDRange local_item_size(1,1,1);
    const cl::NDRange global_item_size(contrainSize,1,1);
    for (int i=0; i<plat_dev_list.contextsize(); i++){
        w_factors.push_back(cl::Buffer(plat_dev_list.context(i),CL_MEM_READ_WRITE,sizeof(cl_float2)*FFT_SIZE/2, 0, NULL));
        createSpinFactor(w_factors[i], plat_dev_list.queue(i,0), programs[i]);
        
        vector<shellObjects> shObj_atP;
        for (int sh=0; sh<num_shell; sh++) {
            shObj_atP.push_back(shellObjects(plat_dev_list.queue(i,0), programs[i], shell[sh], imgSizeX, processImageSizeYs[i][0]));
        }
        shObjs.push_back(shObj_atP);
        
        
        //set free/fix
        for (int sh=0; sh<num_shell; sh++) {
            shObjs[i][sh].setFreeFixPara("CN", freefixpara[sh][0]);
            shObjs[i][sh].setFreeFixPara("dR", freefixpara[sh][1]);
            shObjs[i][sh].setFreeFixPara("dE0", freefixpara[sh][2]);
            shObjs[i][sh].setFreeFixPara("ss", freefixpara[sh][3]);
        }
        
        
        //contrain
        C_mats.push_back(cl::Buffer(plat_dev_list.context(i),CL_MEM_READ_WRITE,sizeof(cl_float)*contrainSize*num_fpara, 0, NULL));
        D_vecs.push_back(cl::Buffer(plat_dev_list.context(i),CL_MEM_READ_WRITE,sizeof(cl_float)*contrainSize, 0, NULL));
        C2_vecs.push_back(cl::Buffer(plat_dev_list.context(i),CL_MEM_READ_WRITE,sizeof(cl_float)*contrainSize, 0, NULL));
        for (int cn=0; cn<contrainSize; cn++) {
            plat_dev_list.queue(i,0).enqueueWriteBuffer(C_mats[i], CL_TRUE, sizeof(cl_float)*num_fpara*cn, sizeof(cl_float)*num_fpara, &C_matrix[cn][0]);
        }
        plat_dev_list.queue(i,0).enqueueWriteBuffer(D_vecs[i], CL_TRUE, 0, sizeof(cl_float)*contrainSize, &D_vector[0]);
        cl::Kernel kernel_contrain0(programs[i],"contrain_0");
        kernel_contrain0.setArg(0, C_mats[i]);
        kernel_contrain0.setArg(1, C2_vecs[i]);
        plat_dev_list.queue(i,0).enqueueNDRangeKernel(kernel_contrain0, NULL, global_item_size, local_item_size, NULL, NULL);
        plat_dev_list.queue(i,0).finish();
        
        CSlambda_buff.push_back(cl::Buffer(plat_dev_list.context(i),CL_MEM_READ_WRITE,sizeof(cl_float)*num_fpara, 0, NULL));
        if(inp.getCSbool()){
            plat_dev_list.queue(i,0).enqueueWriteBuffer(CSlambda_buff[i], CL_TRUE, 0, sizeof(cl_float)*num_fpara, &CSlambda[0]);
        }
    }
    
    string filepath=inp.getFittingOutputDir() + "/fittingCondition.log";
    outputFittingcondition(inp, filepath);
    
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
                                     i, j, inp, paraName,shObjs[j],w_factors[j],processImageSizeYs[j],
                                     contrainSize, C_mats[j], D_vecs[j], C2_vecs[j], CSlambda_buff[j]);
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
