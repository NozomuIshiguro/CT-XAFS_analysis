//
//  test.cpp
//  CT-XAFS_analysis
//
//  Created by Nozomu Ishiguro on 2017/08/14.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#include "EXAFS_extraction.hpp"
#include "XANES_fit_cl.hpp"
#include "EXAFS_fit_cl.hpp"
#include "3D_FFT_cl.hpp"
#include "LevenbergMarquardt_cl.hpp"

int testEXAFSextraction(){
    
    float preEdgeStartE  = -300.0f;
    float preEdgeEndE    = -50.0f;
    float postEdgeStartE = 100.0f;
    float postEdgeEndE   = 800.0f;
    float E0 = 11559.0f;
    float Rbkg = 1.0f;
    int kw=1;
    float kstart = 0.0f;
    float kend   = 17.35f;
    int preEdgeFittingMode = 1;
    int num_param_preEdge=1;
    int num_paramsq_preEdge=1;
    switch (preEdgeFittingMode) {
        case 0: //line
        num_param_preEdge=2;
        num_paramsq_preEdge=4;
        break;
        
        case 1: //victoreen
        num_param_preEdge=4;
        num_paramsq_preEdge=16;
        break;
        
        case 2: //victoreen
        num_param_preEdge=3;
        num_paramsq_preEdge=9;
        break;
        
        default:
        break;
    }
    int imageSizeX=1;
    int processImageSizeY=1;
    int processImageSizeM=1;
    
    OCL_platform_device plat_dev_list("2"/*inp.getPlatDevList()*/,false);
    
    //energy file input
    string inputpath = "/Users/ishiguro/Documents/実験/Fuel_Cell/FC_MEA(0.5mg_cm-2)_13PCCP/20111023-26_mu/Pt\ foil\ (Pt01.txt).xmu";
    char* buffer;
    buffer = new char[512];
        ifstream chi_ifs(inputpath, ios::in);
        vector<float> e_vec, mt_vec;
        int npnts = 0;
    do {
        chi_ifs.getline(buffer, 512);
        istringstream iss(buffer);
        string a, b, c, d, e, f;
        iss >> a >> b >> c >> d >> e >> f;
        //cout << a <<"\t"<< b <<"\t"<< c <<"\t"<< d <<"\t"<< e <<"\t"<< f<<endl;
            
        if (chi_ifs.eof()) break;
        float aa, bb;
        try {
            aa = stof(a);
            bb = stof(b);
        }
        catch (invalid_argument ret) { //ヘッダータグが存在する場合に入力エラーになる際への対応
            continue;
        }
        e_vec.push_back(aa-E0);
        mt_vec.push_back(bb);
        //cout<<npnts+1<<": "<<e_vec[npnts]<<","<<mt_vec[npnts]<<endl;
        npnts++;
    } while (!chi_ifs.eof());
    chi_ifs.close();
    //cout<<endl;
    
    //energy No searching
    int num_evec = (int)e_vec.size();
    float kmax = floor(sqrt(e_vec[num_evec-1]*EFF)/KGRID-1.0f)*KGRID;
    kend = fmin(kend,kmax-WIN_DK);
    cout<< "spline end k: "<<kend<<endl;
    float splineStartE = fmax(0.0f,kstart-WIN_DK)*fmax(0.0f,kstart-WIN_DK)/EFF;
    float splineEndE = fmin(20.0f,kend+WIN_DK)*fmin(20.0f,kend+WIN_DK)/EFF;
    int N_ctrlP = (int)floor(2.0*(kend-kstart)*Rbkg/PI)+1;
    cout <<"N_ctrlP: "<<N_ctrlP<<endl;
    int preEdgeStartEnNo=0, preEdgeEndEnNo=0, postEdgeStartEnNo=0, postEdgeEndEnNo=0;
    int splineStartEnNo=0, splineEndEnNo=0;
    for (int en=0; en<num_evec; en++) {
        preEdgeStartEnNo = (preEdgeStartE<=e_vec[en]) ? preEdgeStartEnNo:en;
        preEdgeEndEnNo = (preEdgeEndE<=e_vec[en]) ? preEdgeEndEnNo:min(en+1,num_evec-1);
        postEdgeStartEnNo = (postEdgeStartE<=e_vec[en]) ? postEdgeStartEnNo:en;
        postEdgeEndEnNo = (postEdgeEndE<=e_vec[en]) ? postEdgeEndEnNo:min(en+1,num_evec-1);
        splineStartEnNo = (splineStartE<=e_vec[en]) ? splineStartEnNo:en;
        splineEndEnNo = (splineEndE<=e_vec[en]) ? splineEndEnNo:min(en+1,num_evec-1);
    }
    cout<< "mt num points: " << e_vec.size()<<endl;
    cout<< "pre-edge start energy No.: "<<preEdgeStartEnNo<<endl;
    cout<< "pre-edge end energy No.: "<<preEdgeEndEnNo<<endl;
    cout<< "post-edge start energy No.: "<<postEdgeStartEnNo<<endl;
    cout<< "post-edge end energy No.: "<<postEdgeEndEnNo<<endl;
    cout<< "spline start energy No.: "<<splineStartEnNo<<endl;
    cout<< "spline end energy No.: "<<splineEndEnNo<<endl;
    
    int startEnergyNo = min(min(preEdgeStartEnNo,postEdgeStartEnNo),splineStartEnNo);
    int endEnergyNo = max(max(preEdgeEndEnNo,postEdgeEndEnNo),splineEndEnNo);
    int num_energy=endEnergyNo-startEnergyNo+1;
    cout << "energy num for EXAFS extracting: "<<num_energy<<endl<<endl;
    
    
    //kernel program source
    ifstream ifs("/Users/ishiguro/Desktop/CT_programs/CT-XANES_analysis/EXAFS_extraction/EXAFSextraction.cl", ios::in);
    if (!ifs) {
        cerr << "   Failed to load kernel \n" << endl;
        return -1;
    }
    istreambuf_iterator<char> it(ifs);
    istreambuf_iterator<char> last;
    string kernel_ext_src(it, last);
    ifs.close();
    //cout<<kernel_code1<<endl;
    
    //OpenCL Program
    vector<vector<cl::Program>> programs;
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        vector<cl::Program> programs_atPlat;
        cl::Program::Sources source;
#if defined (OCL120)
        source.push_back(make_pair(kernel_fit_src.c_str(),kernel_fit_src.length()));
        source.push_back(make_pair(kernel_EXAFSfit_src.c_str(),kernel_EXAFSfit_src.length()));
        source.push_back(make_pair(kernel_3D_FFT_src.c_str(),kernel_3D_FFT_src.length()));
        source.push_back(make_pair(kernel_LM_src.c_str(),kernel_LM_src.length()));
        source.push_back(make_pair(kernel_ext_src.c_str(),kernel_ext_src.length()));
#else
        source.push_back(kernel_fit_src);
        source.push_back(kernel_EXAFSfit_src);
        source.push_back(kernel_3D_FFT_src);
        source.push_back(kernel_LM_src);
        source.push_back(kernel_ext_src);
#endif
        programs_atPlat.push_back(cl::Program(plat_dev_list.context(i), source));//pre-edge用
        programs_atPlat.push_back(cl::Program(plat_dev_list.context(i), source));//postedge用
        programs_atPlat.push_back(cl::Program(plat_dev_list.context(i), source));//spline用
        //kernel build
        string option ="";
#ifdef DEBUG
        option += "-D DEBUG";// -Werror";
#endif
        string GPUvendor =  plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0].getInfo<CL_DEVICE_VENDOR>();
        if(GPUvendor == "nvidia"){
            option += " -cl-nv-maxrregcount=64 -cl-nv-verbose";
        }
        else if (GPUvendor.find("NVIDIA Corporation")==0) {
            option += " -cl-nv-maxrregcount=64";
        }
        ostringstream oss0;
        oss0 << " -D IMAGESIZE_X=" << imageSizeX;
        oss0 << " -D IMAGESIZE_Y=" << processImageSizeY;
        oss0 << " -D IMAGESIZE_M=" << processImageSizeM;
        oss0 << " -D ENERGY_NUM=" << num_energy;
        oss0 << " -D FFT_SIZE=" << FFT_SIZE;
        oss0 << " -D PARA_NUM=" << num_param_preEdge;
        oss0 << " -D PARA_NUM_SQ=" << num_paramsq_preEdge;
        string option0 = option+oss0.str();
        ostringstream oss1;
        oss1 << " -D IMAGESIZE_X=" << imageSizeX;
        oss1 << " -D IMAGESIZE_Y=" << processImageSizeY;
        oss1 << " -D IMAGESIZE_M=" << processImageSizeM;
        oss1 << " -D ENERGY_NUM=" << num_energy;
        oss1 << " -D FFT_SIZE=" << FFT_SIZE;
        oss1 << " -D PARA_NUM=" << 4;
        oss1 << " -D PARA_NUM_SQ=" << 16;
        string option1 = option+oss1.str();
        ostringstream oss2;
        oss1 << " -D IMAGESIZE_X=" << imageSizeX;
        oss1 << " -D IMAGESIZE_Y=" << processImageSizeY;
        oss1 << " -D IMAGESIZE_M=" << processImageSizeM;
        oss1 << " -D ENERGY_NUM=" << num_energy;
        oss2 << " -D FFT_SIZE=" << FFT_SIZE;
        oss2 << " -D PARA_NUM=" << N_ctrlP;
        oss2 << " -D PARA_NUM_SQ=" << N_ctrlP*N_ctrlP;
        string option2 = option + oss2.str();
        programs_atPlat[0].build(option0.c_str());
        programs_atPlat[1].build(option1.c_str());
        programs_atPlat[2].build(option2.c_str());
#ifdef DEBUG
        string logstr=programs_atPlat[2].getBuildInfo<CL_PROGRAM_BUILD_LOG>(plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0]);
        cout << logstr << endl;
#endif
        programs.push_back(programs_atPlat);
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
    vector<cl::Buffer> mt_imgs;
    vector<cl::Buffer> w_factors;
    vector<cl::Buffer> bkg_imgs;
    vector<cl::Buffer> ej_imgs;
    vector<cl::Buffer> chi_imgs;
    for (int i=0; i<plat_dev_list.contextsize(); i++){
        //energy buffers
        energy_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*num_energy, 0, NULL));
        plat_dev_list.queue(i,0).enqueueWriteBuffer(energy_buffers[i], CL_TRUE, 0, sizeof(cl_float)*num_energy, &e_vec[startEnergyNo], NULL, NULL);
        //mt imgs
        mt_imgs.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*num_energy, 0, NULL));
        plat_dev_list.queue(i,0).enqueueWriteBuffer(mt_imgs[i], CL_TRUE, 0, sizeof(cl_float)*num_energy, &mt_vec[startEnergyNo], NULL, NULL);
        //bkg & ej imgs
        bkg_imgs.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float), 0, NULL));
        ej_imgs.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float), 0, NULL));
        //chi imgs
        chi_imgs.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float)*MAX_KRSIZE, 0, NULL));
        
        //spinfactors
        w_factors.push_back(cl::Buffer(plat_dev_list.context(i),CL_MEM_READ_WRITE,sizeof(cl_float2)*FFT_SIZE/2, 0, NULL));
        createSpinFactor(w_factors[i], plat_dev_list.queue(i,0), programs[i][2]);
        
        //pre-edge estimation
        PreEdgeRemoval(plat_dev_list.queue(i,0), programs[i][0], mt_imgs[i], energy_buffers[i],bkg_imgs[i], imageSizeX, processImageSizeY, E0, preEdgeStartEnNo, preEdgeEndEnNo, preEdgeFittingMode,0.01f,30);
        
        float bkg;
        plat_dev_list.queue(i,0).enqueueReadBuffer(bkg_imgs[i], CL_TRUE, 0, sizeof(cl_float), &bkg);
        cout <<"bkg: "<<bkg<<endl;
        
        //post-edge estimation 
        PostEdgeEstimation(plat_dev_list.queue(i,0),programs[i][1], mt_imgs[i], energy_buffers[i],bkg_imgs[i], ej_imgs[i], imageSizeX, processImageSizeY,postEdgeStartEnNo, postEdgeEndEnNo,0.01f,30);
        
        float ej;
        plat_dev_list.queue(i,0).enqueueReadBuffer(ej_imgs[i], CL_TRUE, 0, sizeof(cl_float), &ej);
        cout <<"ej: "<<ej<<endl;
        
        SplineBkgRemoval(plat_dev_list.queue(i,0),programs[i][2], mt_imgs[i], energy_buffers[i],w_factors[i], chi_imgs[i], imageSizeX, processImageSizeY, processImageSizeY, num_energy,kstart, kend, Rbkg, kw, 0.01f, 30, true);
        float* chidata;
        chidata=new float[MAX_KRSIZE];
        plat_dev_list.queue(i,0).enqueueReadBuffer(chi_imgs[i], CL_TRUE, 0, sizeof(cl_float)*MAX_KRSIZE, chidata);
        cout <<"chi"<<endl;
        for (int j=0; j<MAX_KRSIZE; j++) {
            cout<<j*0.05f<<"\t"<<chidata[j]<<endl;
        }
    }
    
    
    
    return 0;
}
