//
//  testEXAFSfit.cpp
//  CT-XAFS_analysis
//
//  Created by Nozomu Ishiguro on 2017/08/12.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#include "EXAFS.hpp"



int testEXAFS(vector<FEFF_shell> shells, OCL_platform_device plat_dev_list) {
    try {
        //kernel program source
        ifstream ifs("/Users/ishiguro/Desktop/CT_programs/CT-XANES_analysis/EXAFS_fitting/EXAFS_fit.cl", ios::in);
        if (!ifs) {
            cerr << "   Failed to load kernel \n" << endl;
            return -1;
        }
        istreambuf_iterator<char> it(ifs);
        istreambuf_iterator<char> last;
        string kernel_code1(it, last);
        ifs.close();
        //cout<<kernel_code1<<endl;
        
        //kernel program source
        ifstream ifs2("/Users/ishiguro/Desktop/CT_programs/CT-XANES_analysis/EXAFS_fitting/3D_FFT.cl", ios::in);
        if (!ifs2) {
            cerr << "   Failed to load kernel \n" << endl;
            return -1;
        }
        istreambuf_iterator<char> it2(ifs2);
        istreambuf_iterator<char> last2;
        string kernel_code2(it2, last2);
        ifs2.close();
        
        ifstream ifs3("/Users/ishiguro/Desktop/CT_programs/CT-XANES_analysis/share\ source/LevenbergMarquardt.cl", ios::in);
        if (!ifs3) {
            cerr << "   Failed to load kernel \n" << endl;
            return -1;
        }
        istreambuf_iterator<char> it3(ifs3);
        istreambuf_iterator<char> last3;
        string kernel_code3(it3, last3);
        ifs3.close();
        
        //cout << kernel_code<<endl;
        vector<cl::Program> programs;
        cl_int ret;
        for (int i = 0; i<plat_dev_list.contextsize(); i++) {
#if defined (OCL120)
            //cl::Program::Sources source(1,std::make_pair(kernel_code1.c_str(),kernel_code1.length()));
            cl::Program::Sources source;
            source.push_back(std::make_pair(kernel_code1.c_str(), kernel_code1.length()));
            source.push_back(std::make_pair(kernel_code2.c_str(), kernel_code2.length()));
            source.push_back(std::make_pair(kernel_code3.c_str(), kernel_code3.length()));
#else
            cl::Program::Sources source;
            source.push_back(kernel_code1);
            source.push_back(kernel_code2);
            source.push_back(kernel_code3);
#endif
            programs.push_back(cl::Program(plat_dev_list.context(i), source, &ret));
            //kernel build
            ostringstream oss;
            oss << "-D FFT_SIZE=" << FFT_SIZE << " ";
            oss << "-D PARA_NUM=" << 4 << " ";
            oss << "-D PARA_NUM_SQ=" << 16 << " ";
            string option = oss.str();
            option += "-cl-fp32-correctly-rounded-divide-sqrt -cl-single-precision-constant ";
#ifdef DEBUG
            option += "-D DEBUG ";//-Werror";
#endif
            string GPUvendor = plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0].getInfo<CL_DEVICE_VENDOR>();
            if (GPUvendor == "nvidia") {
                option += " -cl-nv-maxrregcount=64 -cl-nv-verbose";
            }
            else if (GPUvendor.find("NVIDIA Corporation") == 0) {
                option += " -cl-nv-maxrregcount=64";
            }
            ret = programs[i].build(option.c_str());
#ifdef DEBUG
            string logstr = programs[i].getBuildInfo<CL_PROGRAM_BUILD_LOG>(plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0]);
            cout << logstr << endl;
#endif
            
        }
        
        
        string inputpath = "/Users/ishiguro/Documents/実験/Fuel_Cell/FC_MEA(0.5mg_cm-2)_13PCCP/chi(igor)/Pt_foil.chi";
        vector<float*> chi_pointers;
        for (int j = 0; j<4; j++) {
            ifstream chi_ifs(inputpath, ios::in);
            vector<float> k_vec, chi_vec;
            int npnts = 0;
            do {
                string a, b;
                chi_ifs >> a >> b;
                
                if (chi_ifs.eof()) break;
                float aa, bb;
                try {
                    aa = stof(a);
                    bb = stof(b);
                }
                catch (invalid_argument ret) { //ヘッダータグが存在する場合に入力エラーになる際への対応
                    continue;
                }
                k_vec.push_back(aa);
                chi_vec.push_back(bb);
                //cout<<npnts+1<<": "<<aa<<","<<bb<<endl;
                npnts++;
            } while (!chi_ifs.eof());
            chi_ifs.close();
            //cout<<endl;
            
            chi_pointers.push_back(new float[npnts]);
            for (int i = 0; i<npnts; i++) {
                chi_pointers[j][i] = chi_vec[i];//*(j+1.0f);
                //cout << chi_pointers[0][i] <<endl;
            }
        }
        float* ej;
        ej = new float[4];
        ej[0] = 1.0f;
        ej[1] = 2.0f;
        ej[2] = 3.0f;
        ej[3] = 4.0f;
        
        
        
        vector<vector<shellObjects>> ShObj;
        vector<cl::Buffer> chiData_buff;
        vector<cl::Buffer> FTchiData_buff;
        vector<cl::Buffer> chiqData_buff;
        vector<cl::Buffer> S02;
        vector<cl::Buffer> ej_buff;
        float kstart = 3.0f;
        float kend = 14.0f;
        float Rstart = 0.0f;
        float Rend = 3.0f;
        int kw = 3;
        float qstart = 3.0f;
        float qend = 14.0f;
        int ksize = ceil((min(float(kend + WIN_DK), (float)MAX_KQ) - max(float(kstart - WIN_DK), 0.0f)) / KGRID);
        int koffset = floor(max((float)(kstart - WIN_DK), 0.0f) / KGRID);
        int Rsize = ceil((min(float(Rend + WIN_DR), (float)MAX_R) - max(float(Rstart - WIN_DR), 0.0f)) / RGRID);
        int Roffset = floor(max((float)(Rstart - WIN_DR), 0.0f) / RGRID);
        int qsize = ceil((min(float(qend + WIN_DK), (float)MAX_KQ) - max(float(qstart - WIN_DK), 0.0f)) / KGRID);
        int qoffset = floor(max((float)(qstart - WIN_DK), 0.0f) / KGRID);
        int imagesizeX = 4;
        for (int i = 0; i<plat_dev_list.contextsize(); i++) {
            vector<shellObjects> ShObj_atP;
            for (int j = 0; j<shells.size(); j++) {
                ShObj_atP.push_back(shellObjects(plat_dev_list.queue(i, 0), programs[i], shells[j], imagesizeX, 1));
            }
            ShObj.push_back(ShObj_atP);
            
            
            
            chiData_buff.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float2)*ksize*imagesizeX, 0, NULL));
            FTchiData_buff.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float2)*Rsize*imagesizeX, 0, NULL));
            chiqData_buff.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float2)*qsize*imagesizeX, 0, NULL));
            S02.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float)*imagesizeX, 0, NULL));
            ej_buff.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float)*imagesizeX, 0, NULL));
            cl::Buffer w_factor(plat_dev_list.context(i), CL_MEM_READ_WRITE, sizeof(cl_float2)*FFT_SIZE / 2, 0, NULL);
            createSpinFactor(w_factor, plat_dev_list.queue(i, 0), programs[i]);
            plat_dev_list.queue(i, 0).enqueueFillBuffer(S02[i], (cl_float)1.0f, 0, sizeof(cl_float)*imagesizeX);
            
            vector<int> processImageSizeY = GPUmemoryControl(2048, 2048, ksize, Rsize, qsize, 2, 4, ShObj[i], plat_dev_list.queue(i, 0));
            
            plat_dev_list.queue(i, 0).enqueueWriteBuffer(ej_buff[i], CL_TRUE, 0, sizeof(cl_float)*imagesizeX, ej);
            //ShObj[i][0].inputIniCN(12.0f, ej_buff[i]);
            
            //chi data
            //ChiData_k(plat_dev_list.queue(i,0),programs[i],chiData_buff[i],chi_pointers,kw, kstart, kend, imagesizeX, 1, false,0);
            ChiData_R(plat_dev_list.queue(i, 0), programs[i], FTchiData_buff[i], chi_pointers, w_factor, kw, kstart, kend, Rstart, Rend, imagesizeX, 1, 1, false, 0);
            //ChiData_q(plat_dev_list.queue(i,0),programs[i],chiqData_buff[i],chi_pointers,w_factor,kw, kstart, kend, Rstart, Rend, qstart, qend, imagesizeX, 1, 1, false, 0);
            
            
            //EXAFS fit
            //EXAFS_kFit(plat_dev_list.queue(i,0),programs[i],chiData_buff[i],S02[i], ShObj[i],kw, kstart, kend, imagesizeX, 1, false, 30, 0.1f);
            EXAFS_RFit(plat_dev_list.queue(i, 0), programs[i], w_factor, FTchiData_buff[i], S02[i], ShObj[i], kw, kstart, kend, Rstart, Rend, imagesizeX, 1, 1, false, 30, 0.1f);
            //EXAFS_qFit(plat_dev_list.queue(i,0),programs[i], w_factor,chiqData_buff[i], S02[i],ShObj[i],kw, kstart, kend, Rstart, Rend, qstart, qend, imagesizeX, 1, 1,false,30,0.1f);
            
            
            float* CN;
            float* Rval;
            float* dE0;
            float* ss;
            CN = new float[imagesizeX];
            Rval = new float[imagesizeX];
            dE0 = new float[imagesizeX];
            ss = new float[imagesizeX];
            ShObj[i][0].readParaImage(CN, 1);
            ShObj[i][0].readParaImage(Rval, 2);
            ShObj[i][0].readParaImage(dE0, 3);
            ShObj[i][0].readParaImage(ss, 4);
            for (int j = 0; j<imagesizeX; j++) {
                cout << CN[j] << "\t" << Rval[j] << "\t" << dE0[j] << "\t" << ss[j] << endl;
            }
            
            /*cl_float2* w_data;
             w_data = new cl_float2[FFT_SIZE];
             plat_dev_list.queue(i,0).enqueueReadBuffer(w_factor, CL_TRUE, 0, sizeof(cl_float2)*FFT_SIZE, w_data);
             for (int j=0; j<FFT_SIZE; j++) {
             cout << w_data[j].x << "\t" << w_data[j].y <<endl;
             }
             delete [] w_data;*/
            
            /*cl_float2* chi_data;
             chi_data = new cl_float2[MAX_KRSIZE];
             for (int j=0; j<MAX_KRSIZE; j++) {
             chi_data[j].x = 0.0f;
             chi_data[j].y = 0.0f;
             }
             plat_dev_list.queue(i,0).enqueueReadBuffer(chiData_buff[i], CL_TRUE, 0, sizeof(cl_float2)*ksize, &chi_data[koffset]);
             for (int j=0; j<MAX_KRSIZE; j++) {
             cout << chi_data[j].y <<endl;
             }
             delete [] chi_data;*/
            
            /*cl_float2* FTchi_data;
             FTchi_data = new cl_float2[MAX_KRSIZE];
             for (int j=0; j<MAX_KRSIZE; j++) {
             FTchi_data[j].x = 0.0f;
             FTchi_data[j].y = 0.0f;
             }
             plat_dev_list.queue(i,0).enqueueReadBuffer(FTchiData_buff[i], CL_TRUE, 0, sizeof(cl_float2)*Rsize, &FTchi_data[Roffset]);
             for (int j=0; j<MAX_KRSIZE; j++) {
             cout << FTchi_data[j].y<< "\t"<< -FTchi_data[j].x <<endl;
             }
             delete [] FTchi_data;*/
        }
        
    }
    catch (const cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
    
    return 0;
}

int main2(int argc, const char * argv[]) {
    string fp_str;
    if (argc>1) {
        fp_str=argv[1];
    }else{
        /*string dummy;
         cout<<"Set input file path, if existed."<<endl;
         getline(cin,dummy);
         istringstream iss(dummy);
         iss>>fp_str;*/
        fp_str="/Users/ishiguro/Desktop/XAFS_tools/feff_inp/Pt_feff/feff0001.dat";
    }
    
    OCL_platform_device plat_dev_list("2"/*inp.getPlatDevList()*/,false);
    
    vector<FEFF_shell> shell;
    shell.push_back(FEFF_shell::FEFF_shell(fp_str));
    testEXAFS(shell,plat_dev_list);
    
    return 0;
}
