//
//  Regmode_class.cpp
//  Image registration share
//
//  Created by Nozomu Ishiguro on 2015/05/28.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"
#include "imageReg.hpp"
#include "imageregistration_kernel_src_cl.hpp"
#include "LevenbergMarquardt_cl.hpp"


regMode::regMode(int regmodeNumber,int cntFitmode){
    regModeNo=regmodeNumber;
    
    switch (regModeNo) {
        case 0: //xy shift
            regModeName="xy shift";
            oss_target  ="    registration shift: dx=0, dy=0\n\n";
            p_num=2;
            break;
            
        case 1: //rotation+xy shift
            regModeName="rotation + xy shift";
            oss_target  ="    registration shift: dx=0, dy=0, d(theta)=0\n\n";
            p_num=3;
            break;
        
        case 2: //scale+xy shift
            regModeName="scale + xy shift";
            oss_target  ="    registration shift: dx=0, dy=0, scale(log)=0.0\n\n";
            p_num=3;
            break;
        
        case 3: //scale+rotation+xy shift
            regModeName="scale + rotation + xy shift";
            oss_target  ="    registration shift: dx=0, dy=0, d(theta)=0, scale(log)=0.0\n\n";
            p_num=4;
            break;
        
        case 4: //affine+xy shift
            regModeName="affine + xy shift";
            oss_target  ="    registration shift: dx=0, dy=0, a11=1.0, a21=0, a12=0, a22=1.0\n\n";
            p_num=6;
            break;
            
        default:
            regModeName="none";
            oss_target="";
            p_num=1;
            break;
    }
    
    diffstep = 5;
    p_ini = new float[p_num+2];
    p_fix = new char[p_num+2];
    for (int i=0; i<p_num; i++) {
        p_ini[i]=0.0f;
        p_fix[i]=49;
    }
    p_ini[p_num]    = 0.0f; //cnt mult
    p_ini[p_num+1]  = 0.0f; //cnt offset
    
    
    
    switch (cntFitmode) {
        case 0:
            p_fix[p_num]    = 48;   //cnt mult
            p_fix[p_num+1]  = 48;   //cnt offset
            displayCnt=false;
            break;
            
        case 1:
            p_fix[p_num]    = 49;   //cnt mult
            p_fix[p_num+1]  = 49;   //cnt offset
            displayCnt=true;
            break;
            
        case 2:
            p_fix[p_num]    = 48;   //cnt mult
            p_fix[p_num+1]  = 49;   //cnt offset
            displayCnt=true;
            break;
        
        case 3:
            p_fix[p_num]    = 49;   //cnt mult
            p_fix[p_num+1]  = 48;   //cnt offset
            displayCnt=true;
            break;
            
        default:
            p_fix[p_num]    = 48;   //cnt mult
            p_fix[p_num+1]  = 49;   //cnt offset
            displayCnt=true;
            break;
    }
}

int regMode::get_regModeNo(){
    return regModeNo;
}

string regMode::ofs_transpara(){
    ostringstream oss;
    switch (regModeNo) {
        case 0: //xy shift
            oss<< "dx\tdx error\tdy\tdy error";
            break;
            
        case 1: //rotation+xy shift
            oss<<"dx\tdx error\tdy\tdy error\td(theta)\td(theta) error";
            break;
            
        case 2: //scale+xy shift
            oss<<"dx\tdx error\tdy\tdy error\tscale(log)\tscale(log) error";
            break;
            
        case 3: //scale+rotation+xy shift
            oss<<"dx\tdx error\tdy\tdy error\t";
            oss<<"d(theta)\td(theta) error\t";
            oss<<"scale(log)\tscale(log) error";
            break;
            
        case 4: //affine+xy shift
            oss<<"dx\tdx error\tdy\tdy error\t";
            oss<<"a11\ta11 error\ta12\ta12 error\t";
            oss<<"a21\ta21 error\ta22\ta22 error";
            break;
            
        default: //none
            oss<<"" << endl;
            break;
    }
    if (displayCnt) {
        oss<<"cLnMult\tcLnMult error\tcBkg\tcBkg error\t"<<endl;
    }else{
        oss<<endl;
    }
    return oss.str();
}

string regMode::get_regModeName(){
    return regModeName;
}


string regMode::get_oss_target(){
    ostringstream oss;
    switch (regModeNo) {
        case 0: //xy shift
            oss<<"    registration shift: dx=";
            oss<<p_ini[0]<<", dy=";
            oss<<p_ini[1]<<endl;
            break;
            
        case 1: //rotation+xy shift
            oss<<"    registration shift: dx=";
            oss<<p_ini[0]<<", dy=";
            oss<<p_ini[1]<<", d(theta)=";
            oss<<p_ini[2]<<endl;
            break;
            
        case 2: //scale+xy shift
            oss<<"    registration shift: dx=";
            oss<<p_ini[0]<<", dy=";
            oss<<p_ini[1]<<", scale(log)=";
            oss<<p_ini[2]<<endl;
            break;
            
        case 3: //scale+rotation+xy shift
            oss<<"    registration shift: dx=";
            oss<<p_ini[0]<<", dy=";
            oss<<p_ini[1]<<", d(theta)=";
            oss<<p_ini[2]<<", scale(log)=";
            oss<<p_ini[3]<<endl;
            break;
            
        case 4: //affine+xy shift
            oss<<"    registration shift: dx=";
            oss<<p_ini[0]<<", dy=";
            oss<<p_ini[1]+1<<", a11=";
            oss<<p_ini[2]<<", a12=";
            oss<<p_ini[3]<<", a21=";
            oss<<p_ini[4]<<", a22=";
            oss<<p_ini[5]+1<<endl;
            break;
            
        default:
            oss<<"";
            break;
    }
    oss<<endl;
    
    return oss.str();
    
    //return oss_target;
}


string regMode::get_oss_target(float *p_vec){
    ostringstream oss;
    switch (regModeNo) {
        case 0: //xy shift
            oss<<"    registration shift: dx=";
            oss<<p_vec[0]<<", dy=";
            oss<<p_vec[1]<<endl;
            break;
            
        case 1: //rotation+xy shift
            oss<<"    registration shift: dx=";
            oss<<p_vec[0]<<", dy=";
            oss<<p_vec[1]<<", d(theta)=";
            oss<<p_vec[2]<<endl;
            break;
            
        case 2: //scale+xy shift
            oss<<"    registration shift: dx=";
            oss<<p_vec[0]<<", dy=";
            oss<<p_vec[1]<<", scale(log)=";
            oss<<p_vec[2]<<endl;
            break;
            
        case 3: //scale+rotation+xy shift
            oss<<"    registration shift: dx=";
            oss<<p_vec[0]<<", dy=";
            oss<<p_vec[1]<<", d(theta)=";
            oss<<p_vec[2]<<", scale(log)=";
            oss<<p_vec[3]<<endl;
            break;
            
        case 4: //affine+xy shift
            oss<<"    registration shift: dx=";
            oss<<p_vec[0]<<", dy=";
            oss<<p_vec[1]+1<<", a11=";
            oss<<p_vec[2]<<", a12=";
            oss<<p_vec[3]<<", a21=";
            oss<<p_vec[4]<<", a22=";
            oss<<p_vec[5]+1<<endl;
            break;
            
        default:
            oss<<"";
            break;
    }
    oss<<endl;
    return oss.str();
    
    //return oss_target;
}

void regMode::set_pfix(input_parameter inp) {

	for (int i = 0; i < p_num; i++) {
		p_fix[i] = inp.getReg_fixpara()[i];
	}
}

string regMode::oss_sample(float *p,float *p_error,
                           int *p_precision, int *p_err_precision){
    ostringstream oss;
    switch (regModeNo) {
        case 0: //xy shift
            oss<<"    registration shift: dx=";
            oss.precision(p_precision[0]);
            oss<<p[0];
            if(p_fix[0]==49){
                oss<<" +/- ";
                oss.precision(1);
                oss<<abs(p_error[0]);
            }
            oss<<", dy=";
            oss.precision(p_precision[1]);
            oss<<p[1];
            if(p_fix[1]==49){
                oss<<" +/- ";
                oss.precision(1);
                oss<<abs(p_error[1]);
            }
            break;
        
        case 1: //rotation+xy shift
            oss<<"    registration shift: dx=";
            oss.precision(p_precision[0]);
            oss<<p[0];
            if(p_fix[0]==49){
                oss<<" +/- ";
                oss.precision(1);
                oss<<abs(p_error[0]);
            }
            oss<<", dy=";
            oss.precision(p_precision[1]);
            oss<<p[1];
            if(p_fix[1]==49){
                oss<<" +/- ";
                oss.precision(1);
                oss<<abs(p_error[1]);
            }
            oss<<", d(theta)=";
            oss.precision(p_precision[2]);
            oss<<p[2];
            if(p_fix[2]==49){
                oss<<" +/- ";
                oss.precision(1);
                oss<<abs(p_error[2]);
            }
            break;
        
        case 2: //scale+xy shift
            oss<<"    registration shift: dx=";
            oss.precision(p_precision[0]);
            oss<<p[0];
            if(p_fix[0]==49){
                oss<<" +/- ";
                oss.precision(1);
                oss<<abs(p_error[0]);
            }
            oss<<", dy=";
            oss.precision(p_precision[1]);
            oss<<p[1];
            if(p_fix[1]==49){
                oss<<" +/- ";
                oss.precision(1);
                oss<<abs(p_error[1]);
            }
            oss<<", scale(log)=";
            oss.precision(p_precision[2]);
            oss<<p[2];
            if(p_fix[2]==49){
                oss<<" +/- ";
                oss.precision(1);
                oss<<abs(p_error[2]);
            }
            break;
        
        case 3: //scale+rotation+xy shift
            oss<<"    registration shift: dx=";
            oss.precision(p_precision[0]);
            oss<<p[0];
            if(p_fix[0]==49){
                oss<<" +/- ";
                oss.precision(1);
                oss<<abs(p_error[0]);
            }
            oss<<", dy=";
            oss.precision(p_precision[1]);
            oss<<p[1];
            if(p_fix[1]==49){
                oss<<" +/- ";
                oss.precision(1);
                oss<<abs(p_error[1]);
            }
            oss<<", d(theta)=";
            oss.precision(p_precision[2]);
            oss<<p[2];
            if(p_fix[2]==49){
                oss<<" +/- ";
                oss.precision(1);
                oss<<abs(p_error[2]);
            }
            oss<<", scale(log)=";
            oss.precision(p_precision[3]);
            oss<<p[3];
            if(p_fix[3]==49){
                oss<<" +/- ";
                oss.precision(1);
                oss<<abs(p_error[3]);
            }
            break;
        
        case 4: //affine+xy shift
            oss<<"    registration shift: dx=";
            oss.precision(p_precision[0]);
            oss<<p[0];
            if(p_fix[0]==49){
                oss<<" +/- ";
                oss.precision(1);
                oss<<abs(p_error[0]);
            }
            oss<<", dy=";
            oss.precision(p_precision[1]);
            oss<<p[1];
            if(p_fix[1]==49){
                oss<<" +/- ";
                oss.precision(1);
                oss<<abs(p_error[1]);
            }
            oss<<", a11=";
            oss.precision(p_precision[2]);
            oss<<p[2]+1;
            if(p_fix[2]==49){
                oss<<" +/- ";
                oss.precision(1);
                oss<<abs(p_error[2]);
            }
            oss<<", a12=";
            oss.precision(p_precision[3]);
            oss<<p[3];
            if(p_fix[3]==49){
                oss<<" +/- ";
                oss.precision(1);
                oss<<abs(p_error[3]);
            }
            oss<<", a21=";
            oss.precision(p_precision[4]);
            oss<<p[4];
            if(p_fix[4]==49){
                oss<<" +/- ";
                oss.precision(1);
                oss<<abs(p_error[4]);
            }
            oss<<", a22=";
            oss.precision(p_precision[5]);
            oss<<p[5]+1;
            if(p_fix[5]==49){
                oss<<" +/- ";
                oss.precision(1);
                oss<<abs(p_error[5]);
            }
            break;
        
        default:
            oss<<"";
            break;
    }
    if (displayCnt) {
        oss<<", cLnMult=";
        oss.precision(p_precision[p_num]);
        oss<<p[p_num];
        if(p_fix[p_num]==49){
            oss<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[p_num]);
        }oss<<", cBkg=";
        oss.precision(p_precision[p_num+1]);
        oss<<p[p_num+1];
        if(p_fix[p_num+1]==49){
            oss<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[p_num+1]);
        }
        oss<<endl;
    }else{
        oss<<endl;
    }
    
    
    oss<<endl;
    return oss.str();
}

int regMode::get_p_num(){
    return p_num;
}


cl::Program regMode::buildImageRegProgram(cl::Context context, int imageSizeX, int imageSizeY){
    
    
    //kernel source from cl file
    /*ifstream ifs("/Users/ishiguro/Desktop/CT_programs/CT-XANES_analysis/ImageReg share source/imageregistration_kernel_src.cl", ios::in);
    if(!ifs) {
        cerr << "   Failed to load kernel" << endl;
    }
    istreambuf_iterator<char> it(ifs);
    istreambuf_iterator<char> last;
    string kernel_src(it,last);
    ifs.close();*/
    //cout<<kernel_src<<endl;
    
    
    string option="-cl-fp32-correctly-rounded-divide-sqrt -cl-single-precision-constant ";
#if DEBUG
    option+="-D DEBUG ";
#endif
    string GPUvendor = context.getInfo<CL_CONTEXT_DEVICES>()[0].getInfo<CL_DEVICE_VENDOR>();
    if(GPUvendor == "nvidia"){
        option += "-cl-nv-maxrregcount=64 ";
        //option += " -cl-nv-verbose -Werror";
    }else if (GPUvendor.find("NVIDIA")==0) {
		option += "-cl-nv-maxrregcount=64 ";
		//option += " -cl-nv-verbose -Werror";
	}
	option += "-D DIFFSTEP=1 ";
    switch (regModeNo) {
        case 0: //XY
            option += "-D REGMODE=0 -D PARA_NUM=4 -D PARA_NUM_SQ=16 -D REG_PARA_NUM=2";
            break;
            
        case 1: //XY+rotation
            option += "-D REGMODE=1 -D PARA_NUM=5 -D PARA_NUM_SQ=25 -D REG_PARA_NUM=3";
            break;
            
        case 2: //XY+scale
            option += "-D REGMODE=2 -D PARA_NUM=5 -D PARA_NUM_SQ=25 -D REG_PARA_NUM=3";
            break;
            
        case 3: //XY+rotation+scale
            option += "-D REGMODE=3 -D PARA_NUM=6 -D PARA_NUM_SQ=36 -D REG_PARA_NUM=4";
            break;
            
        case 4: //XY+affine
            option += "-D REGMODE=4 -D PARA_NUM=8 -D PARA_NUM_SQ=64 -D REG_PARA_NUM=6";
            break;
            
        default:
            break;
    }
    
    ostringstream oss;
    oss << " -D IMAGESIZE_X="<<imageSizeX;
    oss << " -D IMAGESIZE_Y="<<imageSizeY;
    oss << " -D IMAGESIZE_M="<<imageSizeX*imageSizeY;
    option += oss.str();
    
    cl::Program::Sources source;
#if defined (OCL120)
    source.push_back(std::make_pair(kernel_src.c_str(),kernel_src.length()));
    source.push_back(std::make_pair(kernel_LM_src.c_str(),kernel_LM_src.length()));
#else
    source.push_back(kernel_src);
    source.push_back(kernel_LM_src);
#endif
    cl::Program program(context, source);
    //cout << kernel_code<<endl;
    //kernel build
    program.build(option.c_str());
	string logstr=program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]);
	cout << logstr << endl;
    return program;
}
