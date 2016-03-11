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


regMode::regMode(int regmodeNumber,int cntmodeNumber){
    regModeNo=regmodeNumber;
    cntModeNo=cntmodeNumber;
    
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
    
    //cout<<cntModeNo<<endl;
    switch (cntModeNo) {
        case 0: //no contrast factor
            cp_num=0;
            break;
        case 1: //contrast(exp(g)) + bkg(const)
            if(p_num>1) {
                oss_target +="    contrast factor   : contrast(log)=0, darkness=0\n\n";
                cp_num=2;
            }
            break;
        case 2: //contrast(exp(g)) + bkg(s*x+t*y+const: 1st order plane)
            if(p_num>1) {
                oss_target +="    contrast factor   : contrast(log)=0, darkness=0, drk_x=0, drk_y=0\n\n";
                cp_num=4;
            }
            break;
        default:
            cp_num=0;
            break;
    }
    p_ini = new float[p_num+cp_num];
    for (int i=0; i<p_num+cp_num; i++) {
        p_ini[i]=0.0f;
    }
    //cout<<p_num<<endl;
}

int regMode::get_regModeNo(){
    return regModeNo;
}

int regMode::get_contrastModeNo(){
    return cntModeNo;
}

string regMode::ofs_transpara(){
    ostringstream oss;
    switch (regModeNo) {
        case 0: //xy shift
            oss<< "dx\tdx error\tdy\tdy error\t";
            break;
            
        case 1: //rotation+xy shift
            oss<<"dx\tdx error\tdy\tdy error\td(theta)\td(theta) error\t";
            break;
            
        case 2: //scale+xy shift
            oss<<"dx\tdx error\tdy\tdy error\tscale(log)\tscale(log) error\t";
            break;
            
        case 3: //scale+rotation+xy shift
            oss<<"dx\tdx error\tdy\tdy error\t";
            oss<<"d(theta)\td(theta) error\t";
            oss<<"scale(log)\tscale(log) error\t";
            break;
            
        case 4: //affine+xy shift
            oss<<"dx\tdx error\tdy\tdy error\t";
            oss<<"a11\ta11 error\ta12\ta12 error\t";
            oss<<"a21\ta21 error\ta22\ta22 error\t";
            break;
            
        default: //none
            oss<<"";
            break;
    }
    
    switch (cntModeNo) {
        case 0: //no contrast factor
            oss<<endl;
            break;
        case 1: //contrast(exp(g)) + bkg(const)
            oss<< "contrast(log)\tcontrast(log) error\tdarkness\tdarkness error"<<endl;
            break;
        case 2: //contrast(exp(g)) + bkg(s*x+t*y+const: 1st order plane)
            oss<< "contrast(log)\tcontrast(log) error\tdarkness\tdarkness error\t";
            oss<< "drk_x\tdrk_x error\tdrk_y\tdrk_y error"<<endl;
            break;
        default:
            oss<<endl;
            break;
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
    
    switch (cntModeNo) {
        case 0: //no contrast factor
            break;
        case 1: //contrast(exp(g)) + bkg(const)
            
            oss<<"    contrast factor: contrast(log)=";
            oss<<p_ini[p_num]<<", darkness=";
            oss<<p_ini[p_num+1]<<endl;
            break;
        case 2: //contrast(exp(g)) + bkg(s*x+t*y+const: 1st order plane)
            oss<<"    contrast factor: contrast(log)=";
            oss<<p_ini[p_num]<<", darkness=";
            oss<<p_ini[p_num+1]<<", drk_x=";
            oss<<p_ini[p_num+2]<<", drk_y=";
            oss<<p_ini[p_num+3]<<endl;
            break;
        default:
            oss<<endl;
            break;
    }
    oss<<endl;
    return oss.str();
    
    //return oss_target;
}
string regMode::oss_sample(float *p,float *p_error,
                           int *p_precision, int *p_err_precision){
    ostringstream oss;
    switch (regModeNo) {
        case 0: //xy shift
            oss<<"    registration shift: dx=";
            oss.precision(p_precision[0]);
            oss<<p[0]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[0])<<", dy=";
            oss.precision(p_precision[1]);
            oss<<p[1]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[1])<<endl;
            break;
        
        case 1: //rotation+xy shift
            oss<<"    registration shift: dx=";
            oss.precision(p_precision[0]);
            oss<<p[0]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[0])<<", dy=";
            oss.precision(p_precision[1]);
            oss<<p[1]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[1])<<", d(theta)=";
            oss.precision(p_precision[2]);
            oss<<p[2]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[2])<<endl;
            break;
        
        case 2: //scale+xy shift
            oss<<"    registration shift: dx=";
            oss.precision(p_precision[0]);
            oss<<p[0]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[0])<<", dy=";
            oss.precision(p_precision[1]);
            oss<<p[1]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[1])<<", scale(log)=";
            oss.precision(p_precision[2]);
            oss<<p[2]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[2])<<endl;
            break;
        
        case 3: //scale+rotation+xy shift
            oss<<"    registration shift: dx=";
            oss.precision(p_precision[0]);
            oss<<p[0]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[0])<<", dy=";
            oss.precision(p_precision[1]);
            oss<<p[1]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[1])<<", d(theta)=";
            oss.precision(p_precision[2]);
            oss<<p[2]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[2])<<", scale(log)=";
            oss.precision(p_precision[3]);
            oss<<p[3]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[3])<<endl;
            break;
        
        case 4: //affine+xy shift
            oss<<"    registration shift: dx=";
            oss.precision(p_precision[0]);
            oss<<p[0]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[0])<<", dy=";
            oss.precision(p_precision[1]);
            oss<<p[1]+1<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[1])<<", a11=";
            oss.precision(p_precision[2]);
            oss<<p[2]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[2])<<", a12=";
            oss.precision(p_precision[3]);
            oss<<p[3]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[3])<<", a21=";
            oss.precision(p_precision[4]);
            oss<<p[4]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[4])<<", a22=";
            oss.precision(p_precision[5]);
            oss<<p[5]+1<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[5])<<endl;
            break;
        
        default:
            oss<<"";
            break;
    }
    
    switch (cntModeNo) {
        case 0: //no contrast factor
            break;
        case 1: //contrast(exp(g)) + bkg(const)
            
            oss<<"    contrast factor: contrast(log)=";
            oss.precision(p_precision[p_num]);
            oss<<p[p_num]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[p_num])<<", darkness=";
            oss.precision(p_precision[p_num+1]);
            oss<<p[p_num+1]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[p_num+1])<<endl;
            break;
        case 2: //contrast(exp(g)) + bkg(s*x+t*y+const: 1st order plane)
            oss<<"    contrast factor: contrast(log)=";
            oss.precision(p_precision[p_num]);
            oss<<p[p_num]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[p_num])<<", darkness=";
            oss.precision(p_precision[p_num+1]);
            oss<<p[p_num+1]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[p_num+1])<<", drk_x=";
            oss.precision(p_precision[p_num+2]);
            oss<<p[p_num+2]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[p_num+2])<<", drk_y=";
            oss.precision(p_precision[p_num+3]);
            oss<<p[p_num+3]<<" +/- ";
            oss.precision(1);
            oss<<abs(p_error[p_num+3])<<endl;
            break;
        default:
            oss<<endl;
            break;
    }
    oss<<endl;
    return oss.str();
}

int regMode::get_p_num(){
    return p_num;
}

int regMode::get_cp_num(){
    return cp_num;
}

cl::Program regMode::buildImageRegProgram(cl::Context context){
    cl_int ret;
    
    
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
    
    
    string kernel_code="";
    switch (regModeNo) {
        case 0: //XY
            kernel_code += "#define REGMODE 0\n\n";
            break;
            
        case 1: //XY+rotation
            kernel_code += "#define REGMODE 1\n\n";
            break;
            
        case 2: //XY+scale
            kernel_code += "#define REGMODE 2\n\n";
            break;
            
        case 3: //XY+rotation+scale
            kernel_code += "#define REGMODE 3\n\n";
            break;
            
        case 4: //XY+affine
            kernel_code += "#define REGMODE 4\n\n";
            break;
            
        default:
            break;
    }
    switch (cntModeNo) {
        case 0: //no contrast factor
            kernel_code += "#define CNTMODE 0\n\n";
            break;
            
        case 1: //contrast(exp(g)) + bkg(const)
            kernel_code += "#define CNTMODE 1\n\n";
            break;
            
        case 2: //contrast(exp(g)) + bkg(s*x+t*y+const: 1st order plane)
            kernel_code += "#define CNTMODE 2\n\n";
            break;
            
        default:
            break;
    }
    kernel_code += kernel_src;
    //cout << kernel_code<<endl;
    size_t kernel_code_size = kernel_code.length();
    cl::Program::Sources source(1,std::make_pair(kernel_code.c_str(),kernel_code_size));
    cl::Program program(context, source,&ret);
    //cout << kernel_code<<endl;
    //kernel build
    string option = "-cl-nv-maxrregcount=64";
    //option += " -cl-nv-verbose -Werror";
    ret=program.build(option.c_str());
	string logstr=program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]);
	cout << logstr << endl;
    return program;
}