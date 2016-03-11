//
//  Regmode_class.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/05/28.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"
#include "imageReg.hpp"
#include "imageregistration_kernel_src_cl.hpp"
#include "XYshift_cl.hpp"
#include "XYshift_rotation_cl.hpp"
#include "XYshift_scale_cl.hpp"
#include "XYshift_rotation_scale_cl.hpp"
#include "XYshift_affine_cl.hpp"



regMode::regMode(int regmodeNumber){
    regModeNo=regmodeNumber;
    
    switch (regModeNo) {
        case 0: //xy shift
            regModeName="xy shift";
            oss_target="    registration shift: dx=0, dy=0\n\n";
            transpara_num=2;
            reductpara_num=6;
            break;
            
        case 1: //rotation+xy shift
            regModeName="rotation + xy shift";
            oss_target="    registration shift: dx=0, dy=0, d(theta)=0\n\n";
            transpara_num=3;
            reductpara_num=10;
            break;
            
        case 2: //scale+xy shift
            regModeName="scale + xy shift";
            oss_target="    registration shift: dx=0, dy=0, scale=1.0\n\n";
            transpara_num=3;
            reductpara_num=10;
            break;
            
        case 3: //scale+rotation+xy shift
            regModeName="scale + rotation + xy shift";
            oss_target="    registration shift: dx=0, dy=0, d(theta)=0, scale=1.0\n\n";
            transpara_num=4;
            reductpara_num=15;
            break;
        
        case 4: //affine+xy shift
            regModeName="affine + xy shift";
            oss_target="    registration shift: dx=0, dy=0, a11=0, a21=0, a12=0, a22=0\n\n";
            transpara_num=6;
            reductpara_num=29;
            break;
            
        default:
            regModeName="none";
            oss_target="";
            transpara_num=1;
            reductpara_num=3;
            break;
    }
}

void regMode::changeRegMode(int regmodeNumber){
    regModeNo=regmodeNumber;
    
    switch (regModeNo) {
        case 0: //xy shift
            regModeName="xy shift";
            oss_target="    registration shift: dx=0, dy=0\n\n";
            transpara_num=2;
            reductpara_num=6;
            break;
            
        case 1: //rotation+xy shift
            regModeName="rotation + xy shift";
            oss_target="    registration shift: dx=0, dy=0, d(theta)=0\n\n";
            transpara_num=3;
            reductpara_num=10;
            break;
            
        case 2: //scale+xy shift
            regModeName="scale + xy shift";
            oss_target="    registration shift: dx=0, dy=0, scale=1.0\n\n";
            transpara_num=3;
            reductpara_num=10;
            break;
            
        case 3: //scale+rotation+xy shift
            regModeName="scale + rotation + xy shift";
            oss_target="    registration shift: dx=0, dy=0, d(theta)=0, scale=1.0\n\n";
            transpara_num=4;
            reductpara_num=15;
            break;
            
        case 4: //affine+xy shift
            regModeName="affine + xy shift";
            oss_target="    registration shift: dx=0, dy=0, a11=0, a21=0, a12=0, a22=0\n\n";
            transpara_num=6;
            reductpara_num=29;
            break;
            
        default:
            regModeName="none";
            oss_target="";
            transpara_num=1;
            reductpara_num=3;
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
            oss<< "dx\tdx error\tdy\tdy error\n"<<endl;
            break;
            
        case 1: //rotation+xy shift
            oss<<"dx\tdx error\tdy\tdy error\td(theta)\td(theta) error"<<endl;
            break;
            
        case 2: //scale+xy shift
            oss<<"dx\tdx error\tdy\tdy error\tscale\tscale error"<<endl;
            break;
            
        case 3: //scale+rotation+xy shift
            oss<<"dx\tdx error\tdy\tdy error\t";
            oss<<"d(theta)\td(theta) error\t";
            oss<<"scale\tscale error"<<endl;
            break;
            
        case 4: //affine+xy shift
            oss<<"dx\tdx error\tdy\tdy error\t";
            oss<<"a11\ta11 error\ta12\ta12 error\t";
            oss<<"a21\ta21 error\ta22\ta22 error"<<endl;
            break;
            
        default: //none
            oss<<"";
            break;
    }
    return oss.str();
}

string regMode::get_regModeName(){
    return regModeName;
}
string regMode::get_oss_target(){
    return oss_target;
}
string regMode::oss_sample(float *transpara,float *transpara_error,
                               int *error_precision){
    ostringstream oss;
    switch (regModeNo) {
        case 0: //xy shift
            oss<<"    registration shift: dx=";
            oss.precision(error_precision[0]);
            oss<<transpara[0]<<" +/- ";
            oss.precision(1);
            oss<<abs(transpara_error[0])<<", dy=";
            oss.precision(error_precision[1]);
            oss<<transpara[1]<<" +/- ";
            oss.precision(1);
            oss<<abs(transpara_error[1])<<"\n\n";
            break;
            
        case 1: //rotation+xy shift
            oss<<"    registration shift: dx=";
            oss.precision(error_precision[0]);
            oss<<transpara[0]<<" +/- ";
            oss.precision(1);
            oss<<abs(transpara_error[0])<<", dy=";
            oss.precision(error_precision[1]);
            oss<<transpara[1]<<" +/- ";
            oss.precision(1);
            oss<<abs(transpara_error[1])<<", d(theta)=";
            oss.precision(error_precision[2]);
            oss<<transpara[2]<<" +/- ";
            oss.precision(1);
            oss<<abs(transpara_error[2])<<"\n\n";
            break;
            
        case 2: //scale+xy shift
            oss<<"    registration shift: dx=";
            oss.precision(error_precision[0]);
            oss<<transpara[0]<<" +/- ";
            oss.precision(1);
            oss<<abs(transpara_error[0])<<", dy=";
            oss.precision(error_precision[1]);
            oss<<transpara[1]<<" +/- ";
            oss.precision(1);
            oss<<abs(transpara_error[1])<<", scale=";
            oss.precision(error_precision[2]);
            oss<<transpara[2]<<" +/- ";
            oss.precision(1);
            oss<<abs(transpara_error[2])<<"\n\n";
            break;
            
        case 3: //scale+rotation+xy shift
            oss<<"    registration shift: dx=";
            oss.precision(error_precision[0]);
            oss<< transpara[0]<<" +/- ";
            oss.precision(1);
            oss<<abs(transpara_error[0])<<", dy=";
            oss.precision(error_precision[1]);
            oss<<transpara[1]<<" +/- ";
            oss.precision(1);
            oss<<abs(transpara_error[1])<<", d(theta)=";
            oss.precision(error_precision[2]);
            oss<<transpara[2]<<" +/- ";
            oss.precision(1);
            oss<<abs(transpara_error[2])<<", scale=";
            oss.precision(error_precision[3]);
            oss<<transpara[3]<<" +/- ";
            oss.precision(1);
            oss<<abs(transpara_error[3])<<"\n\n";
            break;
            
        case 4: //affine+xy shift
            oss<<"    registration shift: dx=";
            oss.precision(error_precision[0]);
            oss<< transpara[0]<<" +/- ";
            oss.precision(1);
            oss<<abs(transpara_error[0])<<", dy=";
            oss.precision(error_precision[1]);
            oss<<transpara[1]<<" +/- ";
            oss.precision(1);
            oss<<abs(transpara_error[1])<<", a11=";
            oss.precision(error_precision[2]);
            oss<<transpara[2]<<" +/- ";
            oss.precision(1);
            oss<<abs(transpara_error[2])<<", a11=";
            oss.precision(error_precision[3]);
            oss<<transpara[3]<<" +/- ";
            oss.precision(1);
            oss<<abs(transpara_error[3])<<", a11=";
            oss.precision(error_precision[4]);
            oss<<transpara[4]<<" +/- ";
            oss.precision(1);
            oss<<abs(transpara_error[4])<<", a11=";
            oss.precision(error_precision[5]);
            oss<<transpara[5]<<" +/- ";
            oss.precision(1);
            oss<<abs(transpara_error[5])<<"\n\n";
            break;
            
        default:
            oss<<"";
            break;
    }
    return oss.str();
}

int regMode::get_transpara_num(){
    return transpara_num;
}

int regMode::get_reductpara_num(){
    return reductpara_num;
}

void regMode::reset_transpara(float *transpara,int num_parallel){
    switch (regModeNo) {
        
        case 2: //scale+xy shift
            for (int j=0; j<num_parallel; j++) {
                for (int i=0; i<transpara_num-1; i++) {
                    transpara[i+transpara_num*j]=0.0f;
                }
                transpara[transpara_num*(j+1)-1]=1.0f;
            }
            break;
        
        case 3: //scale+rotation+xy shift
            for (int j=0; j<num_parallel; j++) {
                for (int i=0; i<transpara_num-1; i++) {
                    transpara[i+transpara_num*j]=0.0f;
                }
                transpara[transpara_num*(j+1)-1]=1.0f;
            }
            break;
            
        case 4: //affine+xy shift
            for (int j=0; j<num_parallel; j++) {
                for (int i=0; i<transpara_num; i++) {
                    if(i==2){
                        transpara[i+transpara_num*j]=1.0f;
                    }else if(i==5){
                        transpara[i+transpara_num*j]=1.0f;
                    }else{
                        transpara[i+transpara_num*j]=0.0f;
                    }
                }
            }
            break;
        
        default:
            for (int i=0; i<transpara_num*num_parallel; i++) {
                transpara[i]=0.0f;
            }
            break;
    }
}

cl::Program regMode::buildImageRegProgram(cl::Context context, int definite){
    cl_int ret;
    
    /*//kernel source from cl file
    ifstream ifs("/Users/ishiguro/Desktop/CT_programs/CT-XANES_analysis/ImageReg share source/imageregistration_kernel_src.cl", ios::in);
     if(!ifs) {
         cerr << "   Failed to load kernel \n" << endl;
     }
     istreambuf_iterator<char> it(ifs);
     istreambuf_iterator<char> last;
     string kernel_src(it,last);
     ifs.close();
    //kernel preprocessor source (XY) from cl file
     ifstream ifs2("/Users/ishiguro/Desktop/CT_programs/CT-XANES_analysis/CT_imageRegistration_multi/XYshift.cl",ios::in);
     if(!ifs2) {
     cerr << "   Failed to load kernel \n\n" << endl;
     return -1;
     }
     istreambuf_iterator<char> it2(ifs2);
     istreambuf_iterator<char> last2;
     string kernel_src_XY(it2,last2);
     ifs2.close();
     //kernel preprocessor source (XY) from cl file
     ifstream ifs3("/Users/ishiguro/Desktop/CT_programs/CT-XANES_analysis/CT_imageRegistration_multi/XYshift_rotation.cl",ios::in);
     if(!ifs3) {
     cerr << "   Failed to load kernel \n\n" << endl;
     return -1;
     }
     istreambuf_iterator<char> it3(ifs3);
     istreambuf_iterator<char> last3;
     string kernel_src_XYrot(it3,last3);
     ifs3.close();
     //kernel preprocessor source (XY) from cl file
     ifstream ifs4("/Users/ishiguro/Desktop/CT_programs/CT-XANES_analysis/ImageReg share source/XYshift_scale.cl",ios::in);
     if(!ifs4) {
         cerr << "   Failed to load kernel \n\n" << endl;
     }
     istreambuf_iterator<char> it4(ifs4);
     istreambuf_iterator<char> last4;
     string kernel_src_XYscal(it4,last4);
     ifs4.close();*/
    
    string kernel_code="";
    kernel_code+="#define NUM_TRIAL 20\n\n";
    switch (regModeNo) {
        case 0: //XY
            kernel_code += kernel_src_XY;
            break;
            
        case 1: //XY+rotation
            kernel_code += kernel_src_XYrot;
            break;
            
        case 2: //XY+scale
            kernel_code += kernel_src_XYscal;
            break;
            
        case 3: //XY+rotation+scale
            kernel_code += kernel_src_XYrotscal;
            break;
            
        case 4: //XY+affine
            kernel_code += kernel_src_XYaffine;
            break;
            
        default:
            break;
    }
    if (definite) {
        kernel_code+="#define DEFINITE\n\n";
    }
    //cout << kernel_code<<endl;
    kernel_code += kernel_src;
    size_t kernel_code_size = kernel_code.length();
    cl::Program::Sources source(1,std::make_pair(kernel_code.c_str(),kernel_code_size));
    cl::Program program(context, source,&ret);
    //cout << kernel_code<<endl;
    //kernel build
    ret=program.build();
    
    return program;
}