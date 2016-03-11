//
//  atan_lor_linear_fitting.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/03/13.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "XANES_fitting.hpp"

float* fitting_eq::freefit_para(){
    return freefitting_para;
}

string fitting_eq::param_name(int i) {
    return parameter_name[i];
}

size_t fitting_eq::ParaSize(){
    return param_size;
}


size_t fitting_eq::freeParaSize(){
    return free_param_size;
}

float* fitting_eq::fit_para(){
    
    return fitting_para;
}

char* fitting_eq::freefix_para(){
    
    return free_para;
}

fitting_eq::fitting_eq(input_parameter inp, string OCL_preprocessor){
    param_size = inp.getFittingPara().size();
    fitting_para = new float[param_size];
    free_para = new char[param_size];
    para_lowerlimit = new float[param_size];
    para_upperlimit = new float[param_size];
    free_param_size=0;
    for (int i=0; i<param_size; i++) {
        fitting_para[i] = inp.getFittingPara()[i];
        free_para[i] = inp.getFreeFixPara()[i];
        para_lowerlimit[i]= inp.getParaLowerLimit()[i];
        para_upperlimit[i]=inp.getParaUpperLimit()[i];
        char buffer=inp.getFreeFixPara()[i];
        if (atoi(&buffer)==1) {
            free_param_size++;
        }
        //cout<<free_param_size<<endl;
    }
    kernel_preprocessor_str = OCL_preprocessor;
    parameter_name = inp.getFittingParaName();
    
    freefitting_para = new float[free_param_size];
	freepara_lowerlimit = new float[free_param_size];
	freepara_upperlimit = new float[free_param_size];
    int t=0;
    for (int i=0; i<param_size; i++) {
        char buffer=free_para[i];
		//cout << buffer << endl;
		//cout << atoi(&buffer) << endl;
        if(atoi(&buffer)==1) {
            freefitting_para[t]=fitting_para[i];
            freepara_lowerlimit[t]=para_lowerlimit[i];
            freepara_upperlimit[t]=para_upperlimit[i];
			t++;
        }
    }
}


string fitting_eq::preprocessor_str(){
    
    //初期値を文字列ベクトルにする。置換する パラメータ（固定するパラメータ）として使用
    vector<string> replaced_str;
    for (int i=0; i<param_size; i++) {
        ostringstream OSS2;
        OSS2<<fitting_para[i];
        replaced_str.push_back(OSS2.str());
        OSS2.flush();
		//cout << replaced_str[i] << endl;
    }
    

	//ターゲットとなる文字列("((float*)(fp))[i]")を文字列ベクトルにする
    vector<string> target_str;
    for (int i=0; i<param_size; i++) {
        ostringstream OSS2;
        OSS2<<"((float*)(fp))["<<i<<"]";
        target_str.push_back(OSS2.str());
        OSS2.flush();
		//cout << target_str[i] << endl;
    }
    

	
    string fit_jacobian_str=kernel_preprocessor_str;
    int t=0;
    for (int i=0; i+t<param_size;) {
		//cout << (int)free_para[i + t] << endl;
		if ((int)free_para[i + t] == (int)'0'){
			//固定するtarget_str[i]="((float*)(fp))[i]"をreplaced_str[i+t](初期値)に置換
			//cout << i + t << endl;
			for (size_t a=fit_jacobian_str.find(target_str[i],0);
                a!=string::npos;
                a = fit_jacobian_str.find(target_str[i],a)) {
                
                fit_jacobian_str.replace(a, target_str[i].length(), replaced_str[i+t]);
            }
			//cout << fit_jacobian_str << endl;

			//　j>iのtarget_str[j]="((float*)(fp))[j]"をtarget_str[j-1]="((float*)(fp))[j]" に置換
            for (int j=i+1; j+t<param_size; j++) {
                for (size_t a=fit_jacobian_str.find(target_str[j],0);
                     a!=string::npos;
                     a=fit_jacobian_str.find(target_str[j],a)) {
                    
                    fit_jacobian_str.replace(a, target_str[j].length(), target_str[j-1]);
                }
            }
			//cout << fit_jacobian_str << endl;
            t++;
		}
		else{
			i++;
		}
    }
	//cout << fit_jacobian_str << endl;
    
    
	//ヤコビアンの削除
    //replacement of jacobian part
	//ターゲットとなる文字列("((float*)(j))[i]")を文字列ベクトルにする
    for (int i=0; i<param_size; i++) {
        ostringstream OSS2;
        OSS2<<"((float*)(j))["<<i<<"]";
        target_str[i]=OSS2.str();
        //cout<<target_str[i]<<endl;
        OSS2.flush();
    }
    
    t=0;
    for (int i=0; i+t<param_size;) {
        if((int)free_para[i+t]==(int)'0'){
			//cout << i + t << endl;
            //固定するtarget_str[i]="((float*)(j))[i]"のある行を削除
			size_t a=fit_jacobian_str.find(target_str[i],0);
            //cout<<"a_"<<i<<":"<<a<<endl;
            size_t b=fit_jacobian_str.find(";",a);
            //cout<<"b_"<<i<<":"<<b<<endl;
            fit_jacobian_str.erase(a,b-a+3);
			//cout << fit_jacobian_str << endl;

			//　j>iのtarget_str[j]="((float*)(j))[j]"をtarget_str[j-1]="((float*)(j))[j]" に置換
			for (int j = i + 1; j + t<param_size; j++) {
				for (size_t a = fit_jacobian_str.find(target_str[j], 0);
					a != string::npos;
					a = fit_jacobian_str.find(target_str[j], a)) {

					fit_jacobian_str.replace(a, target_str[j].length(), target_str[j - 1]);
				}
			}
			//cout << fit_jacobian_str << endl;
            t++;
		}
		else{
			i++;
		}
    }
    
    ostringstream OSS;
    OSS<<fit_jacobian_str;
    return OSS.str();
}
