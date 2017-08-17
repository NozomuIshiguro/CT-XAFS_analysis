//
//  fitting_class.cpp
//  XANES_fitting
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

float* fitting_eq::lowerLimit(){
    
    return para_lowerlimit;
}

float* fitting_eq::upperLimit(){
    
    return para_upperlimit;
}

float* fitting_eq::paraAttenuator() {

	return paraAtten;
}

char* fitting_eq::freefix_para(){
    
    return free_para;
}

void fitting_eq::setInitialParameter(vector<float> iniPara){
    
    for (int i=0; i<param_size; i++) {
        fitting_para[i]=iniPara[i];
    }
}

void fitting_eq::setFreeFixParameter(vector<char> freefixPara){
    for (int i=0; i<param_size; i++) {
        free_para[i]=freefixPara[i];
    }
}


void fitting_eq::setFittingEquation(vector<string> fittingFuncList){
    //input function mode list
    numFunc = (int)(fittingFuncList.size());
    param_size = 0;
    for (int i=0; i<numFunc; i++) {
        if(fittingFuncList[i]=="line"){
            funcmode.push_back(0);
            param_size += 2;
        }else if(fittingFuncList[i]=="Gaussian"){
            funcmode.push_back(1);
            param_size += 3;
        }else if(fittingFuncList[i]=="Lorentzian"){
            funcmode.push_back(2);
            param_size += 3;
        }else if(fittingFuncList[i]=="atanE"){
            funcmode.push_back(3);
            param_size += 3;
        }else if(fittingFuncList[i]=="erfE"){
            funcmode.push_back(4);
            param_size += 3;
        }else if(fittingFuncList[i]=="stdXANES"){
            funcmode.push_back(5);
            param_size += 2;
        }else if(fittingFuncList[i]=="Victoreen"){
            funcmode.push_back(6);
            param_size += 4;
        }else if(fittingFuncList[i]=="McMaster"){
            funcmode.push_back(7);
            param_size += 3;
        }else if(fittingFuncList[i]=="3rdPolynomical"){
            funcmode.push_back(8);
            param_size += 4;
        }
    }
    
    fitting_para = new float[param_size];
    free_para = new char[param_size];
    para_lowerlimit = new float[param_size];
    para_upperlimit = new float[param_size];
    paraAtten = new float[param_size];
  
}


fitting_eq::fitting_eq(){
    numLCF=0;
    param_size=0;
    numFunc=0;
}

fitting_eq::fitting_eq(input_parameter inp){
    //input function mode list
    setFittingEquation(inp.funcNameList);
    //param_size = inp.numParameter;//getFittingPara().size();

    free_param_size=0;
    for (int i=0; i<param_size; i++) {
        fitting_para[i] = inp.getFittingPara()[i];
        free_para[i] = (inp.getFreeFixPara()[i]) ? '1':'0';
        para_lowerlimit[i]= inp.getParaLowerLimit()[i];
        para_upperlimit[i]=inp.getParaUpperLimit()[i];
		if (inp.getParaAttenuator().size() != 0) {
			paraAtten[i] = inp.getParaAttenuator()[i];
		}
        if (inp.getFreeFixPara()[i]) {
            free_param_size++;
        }
        //cout<<free_param_size<<endl;
    }
    parameter_name = inp.getFittingParaName();
	//cout << free_param_size << endl;
    
	freefitting_para = new float[free_param_size];
	freepara_lowerlimit = new float[free_param_size];
	freepara_upperlimit = new float[free_param_size];

    int t=0;
    for (int i=0; i<param_size; i++) {
        //char buffer=free_para[i];
		//cout << buffer << endl;
		//cout << atoi(&buffer) << endl;
        //if(atoi(&buffer)==1) {
		if (free_para[i] == '1') {
            freefitting_para[t]=fitting_para[i];
            freepara_lowerlimit[t]=para_lowerlimit[i];
            freepara_upperlimit[t]=para_upperlimit[i];
			t++;
        }
    }
    
    for (int i=0; i<param_size; i++) {
        //char buffer=free_para[i];
		//cout << para_lowerlimit[i] << "," << para_upperlimit[i] << endl;
		//cout << isnan(para_lowerlimit[i]) << "," << isnan(para_upperlimit[i]) << endl;
        //bool b1 = (atoi(&buffer)==1);
		bool b1 = (free_para[i] == '1');
		bool b2 = (isnan(para_lowerlimit[i])==0);
        bool b3 = (isinf(para_lowerlimit[i])==0);
        bool b4 = (isnan(para_upperlimit[i])==0);
        bool b5 = (isinf(para_upperlimit[i])==0);
        if(b1 && b2 && b3) {
            vector<float> Cm(param_size,0.0f);
            Cm[i]=-1.0f;
            C_matrix.push_back(Cm);
            D_vector.push_back(-para_lowerlimit[i]);
		}
        if(b1 && b4 && b5) {
            vector<float> Cm(param_size,0.0f);
            Cm[i]=1.0f;
            C_matrix.push_back(Cm);
            D_vector.push_back(para_upperlimit[i]);
        }
    }
	cout << "C matrix | D vector" << endl;
	for (int i = 0; i < C_matrix.size(); i++) {
		for (int j = 0; j < C_matrix[i].size(); j++) {
			cout << C_matrix[i][j] << " ";
		}
		cout << "| " << D_vector[i] << endl;
	}
    cout<<endl;
    constrain_size = D_vector.size();
    
    
    //input LCF standard
    numLCF=inp.numLCF;
    //energy file input & processing
    for (int i=0; i<numLCF; i++) {
		cout << "Data input of standard XANES spectrum(" << i + 1 << ")" << endl;
		ifstream energy_ifs(inp.LCFstd_paths[i],ios::in);
        if (!energy_ifs.is_open()) {
            cout<<"Standard XAFS file not found."<<endl;
            cout <<  "Press 'Enter' to quit." << endl;
            string dummy;
            getline(cin,dummy);
            exit(-1);
        }
        vector<float> energy, mt;
        LCFstd_size.push_back(0);
		int j = 0;
        do {
			string a,b;
			energy_ifs >> a >> b;

            if (energy_ifs.eof()) break;
			float aa, bb;
			try {
				aa = stof(a);
				bb = stof(b);
			}catch(invalid_argument ret){ //ヘッダータグが存在する場合に入力エラーになる際への対応
				continue;
			}
            energy.push_back(aa);
            mt.push_back(bb);
            cout<<j+1<<": "<<aa<<","<<bb<<endl;
            LCFstd_size[i]++;
            j++;
        } while (!energy_ifs.eof());
        energy_ifs.close();
        cout<<endl;

        
        LCFstd_E.push_back(energy);
        LCFstd_mt.push_back(mt);
    }
}


