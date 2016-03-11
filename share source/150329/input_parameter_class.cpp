//
//  input_class.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/03/14.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"

int buffsize=512;

string input_parameter::getPlatDevList(){
    return ocl_plat_dev_list;
}

string input_parameter::getInputDir(){
    return data_input_dir;
}

void input_parameter::setInputDir(string message){
    cout << message;
    cin >> data_input_dir;
}

string input_parameter::getOutputDir(){
    return data_output_dir;
}

void input_parameter::setOutputDir(string message){
    cout << message;
    cin >> data_output_dir;
}

string input_parameter::getEnergyFilePath(){
    return energyFilePath;
}

void input_parameter::setEnergyFilePath(string message){
    cout << message;
    cin >> energyFilePath;
}

float input_parameter::getE0(){
    return E0;
}

void input_parameter::setE0(string message){
    cout << message;
    cin >> E0;
}

float input_parameter::getStartEnergy(){
    return startEnergy;
}

float input_parameter::getEndEnergy(){
    return endEnergy;
}

void input_parameter::setEnergyRange(string message){
    cout << message;
    cin >> startEnergy; cin.ignore() >> endEnergy;
}

int input_parameter::getStartEnergyNo(){
    return startEnergyNo;
}

int input_parameter::getEndEnergyNo(){
    return endEnergyNo;
}

void input_parameter::setEnergyNoRange(string message){
    cout << message;
    cin >> startEnergyNo; cin.ignore() >> endEnergyNo;
}

int input_parameter::getTargetEnergyNo(){
    return targetEnergyNo;
}

void input_parameter::setTargetEnergyNo(string message){
    cout << message;
    cin >> targetEnergyNo;
}

int input_parameter::getStartAngleNo(){
    return startAngleNo;
}

int input_parameter::getEndAngleNo(){
    return endAngleNo;
}

void input_parameter::setAngleRange(string message){
    cout << message;
    cin >> startAngleNo; cin.ignore() >> endAngleNo;
}

vector<float> input_parameter::getFittingPara(){
    return  fitting_para;
}

void input_parameter::setFittingPara(string message){
    cout << message;
    char *buffer;
	buffer = new char[buffsize];
    cin.getline(buffer, buffsize);
    istringstream iss(buffer);
    float a;
    for (iss>>a; !iss.eof(); iss.ignore()>>a) {
        fitting_para.push_back(a);
    }
    fitting_para.push_back(a);
}

vector<char> input_parameter::getFreeFixPara(){
    return free_para;
}

void input_parameter::setFreeFixPara(string message){
    cout << message;
	char *buffer;
	buffer = new char[buffsize];
    cin.getline(buffer, buffsize);
    istringstream iss(buffer);
    char b;
    for (iss>>b; !iss.eof(); iss.ignore()>>b) {
        free_para.push_back(b);
    }
    free_para.push_back(b);
    
    while (free_para.size()<fitting_para.size()) {
        free_para.push_back(1);
    }
}

vector<string> input_parameter::getFittingParaName(){
    return parameter_name;
}

void input_parameter::setFittingParaName(string message){
    cout << message;
	char *buffer;
	buffer = new char[buffsize];
    cin.getline(buffer, buffsize);
    istringstream iss(buffer);
    for (;!iss.eof();) {
        iss.get(buffer, buffsize,',');
        //cout<<buffer<<endl;
        parameter_name.push_back(buffer);
        iss.ignore();
    }
}

input_parameter::input_parameter(string inputfile_path){
    
    ocl_plat_dev_list="";
    data_input_dir="";
    data_output_dir="";
    startEnergyNo=NAN;
    endEnergyNo=NAN;
    startAngleNo=NAN;
    endAngleNo=NAN;
    targetEnergyNo=NAN;
    
    energyFilePath="";
    E0=NAN;
    startEnergy=NAN;
    endEnergy=NAN;
    
    ifstream inp_ifs(inputfile_path,ios::in);
    if(inp_ifs) {
        cout<<"input file found.\n\n";
        while (!inp_ifs.eof()) {
			char *buffer;
			buffer = new char[buffsize];
            inp_ifs.getline(buffer, buffsize);
            //cout<<buffer<<endl;
            if((string)buffer=="#Open CL Platform & Device No"){
                inp_ifs.getline(buffer, buffsize);
                ocl_plat_dev_list=buffer;
            }else if((string)buffer=="#Input directory path"){
                inp_ifs.getline(buffer, buffsize);
                data_input_dir=buffer;
            }else if((string)buffer=="#Output directory path"){
                inp_ifs.getline(buffer, buffsize);
                data_output_dir=buffer;
            }else if((string)buffer=="#Energy data file path"){
                inp_ifs.getline(buffer, buffsize);
                energyFilePath=buffer;
            }else if((string)buffer=="#E0"){
                inp_ifs>>E0;
            }else if((string)buffer=="#Reference energy number for image registration"){
                inp_ifs>>targetEnergyNo;
            }else if((string)buffer=="#Energy range"){
                inp_ifs>>startEnergy; inp_ifs.ignore() >> endEnergy;
            }else if((string)buffer=="#Angle number range"){
                inp_ifs>>startAngleNo; inp_ifs.ignore() >> endAngleNo;
            }else if((string)buffer=="#Energy number range"){
                inp_ifs>>startEnergyNo; inp_ifs.ignore() >> endEnergyNo;
            }else if((string)buffer=="#Fitting parameter name"){
				char *buffer;
				buffer = new char[buffsize];
                inp_ifs.getline(buffer, buffsize);
                istringstream iss(buffer);
                for (;!iss.eof();) {
                    iss.get(buffer, buffsize,',');
                    //cout<<buffer<<endl;
                    parameter_name.push_back(buffer);
                    iss.ignore();
                }
            }else if((string)buffer=="#Initial fitting parameter"){
				char *buffer;
				buffer = new char[buffsize];
                
                inp_ifs.getline(buffer, buffsize);
                istringstream iss(buffer);
                float a;
                for (iss>>a; !iss.eof(); iss.ignore()>>a) {
                    fitting_para.push_back(a);
                }
                fitting_para.push_back(a);
            }else if((string)buffer=="#Free/fix parameter"){
				char *buffer;
				buffer = new char[buffsize];
                inp_ifs.getline(buffer, buffsize);
                istringstream iss(buffer);
                char b;
                for (iss>>b; !iss.eof(); iss.ignore()>>b) {
                    free_para.push_back(b);
                }
                
                while (free_para.size()<fitting_para.size()) {
                    free_para.push_back(1);
                }
            }
        }
        
        inp_ifs.close();
    }
    
}