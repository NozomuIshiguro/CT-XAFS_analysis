//
//  input_parameter_fitting_class.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/11/22.
//  Copyright © 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"

static int buffsize=512;

//get
string input_parameter_fitting::getFittingOutputDir(){
    return fitting_output_dir;
}

string input_parameter_fitting::getFittingFileBase(){
    return fitting_filebase;
}

int input_parameter_fitting::getFittingStartEnergyNo(){
    return fittingStartEnergyNo;
}

int input_parameter_fitting::getFittingEndEnergyNo(){
    return fittingEndEnergyNo;
}

string input_parameter_fitting::getEnergyFilePath(){
    return energyFilePath;
}

float input_parameter_fitting::getE0(){
    return E0;
}

float input_parameter_fitting::getStartEnergy(){
    return startEnergy;
}

float input_parameter_fitting::getEndEnergy(){
    return endEnergy;
}

vector<float> input_parameter_fitting::getFittingPara(){
    return  fitting_para;
}

vector<char> input_parameter_fitting::getFreeFixPara(){
    return free_para;
}

vector<string> input_parameter_fitting::getFittingParaName(){
    return parameter_name;
}

vector<float> input_parameter_fitting::getParaUpperLimit(){
    return  para_upperLimit;
}

vector<float> input_parameter_fitting::getParaLowerLimit(){
    return para_lowerLimit;
}



//set from dialog
void input_parameter_fitting::setFittingOutputDirFromDialog(string message){
    cout << message;
    cin >> fitting_output_dir;
}

void input_parameter_fitting::setFittingFileBaseFromDialog(string message){
    cout << message;
    cin >> fitting_filebase;
}

void input_parameter_fitting::setEnergyFilePathFromDialog(string message){
    cout << message;
    cin >> energyFilePath;
}

void input_parameter_fitting::setE0FromDialog(string message){
    cout << message;
    cin >> E0;
}

void input_parameter_fitting::setEnergyRangeFromDialog(string message){
    cout << message;
    cin >> startEnergy; cin.ignore() >> endEnergy;
}

void input_parameter_fitting::setFittingParaFromDialog(string message){
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

void input_parameter_fitting::setFreeFixParaFromDialog(string message){
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

void input_parameter_fitting::setFittingParaNameFromDialog(string message){
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

void input_parameter_fitting::setValidParaLowerLimitFromDialog(string message){
    cout << message;
    char *buffer;
    buffer = new char[buffsize];
    cin.getline(buffer, buffsize);
    istringstream iss(buffer);
    float a;
    for (iss>>a; !iss.eof(); iss.ignore()>>a) {
        para_lowerLimit.push_back(a);
    }
    para_lowerLimit.push_back(a);
    
    while (para_lowerLimit.size()<fitting_para.size()) {
        para_lowerLimit.push_back(0);
    }
}

void input_parameter_fitting::setValidParaUpperLimitFromDialog(string message){
    cout << message;
    char *buffer;
    buffer = new char[buffsize];
    cin.getline(buffer, buffsize);
    istringstream iss(buffer);
    float a;
    for (iss>>a; !iss.eof(); iss.ignore()>>a) {
        para_upperLimit.push_back(a);
    }
    para_upperLimit.push_back(a);
    
    while (para_upperLimit.size()<fitting_para.size()) {
        para_upperLimit.push_back(0);
    }
}



//set
void input_parameter_fitting::setFittingOutputDir(string outputDir){
    fitting_output_dir=outputDir;
}

void input_parameter_fitting::setFittingFileBase(string outputfilebase){
    fitting_filebase=outputfilebase;
}

void input_parameter_fitting::setFittingStartEnergyNo(int energyNo){
    fittingStartEnergyNo=energyNo;
}

void input_parameter_fitting::setFittingEndEnergyNo(int energyNo){
    fittingEndEnergyNo=energyNo;
}

void input_parameter_fitting::setEnergyFilePath(string energy_path){
    energyFilePath=energy_path;
}

void input_parameter_fitting::setE0(string E0_str){
    istringstream iss(E0_str);
    iss >> E0;
}

void input_parameter_fitting::setEnergyRange(string E_range){
    istringstream iss(E_range);
    iss >> startEnergy; iss.ignore() >> endEnergy;
}

void input_parameter_fitting::setFittingPara(string fitpara_input){
    istringstream iss(fitpara_input);
    float a;
    for (iss>>a; !iss.eof(); iss.ignore()>>a) {
        fitting_para.push_back(a);
    }
    fitting_para.push_back(a);
}

void input_parameter_fitting::setFreeFixPara(string freepara_inp){
    istringstream iss(freepara_inp);
    char b;
    for (iss>>b; !iss.eof(); iss.ignore()>>b) {
        free_para.push_back(b);
    }
    free_para.push_back(b);
    
    while (free_para.size()<fitting_para.size()) {
        free_para.push_back(1);
    }
}

void input_parameter_fitting::setFittingParaName(string fitparaname_inp){
    char *buffer;
    buffer = new char[buffsize];
    istringstream iss(fitparaname_inp);
    for (;!iss.eof();) {
        iss.get(buffer, buffsize,',');
        //cout<<buffer<<endl;
        parameter_name.push_back(buffer);
        iss.ignore();
    }
}

void input_parameter_fitting::setValidParaLimit(string limit){
    char *buffer;
    buffer = new char[buffsize];
    istringstream iss(limit);
    float a,b;
    for (iss>>a; !iss.eof(); iss.ignore()>>a) {
        iss.ignore()>>b;
        para_lowerLimit.push_back(a);
        para_upperLimit.push_back(b);
    }
    para_lowerLimit.push_back(a);
    para_upperLimit.push_back(b);
    
    while (para_lowerLimit.size()<fitting_para.size()) {
        para_lowerLimit.push_back(0);
    }
    while (para_upperLimit.size()<fitting_para.size()) {
        para_upperLimit.push_back(0);
    }
}

int input_parameter_fitting::getNumTrial_fit() {
	return num_trial_fit;
}

float input_parameter_fitting::getLambda_t_fit() {
	return lamda_t_fit;
}

void input_parameter_fitting::setNumtrial_fit(int numTrial_inp) {
	num_trial_fit = numTrial_inp;
}

void input_parameter_fitting::setLambda_t_fit(float lambda_t_inp) {
	lamda_t_fit = lambda_t_inp;
}



//constructor
input_parameter_fitting::input_parameter_fitting(){

    fitting_output_dir="";
    fitting_filebase="";
    fittingStartEnergyNo=NAN;
    fittingEndEnergyNo=NAN;
    
    energyFilePath="";
    E0=NAN;
    startEnergy=NAN;
    endEnergy=NAN;

	num_trial_fit = 20;
	lamda_t_fit = 0.001f;

}

input_parameter_fitting::input_parameter_fitting(string inputfile_path){
    
    input_parameter_fitting();
    
    ifstream inp_ifs(inputfile_path,ios::in);
    if(inp_ifs) {
        cout<<"input file found."<<endl<<endl;
        while (!inp_ifs.eof()) {
            char *buffer;
            buffer = new char[buffsize];
            inp_ifs.getline(buffer, buffsize);
            if((string)buffer=="#Output directory path"){
                cout<<buffer<<endl;
                inp_ifs.getline(buffer, buffsize);
                fitting_output_dir=buffer;
                cout<<"  "<<fitting_output_dir<<endl;
            }else if((string)buffer=="#Output file base name"){
                cout<<buffer<<endl;
                inp_ifs.getline(buffer, buffsize);
                fitting_filebase=buffer;
                cout<<"  "<<fitting_filebase<<endl;
            }else if((string)buffer=="#Energy data file path"){
                cout<<buffer<<endl;
                inp_ifs.getline(buffer, buffsize);
                energyFilePath=buffer;
                cout<<"  "<<energyFilePath<<endl;
            }else if((string)buffer=="#E0"){
                cout<<buffer<<endl;
                inp_ifs>>E0;
                cout<<"  "<<E0<<endl;
            }else if((string)buffer=="#Energy range"){
                cout<<buffer<<endl;
                inp_ifs>>startEnergy; inp_ifs.ignore() >> endEnergy;
                cout<<"  "<<startEnergy<<"-"<<endEnergy<<endl;
            }else if((string)buffer=="#Fitting parameter name"){
                cout<<buffer<<endl;
                inp_ifs.getline(buffer, buffsize);
                istringstream iss(buffer);
                cout<<"  ";
                for (;!iss.eof();) {
                    iss.get(buffer, buffsize,',');
                    parameter_name.push_back(buffer);
                    cout<<buffer;
                    iss.ignore();
                    if(iss.eof()) cout<<endl;
                    else cout<<",";
                }
            }else if((string)buffer=="#Initial fitting parameter"){
                cout<<buffer<<endl;
                inp_ifs.getline(buffer, buffsize);
                istringstream iss(buffer);
                float a;
                cout<<"  ";
                for (iss>>a; !iss.eof(); iss.ignore()>>a) {
                    fitting_para.push_back(a);
                    cout<<a<<",";
                }
                fitting_para.push_back(a);
                cout<<a<<endl;
            }else if((string)buffer=="#Free/fix parameter"){
                cout<<buffer<<endl;
                inp_ifs.getline(buffer, buffsize);
                istringstream iss(buffer);
                char b;
                cout<<"  ";
                for (iss>>b; !iss.eof(); iss.ignore()>>b) {
                    free_para.push_back(b);
                    if(!iss.eof()) cout<<b<<",";
                    else cout<<b;
                }
                
                while (free_para.size()<fitting_para.size()) {
                    free_para.push_back(1);
                    cout<<","<<1;
                }
                cout<<endl;
            }else if((string)buffer=="#Valid parameter lower limit"){
                cout<<buffer<<endl;
                inp_ifs.getline(buffer, buffsize);
                istringstream iss(buffer);
                float a;
                cout<<"  ";
                for (iss>>a; !iss.eof(); iss.ignore()>>a) {
                    para_lowerLimit.push_back(a);
                    cout<<a<<",";
                }
                para_lowerLimit.push_back(a);
                cout<<a;
                
                while (para_lowerLimit.size()<fitting_para.size()) {
                    para_lowerLimit.push_back(-INFINITY);
                    cout<<",-INF";
                }
                cout<<endl;
            }else if((string)buffer=="#Valid parameter upper limit"){
                cout<<buffer<<endl;
                inp_ifs.getline(buffer, buffsize);
                istringstream iss(buffer);
                float a;
                cout<<"  ";
                for (iss>>a; !iss.eof(); iss.ignore()>>a) {
                    para_upperLimit.push_back(a);
                    cout<<"  "<<a<<",";
                }
                para_upperLimit.push_back(a);
                cout<<a;
                
                while (para_upperLimit.size()<fitting_para.size()) {
                    para_upperLimit.push_back(INFINITY);
                    cout<<",INF";
                }
                cout<<endl;
            }
        }
        inp_ifs.close();
    }
}