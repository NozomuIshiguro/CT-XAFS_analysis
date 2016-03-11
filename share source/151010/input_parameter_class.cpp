//
//  input_class.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/03/14.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"

int buffsize=512;

void input_parameter::setRegModeFromDialog(string message){
    cout << message;
    cin >> regMode;
}

void input_parameter::setRegMode(string inp_str){
    istringstream iss(inp_str);
    iss >> regMode;
}

void input_parameter::setCntModeFromDialog(string message){
    cout << message;
    cin >> cntMode;
}

void input_parameter::setCntMode(string inp_str){
    istringstream iss(inp_str);
    iss >> cntMode;
}

string input_parameter::getPlatDevList(){
    return ocl_plat_dev_list;
}

void input_parameter::setPlatDevList(string inp_str){
    ocl_plat_dev_list=inp_str;
}

string input_parameter::getInputDir(){
    return data_input_dir;
}

void input_parameter::setInputDirFromDialog(string message){
    cout << message;
    cin >> data_input_dir;
}

void input_parameter::setInputDir(string inputDir){
    data_input_dir=inputDir;
}

string input_parameter::getOutputDir(){
    return data_output_dir;
}

string input_parameter::getOutputFileBase(){
    return output_filebase;
}

int input_parameter::getRefMask_shape(){
    return refMask_shape;
}

int input_parameter::getRefMask_x(){
    return refMask_x;
}

int input_parameter::getRefMask_y(){
    return refMask_y;
}

int input_parameter::getRefMask_width(){
    return refMask_width;
}

int input_parameter::getRefMask_height(){
    return refMask_height;
}

float input_parameter::getRefMask_angle(){
    return refMask_angle;
}

int input_parameter::getSampleMask_shape(){
    return sampleMask_shape;
}

int input_parameter::getSampleMask_x(){
    return sampleMask_x;
}

int input_parameter::getSampleMask_y(){
    return sampleMask_y;
}

int input_parameter::getSampleMask_width(){
    return sampleMask_width;
}

int input_parameter::getSampleMask_height(){
    return sampleMask_height;
}

float input_parameter::getSampleMask_angle(){
    return refMask_angle;
}

void input_parameter::setOutputDirFromDialog(string message){
    cout << message;
    cin >> data_output_dir;
}

void input_parameter::setOutputDir(string outputDir){
    data_output_dir=outputDir;
}

void input_parameter::setOutputFileBaseFromDialog(string message){
    cout << message;
    cin >> output_filebase;
}

void input_parameter::setOutputFileBase(string outputfilebase){
    output_filebase=outputfilebase;
}

string input_parameter::getEnergyFilePath(){
    return energyFilePath;
}

void input_parameter::setEnergyFilePathFromDialog(string message){
    cout << message;
    cin >> energyFilePath;
}

void input_parameter::setEnergyFilePath(string energy_path){
    energyFilePath=energy_path;
}

float input_parameter::getE0(){
    return E0;
}

void input_parameter::setE0FromDialog(string message){
    cout << message;
    cin >> E0;
}

void input_parameter::setE0(string E0_str){
    istringstream iss(E0_str);
    iss >> E0;
}

float input_parameter::getStartEnergy(){
    return startEnergy;
}

float input_parameter::getEndEnergy(){
    return endEnergy;
}


void input_parameter::setEnergyRangeFromDialog(string message){
    cout << message;
    cin >> startEnergy; cin.ignore() >> endEnergy;
}

void input_parameter::setEnergyRange(string E_range){
    istringstream iss(E_range);
    iss >> startEnergy; iss.ignore() >> endEnergy;
}

int input_parameter::getStartEnergyNo(){
    return startEnergyNo;
}

int input_parameter::getEndEnergyNo(){
    return endEnergyNo;
}

void input_parameter::setEnergyNoRangeFromDialog(string message){
    cout << message;
    cin >> startEnergyNo; cin.ignore() >> endEnergyNo;
}

void input_parameter::setEnergyNoRange(string E_N_range){
    istringstream iss(E_N_range);
    iss >> startEnergyNo; iss.ignore() >> endEnergyNo;
}

int input_parameter::getTargetEnergyNo(){
    return targetEnergyNo;
}

int input_parameter::getTargetAngleNo(){
    return targetAngleNo;
}

void input_parameter::setTargetEnergyNoFromDialog(string message){
    cout << message;
    cin >> targetEnergyNo;
}

void input_parameter::setTargetAngleNoFromDialog(string message){
    cout << message;
    cin >> targetAngleNo;
}
void input_parameter::setTargetEnergyNo(string target_E){
    istringstream iss(target_E);
    iss >> targetEnergyNo;
}

void input_parameter::setTargetAngleNo(string target_A){
    istringstream iss(target_A);
    iss >> targetAngleNo;
}

int input_parameter::getStartAngleNo(){
    return startAngleNo;
}

int input_parameter::getEndAngleNo(){
    return endAngleNo;
}

void input_parameter::setAngleRangeFromDialog(string message){
    cout << message;
    cin >> startAngleNo; cin.ignore() >> endAngleNo;
}

void input_parameter::setAngleRange(string ang_range){
    istringstream iss(ang_range);
    iss >> startAngleNo; iss.ignore() >> endAngleNo;
}

vector<float> input_parameter::getFittingPara(){
    return  fitting_para;
}

void input_parameter::setFittingParaFromDialog(string message){
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

void input_parameter::setFittingPara(string fitpara_input){
    istringstream iss(fitpara_input);
    float a;
    for (iss>>a; !iss.eof(); iss.ignore()>>a) {
        fitting_para.push_back(a);
    }
    fitting_para.push_back(a);
}

vector<char> input_parameter::getFreeFixPara(){
    return free_para;
}

void input_parameter::setFreeFixParaFromDialog(string message){
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

void input_parameter::setFreeFixPara(string freepara_inp){
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

vector<string> input_parameter::getFittingParaName(){
    return parameter_name;
}

void input_parameter::setFittingParaNameFromDialog(string message){
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

void input_parameter::setFittingParaName(string fitparaname_inp){    
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

vector<float> input_parameter::getParaUpperLimit(){
    return  para_upperLimit;
}

vector<float> input_parameter::getParaLowerLimit(){
    return para_lowerLimit;
}

int input_parameter::getRegMode(){
    return regMode;
}

int input_parameter::getCntMode(){
    return cntMode;
}

void input_parameter::setValidParaLowerLimitFromDialog(string message){
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

void input_parameter::setValidParaUpperLimitFromDialog(string message){
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

void input_parameter::setValidParaLimit(string limit){
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

float input_parameter::getBaseup(){
    return baseup;
}

void input_parameter::setBaseupFromDialog(string message){
    cout << message;
    cin >> baseup;
}

int input_parameter::getStartX(){
    return startX;
}

int input_parameter::getEndX(){
    return endX;
}

int input_parameter::getStartZ(){
    return startZ;
}

int input_parameter::getEndZ(){
    return endZ;
}

bool input_parameter::getZcorr(){
    if (Z_corr==1) return true;
    else return false;
}

bool input_parameter::getXcorr(){
    if (X_corr==1) return true;
    else return false;
}

int input_parameter::getNumTrial(){
    return num_trial;
}

float input_parameter::getLambda_t(){
    return lamda_t;
}

void input_parameter::setNumtrial(int numTrial_inp){
    num_trial = numTrial_inp;
}

void input_parameter::setLambda_t(float lambda_t_inp){
    lamda_t = lambda_t_inp;
}

input_parameter::input_parameter(string inputfile_path){
    
    ocl_plat_dev_list="";
    data_input_dir="";
    data_output_dir="";
    output_filebase="";
    startEnergyNo=NAN;
    endEnergyNo=NAN;
    startAngleNo=NAN;
    endAngleNo=NAN;
    targetEnergyNo=NAN;
    targetAngleNo=NAN;
    
    energyFilePath="";
    E0=NAN;
    startEnergy=NAN;
    endEnergy=NAN;
    
    refMask_shape=-1;
    refMask_x=IMAGE_SIZE_X/2;
    refMask_y=IMAGE_SIZE_Y/2;
    refMask_width=IMAGE_SIZE_X;
    refMask_height=IMAGE_SIZE_Y;
    refMask_angle=0.0f;
    
    sampleMask_shape=-1;
    sampleMask_x=IMAGE_SIZE_X/2;
    sampleMask_y=IMAGE_SIZE_Y/2;
    sampleMask_width=IMAGE_SIZE_X;
    sampleMask_height=IMAGE_SIZE_Y;
    sampleMask_angle=0.0f;
    
    baseup=NAN;
    Z_corr=0;
    X_corr=0;
    startX=NAN;
    endX=NAN;
    startZ=NAN;
    endZ=NAN;
    
    num_trial = 20;
    lamda_t = 0.001f;
    
    regMode = 0;
    cntMode = 0;
    
    
    ifstream inp_ifs(inputfile_path,ios::in);
    if(inp_ifs) {
        cout<<"input file found."<<endl<<endl;
        while (!inp_ifs.eof()) {
			char *buffer;
			buffer = new char[buffsize];
            inp_ifs.getline(buffer, buffsize);
            if((string)buffer=="#Open CL Platform & Device No"){
                cout<<buffer<<endl;
                inp_ifs.getline(buffer, buffsize);
                ocl_plat_dev_list=buffer;
                cout<<"  "<<ocl_plat_dev_list<<endl;
            }else if((string)buffer=="#Input directory path"){
                cout<<buffer<<endl;
                inp_ifs.getline(buffer, buffsize);
                data_input_dir=buffer;
                cout<<"  "<<data_input_dir<<endl;
            }else if((string)buffer=="#Output directory path"){
                cout<<buffer<<endl;
                inp_ifs.getline(buffer, buffsize);
                data_output_dir=buffer;
                cout<<"  "<<data_output_dir<<endl;
            }else if((string)buffer=="#Output file base name"){
                cout<<buffer<<endl;
                inp_ifs.getline(buffer, buffsize);
                output_filebase=buffer;
                cout<<"  "<<output_filebase<<endl;
            }else if((string)buffer=="#Energy data file path"){
                cout<<buffer<<endl;
                inp_ifs.getline(buffer, buffsize);
                energyFilePath=buffer;
                cout<<"  "<<energyFilePath<<endl;
            }else if((string)buffer=="#E0"){
                cout<<buffer<<endl;
                inp_ifs>>E0;
                cout<<"  "<<E0<<endl;
            }else if((string)buffer=="#Reference energy number for image registration"){
                cout<<buffer<<endl;
                inp_ifs>>targetEnergyNo;
                cout<<"  "<<targetEnergyNo<<endl;
            }else if((string)buffer=="#Reference Angle number for image registration"){
                cout<<buffer<<endl;
                inp_ifs>>targetAngleNo;
                cout<<"  "<<targetAngleNo<<endl;
            }else if((string)buffer=="#Reference loop number for image registration"){
                cout<<buffer<<endl;
                inp_ifs>>targetAngleNo; //use targetAngleNo as target loop No.
                cout<<"  "<<targetAngleNo<<endl;
            }else if((string)buffer=="#Energy range"){
                cout<<buffer<<endl;
                inp_ifs>>startEnergy; inp_ifs.ignore() >> endEnergy;
                cout<<startEnergy<<endl;
            }else if((string)buffer=="#Angle number range"){
                cout<<buffer<<endl;
                inp_ifs>>startAngleNo; inp_ifs.ignore() >> endAngleNo;
                cout<<"  "<<startAngleNo<<"-"<<endAngleNo<<endl;
            }else if((string)buffer=="#Loop number range"){
                cout<<buffer<<endl;
                inp_ifs>>startAngleNo; inp_ifs.ignore() >> endAngleNo;
                //use start/endAngleNo as start/end loop No.
                cout<<"  "<<startAngleNo<<"-"<<endAngleNo<<endl;
            }else if((string)buffer=="#Energy number range"){
                cout<<buffer<<endl;
                inp_ifs>>startEnergyNo; inp_ifs.ignore() >> endEnergyNo;
                cout<<"  "<<startEnergyNo<<"-"<<endEnergyNo<<endl;
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
            }else if((string)buffer=="#Registration mode"){
                cout<<buffer<<endl;
                inp_ifs>>regMode;
                cout<<"  "<<regMode<<endl;
            }else if((string)buffer=="#Registration contrast mode"){
                cout<<buffer<<endl;
                inp_ifs>>cntMode;
                cout<<"  "<<cntMode<<endl;
            }else if((string)buffer=="#Reference mask shape"){
                cout<<buffer<<endl;
                inp_ifs>>refMask_shape;
                cout<<"  "<<refMask_shape<<endl;
            }else if((string)buffer=="#Reference mask center coordination"){
                cout<<buffer<<endl;
                inp_ifs>>refMask_x; inp_ifs.ignore() >> refMask_y;
                cout<<"  "<<refMask_x<<"-"<<refMask_y<<endl;
            }else if((string)buffer=="#Reference mask size"){
                cout<<buffer<<endl;
                inp_ifs>>refMask_width; inp_ifs.ignore() >> refMask_height;
                cout<<"  "<<refMask_width<<"-"<<refMask_height<<endl;
            }else if((string)buffer=="#Reference mask rotation angle"){
                cout<<buffer<<endl;
                inp_ifs>>refMask_angle;
                cout<<"  "<<refMask_angle<<endl;
            }else if((string)buffer=="#Sample mask shape"){
                cout<<buffer<<endl;
                inp_ifs>>sampleMask_shape;
                cout<<"  "<<sampleMask_shape<<endl;
            }else if((string)buffer=="#Sample mask center coordination"){
                cout<<buffer<<endl;
                inp_ifs>>sampleMask_x; inp_ifs.ignore() >> sampleMask_y;
                cout<<"  "<<sampleMask_x<<"-"<<sampleMask_y<<endl;
            }else if((string)buffer=="#Sample mask size"){
                cout<<buffer<<endl;
                inp_ifs>>sampleMask_width; inp_ifs.ignore() >> sampleMask_height;
                cout<<"  "<<sampleMask_width<<"-"<<sampleMask_height<<endl;
            }else if((string)buffer=="#Sample mask rotation angle"){
                cout<<buffer<<endl;
                inp_ifs>>refMask_angle;
                cout<<"  "<<refMask_angle<<endl;
            }else if((string)buffer=="#Image baseup"){
                cout<<buffer<<endl;
                inp_ifs>>baseup;
                cout<<"  "<<baseup<<endl;
            }else if((string)buffer=="#Z-correction"){
                cout<<buffer<<endl;
                inp_ifs>>Z_corr;
                cout<<"  "<<baseup<<endl;
            }else if((string)buffer=="#X-correction"){
                cout<<buffer<<endl;
                inp_ifs>>X_corr;
                cout<<"  "<<baseup<<endl;
            }else if((string)buffer=="#X range of correction evaluation"){
                cout<<buffer<<endl;
                inp_ifs>>startX; inp_ifs.ignore() >> endX;
                cout<<"  "<<startX<<"-"<<endX<<endl;
            }else if((string)buffer=="#Z range of correction evaluation"){
                cout<<buffer<<endl;
                inp_ifs>>startZ; inp_ifs.ignore() >> endZ;
                cout<<"  "<<startZ<<"-"<<endZ<<endl;
            }else if((string)buffer=="#Number of trial for LM-optimization"){
                cout<<buffer<<endl;
                inp_ifs>>num_trial;
                cout<<"  "<<num_trial<<endl;
            }else if((string)buffer=="#Damping parameter for LM-optimization"){
                cout<<buffer<<endl;
                inp_ifs>>lamda_t;
                cout<<"  "<<lamda_t<<endl;
            }
        }
        inp_ifs.close();
    }
}