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

int input_parameter::getRegMode(){
    return regMode;
}

int input_parameter::getCntMode(){
    return cntMode;
}


int input_parameter::getNumTrial(){
    return num_trial;
}

float input_parameter::getLambda_t(){
    return lamda_t;
}

bool input_parameter::getImgRegOutput(){
    return imgRegOutput;
}

void input_parameter::setNumtrial(int numTrial_inp){
    num_trial = numTrial_inp;
}

void input_parameter::setLambda_t(float lambda_t_inp){
    lamda_t = lambda_t_inp;
}

int input_parameter::getNumParallel(){
    return numParallel;
}

void input_parameter::setNumParallel(int numParallel_inp){
    numParallel = numParallel_inp;
}

vector<float> input_parameter::getRegCnt_inipara(){
    
    return regCnt_inipara;
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
    
    input_parameter_mask();
    input_parameter_fitting();
    input_parameter_reslice();
    
    num_trial = 10;
    lamda_t = 0.001f;
    
    regMode = 0;
    cntMode = 0;
    imgRegOutput = true;
    
    numParallel=NAN;
    
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
            }else if((string)buffer=="#Number of Parallel processing per one device"){
                cout<<buffer<<endl;
                inp_ifs>>numParallel;
                cout<<"  "<<numParallel<<endl;
            }
                else if((string)buffer=="#Input directory path"){
                cout<<buffer<<endl;
                inp_ifs.getline(buffer, buffsize);
                data_input_dir=buffer;
                cout<<"  "<<data_input_dir<<endl;
            }else if((string)buffer=="#Output directory path"){
                cout<<buffer<<endl;
                inp_ifs.getline(buffer, buffsize);
                data_output_dir=buffer;
                cout<<"  "<<data_output_dir<<endl;
            }else if((string)buffer=="#Output directory path for XANES fitting"){
                cout<<buffer<<endl;
                inp_ifs.getline(buffer, buffsize);
                fitting_output_dir=buffer;
                cout<<"  "<<fitting_output_dir<<endl;
            }else if((string)buffer=="#Output file base name"){
                cout<<buffer<<endl;
                inp_ifs.getline(buffer, buffsize);
                output_filebase=buffer;
                cout<<"  "<<output_filebase<<endl;
            }else if((string)buffer=="#Output file base name for XANES fitting"){
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
            }else if((string)buffer=="#Reference energy number for image registration"){
                cout<<buffer<<endl;
                inp_ifs>>targetEnergyNo;
                cout<<"  "<<targetEnergyNo<<endl;
            }else if((string)buffer=="#Target energy number for rotation center search"){
                cout<<buffer<<endl;
                inp_ifs>>targetEnergyNo;
                cout<<"  "<<targetEnergyNo<<endl;
            }else if((string)buffer=="#Reference  number for image registration"){
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
                cout<<"  "<<startEnergy<<"-"<<endEnergy<<endl;
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
            }else if((string)buffer=="#Processing parameter name"){
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
                inp_ifs>>sampleMask_angle;
                cout<<"  "<<sampleMask_angle<<endl;
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
            }else if((string)buffer=="#Initial transform parameter for image Registration"){
                cout<<buffer<<endl;
                inp_ifs.getline(buffer, buffsize);
                istringstream iss(buffer);
                float a;
                cout<<"  ";
                for (iss>>a; !iss.eof(); iss.ignore()>>a) {
                    regCnt_inipara.push_back(a);
                    cout<<a<<",";
                }
                regCnt_inipara.push_back(a);
                cout<<a<<endl;
            }else if((string)buffer=="#Target energy number for rotation center search"){
                cout<<buffer<<endl;
                inp_ifs>>layerN;
                cout<<"  "<<layerN<<endl;
            }else if((string)buffer=="#Start shift for rotation center search"){
                cout<<buffer<<endl;
                inp_ifs>>rotCenterShiftStart;
                cout<<"  "<<rotCenterShiftStart<<endl;
            }else if((string)buffer=="#Number of shift step for rotation center search"){
                cout<<buffer<<endl;
                inp_ifs>>rotCenterShiftN;
                cout<<"  "<<rotCenterShiftN<<endl;
            }else if((string)buffer=="#Shift step for rotation center search"){
                cout<<buffer<<endl;
                inp_ifs>>rotCenterShiftStep;
                cout<<"  "<<rotCenterShiftStep<<endl;
            }else if ((string)buffer == "#Target layer number for rotation center search") {
                cout << buffer << endl;
                inp_ifs >> layerN;
                cout << "  " << layerN << endl;
            }
        }
        inp_ifs.close();
    }
}