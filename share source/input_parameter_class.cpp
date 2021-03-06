﻿//
//  input_class.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/03/14.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"

int buffsize=512;

int input_parameter::getCntFitMode(){
    return cntFit;
}


void input_parameter::setRegModeFromDialog(string message){
    cout << message;
    cin >> regMode;
}

void input_parameter::setRegMode(string inp_str){
    istringstream iss(inp_str);
    iss >> regMode;
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


string input_parameter::getIniFilePath(){
    return iniFilePath;
}

void input_parameter::setIniFilePathFromDialog(string message){
    cout << message;
    cin >> iniFilePath;
}

void input_parameter::setIniFilePath(string ini_path){
    iniFilePath=ini_path;
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

int input_parameter::getScanN(){
    return scanN;
}

int input_parameter::getMergeN(){
    return mergeN;
}

void input_parameter::setNumParallel(int numParallel_inp){
    numParallel = numParallel_inp;
}

vector<float> input_parameter::getReg_inipara(){
    
    return reg_inipara;
}

vector<char> input_parameter::getReg_fixpara() {

	return reg_fixpara;
}


string input_parameter::getRawAngleFilePath(){
    return rawAngleFilePath;
}

string input_parameter::getXAFSparameterFilePath(){
    return XAFSparameterFilePath;
}

void input_parameter::setRawAngleFilePath(string filepath){
    rawAngleFilePath=filepath;
}

void input_parameter::setXAFSparameterFilePath(string filepath){
    XAFSparameterFilePath=filepath;
}

bool input_parameter::getSmootingEnable(){
    return enableSmoothing;
}

int input_parameter::getImageSizeX(){
    return imageSizeX;
}

int input_parameter::getImageSizeY(){
    return imageSizeY;
}

int input_parameter::getImageSizeM(){
    return imageSizeX*imageSizeY;
}

void input_parameter::setImageSizeX(int size){
    imageSizeX = size;
}

void input_parameter::setImageSizeY(int size){
    imageSizeY = size;
}
void input_parameter::adjustLayerRange(){
    startLayer = min(max(startLayer,1),imageSizeY);
    endLayer = min(max(startLayer,endLayer),imageSizeY);
}

void input_parameter::setReg_iniparaFromDialog(string message){
    cout << message;
    char *buffer;
    buffer = new char[buffsize];
    cin.getline(buffer, buffsize);
    istringstream iss(buffer);
    float b;
    for (iss>>b; !iss.eof(); iss.ignore()>>b) {
        reg_inipara.push_back(b);
    }
    reg_inipara.push_back(b);
}

void input_parameter::setReg_fixparaFromDialog(string message) {
	cout << message;
	char *buffer;
	buffer = new char[buffsize];
	cin.getline(buffer, buffsize);
	istringstream iss(buffer);
	char b;
	for (iss >> b; !iss.eof(); iss.ignore() >> b) {
		reg_fixpara.push_back(b);
	}
	reg_fixpara.push_back(b);
}


input_parameter::input_parameter(string inputfile_path){
    
    ocl_plat_dev_list="";
    data_input_dir="";
    data_output_dir="";
    output_filebase="";
    startEnergyNo=-1;
    endEnergyNo=-1;
    startAngleNo=-1;
    endAngleNo=-1;
    targetEnergyNo=-1;
    targetAngleNo=-1;
    mergeN=1;
    cntFit=2;
    
    imageSizeX=2048;
    imageSizeY=2048;
    
    input_parameter_mask();
    input_parameter_fitting();
    input_parameter_reslice();
    
    num_trial = 10;
    lamda_t = 0.001f;
    
    regMode = 0;
    imgRegOutput = true;
    
    scanN=30;
    enableSmoothing=false;
    
    numParallel=-1;
    ifstream inp_ifs(inputfile_path,ios::in);
    if(inp_ifs) {
        cout<<"input file found."<<endl<<endl;
        while (!inp_ifs.eof()) {
            string str = ifs_getline(&inp_ifs);
            inputFromFile_mask(str,&inp_ifs);
            inputFromFile_fitting(str,&inp_ifs);
            inputFromFile_reslice(str,&inp_ifs);
            if(str=="#Open CL Platform & Device No"){
                cout<<str<<endl;
                str = ifs_getline(&inp_ifs);
                ocl_plat_dev_list=str;
                cout<<"  "<<ocl_plat_dev_list<<endl;
            }else if(str=="#Number of Parallel processing per one device"){
                cout<<str<<endl;
                inp_ifs>>numParallel;
                cout<<"  "<<numParallel<<endl;
            }else if(str=="#Input directory path"){
                cout<<str<<endl;
                str = ifs_getline(&inp_ifs);
                data_input_dir=str;
                cout<<"  "<<data_input_dir<<endl;
            }else if(str=="#Output directory path"){
                cout<<str<<endl;
                str = ifs_getline(&inp_ifs);
                data_output_dir=str;
                cout<<"  "<<data_output_dir<<endl;
            }else if(str=="#Output file base name"){
                cout<<str<<endl;
                str = ifs_getline(&inp_ifs);
                output_filebase=str;
                cout<<"  "<<output_filebase<<endl;
            }else if(str == "#Initial transform parameter file path for image registration"){
                cout<<str<<endl;
                str = ifs_getline(&inp_ifs);
                iniFilePath=str;
                cout<<"  "<<iniFilePath<<endl;
            }else if(str=="#Reference energy number for image registration"){
                cout<<str<<endl;
                inp_ifs>>targetEnergyNo;
                cout<<"  "<<targetEnergyNo<<endl;
            }else if(str=="#Target energy number for rotation center search"){
                cout<<str<<endl;
                inp_ifs>>targetEnergyNo;
                cout<<"  "<<targetEnergyNo<<endl;
            }else if(str=="#Reference  number for image registration"){
                cout<<str<<endl;
                inp_ifs>>targetAngleNo;
                cout<<"  "<<targetAngleNo<<endl;
            }else if(str=="#Reference loop number for image registration"){
                cout<<str<<endl;
                inp_ifs>>targetAngleNo; //use targetAngleNo as target loop No.
                cout<<"  "<<targetAngleNo<<endl;
            }else if(str=="#Angle number range"){
                cout<<str<<endl;
                inp_ifs>>startAngleNo; inp_ifs.ignore() >> endAngleNo;
                cout<<"  "<<startAngleNo<<"-"<<endAngleNo<<endl;
            }else if(str=="#Loop number range"){
                cout<<str<<endl;
                inp_ifs>>startAngleNo; inp_ifs.ignore() >> endAngleNo;
                //use start/endAngleNo as start/end loop No.
                cout<<"  "<<startAngleNo<<"-"<<endAngleNo<<endl;
            }else if(str=="#Energy number range"){
                cout<<str<<endl;
                inp_ifs>>startEnergyNo; inp_ifs.ignore() >> endEnergyNo;
                cout<<"  "<<startEnergyNo<<"-"<<endEnergyNo<<endl;
            }else if(str=="#Registration mode"){
                cout<<str<<endl;
                inp_ifs>>regMode;
                cout<<"  "<<regMode<<endl;
            }else if(str=="#Number of trial for LM-optimization"){
                cout<<str<<endl;
                inp_ifs>>num_trial;
                cout<<"  "<<num_trial<<endl;
            }else if(str=="#Damping parameter for LM-optimization"){
                cout<<str<<endl;
                inp_ifs>>lamda_t;
                cout<<"  "<<lamda_t<<endl;
            }else if(str=="#Initial transform parameter for image registration"){
                cout<<str<<endl;
                str = ifs_getline(&inp_ifs);
                istringstream iss(str);
                float a;
                cout<<"  ";
                for (iss>>a; !iss.eof(); iss.ignore()>>a) {
                    reg_inipara.push_back(a);
                    cout<<a<<",";
                }
                reg_inipara.push_back(a);
                cout<<a<<endl;
            }else if (str == "#Free/fix parameter for image registration") {
				cout << str << endl;
				str = ifs_getline(&inp_ifs);
				istringstream iss(str);
				char a;
				cout << "  ";
				for (iss >> a; !iss.eof(); iss.ignore() >> a) {
                    reg_fixpara.push_back(a);
					cout << a << ",";
				}
				reg_fixpara.push_back(a);
				cout << a << endl;
			}
			else if (str == "#Number of scan for I0 and dark image") {
                cout << str << endl;
                inp_ifs >> scanN;
                cout << "  " << scanN << endl;

            }else if (str == "#Number of scan for I0 and dark image") {
                cout << str << endl;
                inp_ifs >> scanN;
                cout << "  " << scanN << endl;
            }else if(str=="#Raw angle data file path"){
                cout<<str<<endl;
                str = ifs_getline(&inp_ifs);
                rawAngleFilePath=str;
                cout<<"  "<<rawAngleFilePath<<endl;
            }else if(str=="#XAFS parameter file path"){
                cout<<str<<endl;
                str = ifs_getline(&inp_ifs);
                XAFSparameterFilePath=str;
                cout<<"  "<<XAFSparameterFilePath<<endl;
            }else if(str=="#Output smoothed energy data file path"){
                cout<<str<<endl;
                str = ifs_getline(&inp_ifs);
                energyFilePath=str;
                cout<<"  "<<energyFilePath<<endl;
            }else if(str=="#Enable/disable smoothing"){
                cout<<str<<endl;
                int a;
                inp_ifs>>a;
                if(a==1){
                    enableSmoothing=true;
                }else{
                    enableSmoothing=false;
                }
            }else if(str=="#Number of scans for dark/I0 images"){
                cout<<str<<endl;
                inp_ifs>>scanN;
                cout<<"  "<<scanN<<endl;
            }else if(str=="#Image pixel size"){
                cout<<str<<endl;
                inp_ifs>>imageSizeX; inp_ifs.ignore() >> imageSizeY;
                cout<<"  "<<imageSizeX<<"x"<<imageSizeY<<endl;
                endLayer=imageSizeY;
            }else if(str=="#Image binning size"){
                cout<<str<<endl;
                inp_ifs>>mergeN;
                cout<<"  "<<mergeN<<endl;
            }else if(str=="#Contrast adjustment"){
                cout<<str<<endl;
                inp_ifs>>cntFit;
                cout<<"  "<<cntFit<<endl;
            }
            
        }
        inp_ifs.close();
    }
}
