//
//  input_parameter_reslice_class.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/11/22.
//  Copyright © 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"

static int buffsize=512;

float input_parameter_reslice::getBaseup(){
    return baseup;
}

int input_parameter_reslice::getStartX(){
    return startX;
}

int input_parameter_reslice::getEndX(){
    return endX;
}

int input_parameter_reslice::getStartZ(){
    return startZ;
}

int input_parameter_reslice::getEndZ(){
    return endZ;
}

bool input_parameter_reslice::getZcorr(){
    if (Z_corr==1) return true;
    else return false;
}

bool input_parameter_reslice::getXcorr(){
    if (X_corr==1) return true;
    else return false;
}

int input_parameter_reslice::getLayerN(){
    return layerN;
}
float input_parameter_reslice::getRotCenterShiftStart(){
    return rotCenterShiftStart;
}
int input_parameter_reslice::getRotCenterShiftN(){
    return rotCenterShiftN;
}
float input_parameter_reslice::getRotCenterShiftStep(){
    return rotCenterShiftStep;
}

int input_parameter_reslice::getStartLayer(){
    return startLayer;
}

int input_parameter_reslice::getEndLayer(){
    return endLayer;
}


//set from dialog
void input_parameter_reslice::setBaseupFromDialog(string message){
    cout << message;
    cin >> baseup;
}
void input_parameter_reslice::setLayerNFromDialog(string message){
    cout << message;
    cin >> layerN;
}
void input_parameter_reslice::setRotCenterShiftStartFromDialog(string message){
    cout << message;
    cin >> rotCenterShiftStart;
}
void input_parameter_reslice::setRotCenterShiftNFromDialog(string message){
    cout << message;
    cin >> rotCenterShiftN;
}
void input_parameter_reslice::setRotCenterShiftStepFromDialog(string message){
    cout << message;
    cin >> rotCenterShiftStep;
}


void input_parameter_reslice::setLayerN(string layerN_str){
    istringstream iss(layerN_str);
    iss >> layerN;
}
void input_parameter_reslice::setRotCenterShiftStart(string str){
    istringstream iss(str);
    iss >> rotCenterShiftStart;
}
void input_parameter_reslice::setRotCenterShiftN(string str){
    istringstream iss(str);
    iss >> rotCenterShiftN;
}
void input_parameter_reslice::setRotCenterShiftStep(string str){
    istringstream iss(str);
    iss >> rotCenterShiftStep;
}

bool input_parameter_reslice::getDenoiseBool(){
    return denoise;
}

float input_parameter_reslice::getDenoiseThreshold(){
    return denoise_threshold;
}

//constructor
input_parameter_reslice::input_parameter_reslice(){

    baseup=0.0f;
    Z_corr=0;
    X_corr=0;
    startX=0;
    endX=0;
    startZ=0;
    endZ=0;
    layerN=0;
    startLayer=1;
    endLayer=2048;
	
	rotCenterShiftStart = 0;
	rotCenterShiftN = 0;
	rotCenterShiftStep = NAN;
    
    denoise =false;
    denoise_threshold = 0.0f;
}

input_parameter_reslice::input_parameter_reslice(string inputfile_path){
    
    input_parameter_reslice();
    
    ifstream inp_ifs(inputfile_path,ios::in);
    if(inp_ifs) {
        cout<<"input file found."<<endl<<endl;
        while (!inp_ifs.eof()) {
            char *buffer;
            buffer = new char[buffsize];
            inp_ifs.getline(buffer, buffsize);
            inputFromFile_reslice(buffer, &inp_ifs);
        }
        inp_ifs.close();
    }
}

void input_parameter_reslice::inputFromFile_reslice(char *buffer, ifstream *inp_ifs){
    if((string)buffer=="#Image baseup"){
        cout<<buffer<<endl;
        (*inp_ifs)>>baseup;
        cout<<"  "<<baseup<<endl;
    }else if((string)buffer=="#Z-correction"){
        cout<<buffer<<endl;
        (*inp_ifs)>>Z_corr;
        cout<<"  "<<baseup<<endl;
    }else if((string)buffer=="#X-correction"){
        cout<<buffer<<endl;
        (*inp_ifs)>>X_corr;
        cout<<"  "<<baseup<<endl;
    }else if((string)buffer=="#X range of correction evaluation"){
        cout<<buffer<<endl;
        (*inp_ifs)>>startX; (*inp_ifs).ignore() >> endX;
        cout<<"  "<<startX<<"-"<<endX<<endl;
    }else if((string)buffer=="#Z range of correction evaluation"){
        cout<<buffer<<endl;
        (*inp_ifs)>>startZ; (*inp_ifs).ignore() >> endZ;
        cout<<"  "<<startZ<<"-"<<endZ<<endl;
    }else if((string)buffer=="#Start shift for rotation center search"){
        cout<<buffer<<endl;
        (*inp_ifs)>>rotCenterShiftStart;
        cout<<"  "<<rotCenterShiftStart<<endl;
    }else if((string)buffer=="#Number of shift step for rotation center search"){
        cout<<buffer<<endl;
        (*inp_ifs)>>rotCenterShiftN;
        cout<<"  "<<rotCenterShiftN<<endl;
    }else if((string)buffer=="#Shift step for rotation center search"){
        cout<<buffer<<endl;
        (*inp_ifs)>>rotCenterShiftStep;
        cout<<"  "<<rotCenterShiftStep<<endl;
    }else if((string)buffer=="#Image baseup"){
        cout<<buffer<<endl;
        (*inp_ifs)>>baseup;
        cout<<"  "<<baseup<<endl;
    }else if((string)buffer=="#Z-correction"){
        cout<<buffer<<endl;
        (*inp_ifs)>>Z_corr;
        cout<<"  "<<baseup<<endl;
    }else if((string)buffer=="#X-correction"){
        cout<<buffer<<endl;
        (*inp_ifs)>>X_corr;
        cout<<"  "<<baseup<<endl;
    }else if((string)buffer=="#X range of correction evaluation"){
        cout<<buffer<<endl;
        (*inp_ifs)>>startX; (*inp_ifs).ignore() >> endX;
        cout<<"  "<<startX<<"-"<<endX<<endl;
    }else if((string)buffer=="#Z range of correction evaluation"){
        cout<<buffer<<endl;
        (*inp_ifs)>>startZ; (*inp_ifs).ignore() >> endZ;
        cout<<"  "<<startZ<<"-"<<endZ<<endl;
    }else if((string)buffer=="#Start shift for rotation center search"){
        cout<<buffer<<endl;
        (*inp_ifs)>>rotCenterShiftStart;
        cout<<"  "<<rotCenterShiftStart<<endl;
    }else if((string)buffer=="#Number of shift step for rotation center search"){
        cout<<buffer<<endl;
        (*inp_ifs)>>rotCenterShiftN;
        cout<<"  "<<rotCenterShiftN<<endl;
    }else if((string)buffer=="#Shift step for rotation center search"){
        cout<<buffer<<endl;
        (*inp_ifs)>>rotCenterShiftStep;
        cout<<"  "<<rotCenterShiftStep<<endl;
    }else if ((string)buffer == "#Target layer number for rotation center search") {
        cout << buffer << endl;
        (*inp_ifs) >> layerN;
        cout << "  " << layerN << endl;
    }else if ((string)buffer == "#Output Layer No. range") {
        cout << buffer << endl;
        (*inp_ifs) >> startLayer; (*inp_ifs).ignore()>>endLayer;
        cout << "  " << startLayer << endLayer << endl;
    }else if ((string)buffer == "#Noise removal") {
        cout << buffer << endl;
        (*inp_ifs).getline(buffer, buffsize);
        istringstream iss(buffer);
        char b;
        (*inp_ifs) >> b;
        if (atoi(&b)==1) {
            denoise=true;
        }else{
            denoise=false;
        }
        cout<<"  "<< boolalpha <<denoise<<endl;
    }else if ((string)buffer == "#Noise removal threshold") {
        cout << buffer << endl;
        (*inp_ifs) >> denoise_threshold;
        cout << "  " << denoise_threshold << endl;
    }
}
