//
//  input_parameter_mask_class.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/11/22.
//  Copyright © 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"

static int buffsize=512;

//ref mask get
int input_parameter_mask::getRefMask_shape(){
    return refMask_shape;
}

int input_parameter_mask::getRefMask_x(){
    return refMask_x;
}

int input_parameter_mask::getRefMask_y(){
    return refMask_y;
}

int input_parameter_mask::getRefMask_width(){
    return refMask_width;
}

int input_parameter_mask::getRefMask_height(){
    return refMask_height;
}

float input_parameter_mask::getRefMask_angle(){
    return refMask_angle;
}



//sample mask get
int input_parameter_mask::getSampleMask_shape(){
    return sampleMask_shape;
}

int input_parameter_mask::getSampleMask_x(){
    return sampleMask_x;
}

int input_parameter_mask::getSampleMask_y(){
    return sampleMask_y;
}

int input_parameter_mask::getSampleMask_width(){
    return sampleMask_width;
}

int input_parameter_mask::getSampleMask_height(){
    return sampleMask_height;
}

float input_parameter_mask::getSampleMask_angle(){
    return sampleMask_angle;
}



//constructor
input_parameter_mask::input_parameter_mask(){
    
    
    refMask_shape=-1;
    refMask_x=1024;
    refMask_y=1024;
    refMask_width=2048;
    refMask_height=2048;
    refMask_angle=0.0f;
    
    sampleMask_shape=-1;
    sampleMask_x=1024;
    sampleMask_y=1024;
    sampleMask_width=2048;
    sampleMask_height=2048;
    sampleMask_angle=0.0f;

}

input_parameter_mask::input_parameter_mask(string inputfile_path){
    
    input_parameter_mask();
    
    ifstream inp_ifs(inputfile_path,ios::in);
    if(inp_ifs) {
        cout<<"input file found."<<endl<<endl;
        while (!inp_ifs.eof()) {
            char *buffer;
            buffer = new char[buffsize];
            inp_ifs.getline(buffer, buffsize);
            inputFromFile_mask(buffer,&inp_ifs);
        }
        inp_ifs.close();
    }
}

void input_parameter_mask::inputFromFile_mask(string str,ifstream *inp_ifs){
    if(str=="#Reference mask shape"){
        cout<<str<<endl;
        *inp_ifs>>refMask_shape;
        cout<<"  "<<refMask_shape<<endl;
    }else if(str=="#Reference mask center coordination"){
        cout<<str<<endl;
        *inp_ifs>>refMask_x; (*inp_ifs).ignore() >> refMask_y;
        cout<<"  "<<refMask_x<<","<<refMask_y<<endl;
    }else if(str=="#Reference mask size"){
        cout<<str<<endl;
        *inp_ifs>>refMask_width; (*inp_ifs).ignore() >> refMask_height;
        cout<<"  "<<refMask_width<<","<<refMask_height<<endl;
    }else if(str=="#Reference mask rotation angle"){
        cout<<str<<endl;
        *inp_ifs>>refMask_angle;
        cout<<"  "<<refMask_angle<<endl;
    }else if(str=="#Sample mask shape"){
        cout<<str<<endl;
        *inp_ifs>>sampleMask_shape;
        cout<<"  "<<sampleMask_shape<<endl;
    }else if(str=="#Sample mask center coordination"){
        cout<<str<<endl;
        *inp_ifs>>sampleMask_x; (*inp_ifs).ignore() >> sampleMask_y;
        cout<<"  "<<sampleMask_x<<","<<sampleMask_y<<endl;
    }else if(str=="#Sample mask size"){
        cout<<str<<endl;
        *inp_ifs>>sampleMask_width; (*inp_ifs).ignore() >> sampleMask_height;
        cout<<"  "<<sampleMask_width<<","<<sampleMask_height<<endl;
    }else if(str=="#Sample mask rotation angle"){
        cout<<str<<endl;
        *inp_ifs>>sampleMask_angle;
        cout<<"  "<<sampleMask_angle<<endl;
    }
}
