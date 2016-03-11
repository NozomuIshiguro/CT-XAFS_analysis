//
//  main.cpp
//  OpenCL_Device_Info
//
//  Created by Nozomu Ishiguro on 2015/03/06.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"

int Deviceinfo(OCL_platform_device plat_dev_list,bool supportImg);

int main(int argc, const char * argv[]) {
        
    input_parameter inp("");
    if (argc>1) {
        inp.setPlatDevList(argv[1]);
    }
    
    OCL_platform_device plat_dev_list(inp.getPlatDevList(),false);
    cout << endl;
    
    cout << "Display supported image formats? (y/n): n"<< endl;
    string boolstr,dummy;
    getline(cin,dummy);
    getline(cin,boolstr);
    bool supportImg;
    if (boolstr=="y") {
        supportImg=true;
    }else if (boolstr=="n"){
        supportImg=false;
    }else{
        supportImg=false;
    }
    cout << endl;
    
    Deviceinfo(plat_dev_list,supportImg);
    
    cout << endl << "Press 'Enter' to quit." << endl;
    getline(cin,dummy);
    
    return 0;
}
