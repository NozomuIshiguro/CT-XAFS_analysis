//
//  main.cpp
//  Reslice
//
//  Created by Nozomu Ishiguro on 2015/06/22.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "reslice.hpp"
vector<thread> input_th, reslice_th, output_th1, output_th2;

int main(int argc, const char * argv[]) {
    string fp_str;
    if (argc>1) {
        fp_str=argv[1];
    }else{
        string dummy;
        cout<<"Set input file path, if existed."<<endl;
        getline(cin,dummy);
        istringstream iss(dummy);
        iss>>fp_str;
        //fp_str="./image_reg.inp";
    }
    
    input_parameter inp(fp_str);
    
    OCL_platform_device plat_dev_list(inp.getPlatDevList(),true);
    cout << endl;
    
    
    // Input directory settings
    string buffer;
    buffer = output_flag("-ip", argc, argv);
    if (buffer.length()>0) {
        inp.setInputDir(buffer);
    }
    if (inp.getInputDir().length()==0) {
        inp.setInputDirFromDialog("Set input mt raw file directory.\n");
    }
    //processing energy No range
    buffer = output_flag("-enr", argc, argv);
    if (buffer.length()>0) {
        inp.setEnergyNoRange(buffer);
    }
    //parameter name
    buffer = output_flag("-fpn", argc, argv);
    if (buffer.length()>0) {
        inp.setFittingParaName(buffer);
    }
    
    if ((inp.getStartEnergyNo()<0)|(inp.getEndEnergyNo()<0)) {
		if (inp.getFittingParaName().size() == 0) {
			//Input processing energy range from dialog
			inp.setEnergyNoRangeFromDialog("Set energy num range (ex. 1-100).\n");
			if ((inp.getStartEnergyNo() < 0) | (inp.getEndEnergyNo() < 0)) {
				//Input processing parameter name from dialog
				inp.setFittingParaNameFromDialog("Set fitting parameter names (ex. a0,a1,.....)\n");
			}
		}
    }
    

    //processing angle No range
    buffer = output_flag("-ar", argc, argv);
    if (buffer.length()>0) {
        inp.setAngleRange(buffer);
    }
    if ((inp.getStartAngleNo()<0)|(inp.getEndAngleNo()<0)) {
        inp.setAngleRangeFromDialog("Set angle num range (ex. 1-1600).\n");
    }
    
    //check if input data exist
    string fileName_base = inp.getInputDir();
    if (!(inp.getStartEnergyNo()<0)) {
        fileName_base += EnumTagString(inp.getStartEnergyNo(),"/","");
    }else if(inp.getFittingParaName().size()>0){
        fileName_base += "/"+inp.getFittingParaName()[0];
    }
    DIR *dir;
    struct dirent *dp;
    dir=opendir(fileName_base.c_str());
    if (dir==NULL) {
        cout <<"Directory not found."<<endl;
        return -1;
    }
    for(dp=readdir(dir);dp!=NULL;dp=readdir(dir)){
        string Edirname = dp->d_name;
        
        //Image registration by OpenCL pr
        if (Edirname.find("0001.raw")!=string::npos) {
            cout << "    raw file found: " << Edirname <<endl<<endl;
            fileName_base += +"/"+Edirname;
            fileName_base.erase(fileName_base.size()-8);
            fileName_base.erase(0,inp.getInputDir().size()+4);
            break;
        }
    }
    if (dp==NULL) {
        cout <<"No raw file found."<<endl;
        return -1;
    }
    closedir(dir);
    //cout<<fileName_base;
    
    //output directory settings
    buffer = output_flag("-op", argc, argv);
    if (buffer.length()>0) {
        inp.setOutputDir(buffer);
    }
    if (inp.getOutputDir().length()==0) {
        inp.setOutputDirFromDialog("Set output file directory.\n");
    }
    MKDIR(inp.getOutputDir().c_str());
    cout <<endl;
    
    inp.adjustLayerRange();
    
    time_t start,end;
    time(&start);
    
    //reslice
    reslice_ocl(inp,plat_dev_list,fileName_base);
    
    time(&end);
    double delta_t = difftime(end,start);
    int day = (int)floor(delta_t/24/60/60);
    int hour = (int)floor((delta_t-(double)(day*24*60*60))/60/60);
    int min = (int)floor((delta_t-(double)(day*24*60*60+hour*60*60))/60);
    double sec = delta_t-(double)(day*24*60*60+hour*60*60+min*60);
    if (day > 0 ) {
        cout << "process time: "<<day<<" day "<<hour<<" hr "<<min<<" min "<<sec<<" sec"<<endl;
    }else if (hour>0){
        cout <<"process time: "<<hour<<" hr "<<min<<" min "<<sec<<" sec"<<endl;
    }else if (min>0){
        cout <<"process time: "<< min << " min " << sec << " sec"<<endl;
    }else{
        cout <<"process time: "<< sec << " sec"<<endl;
    }
    
    /*cout << endl << "Press 'Enter' to quit." << endl;
    string dummy;
    getline(cin, dummy);*/
    
    return 0;
}
