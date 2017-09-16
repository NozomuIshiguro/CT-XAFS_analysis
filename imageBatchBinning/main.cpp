//
//  main.cpp
//  imageBatchBinning
//
//  Created by Nozomu Ishiguro on 2017/07/18.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

//#include "CTXAFS.hpp"
#include "imageBatchBinning.hpp"

mutex m1,m2;
vector<thread> input_th, binning_th, output_th;

int main(int argc, const char * argv[]) {
    
    cout << "-----------------------------------------------"<<endl<<endl;
    cout << "               Batch image binning" <<endl<<endl;
    cout << "         First version: Jul. 18th, 2017"<<endl;
    cout << "         Last update: Aug. 17th, 2017"<<endl<<endl;
    cout << "          Created by Nozomu Ishiguro"<<endl<<endl;
    cout << "-----------------------------------------------"<<endl<<endl;
    
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
    OCL_platform_device plat_dev_list(inp.getPlatDevList(),true); //true:非共有context
    cout <<endl;
    
    //number of parallel processing
    if (inp.getNumParallel()<0) {
        inp.setNumParallel(6);
    }
    
    // Input directory settings
    // Input directory settings
    string buffer;
    buffer = output_flag("-ip", argc, argv);
    if (buffer.length()>0) {
        inp.setInputDir(buffer);
    }
    if (inp.getInputDir().length()==0) {
        inp.setInputDirFromDialog("Set input mt raw file directory.\n");
    }
    string fileName_base = inp.getInputDir();
    fileName_base += "/001";
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
            cout << "raw file found: " << Edirname <<endl<<endl;
            fileName_base += +"/"+Edirname;
            fileName_base.erase(fileName_base.size()-8);
            fileName_base.erase(0,inp.getInputDir().size()+4);
            break;
        }
    }
    if (dp==NULL) {
        cout <<"No raw file found."<<endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(-1);
    }
    closedir(dir);
    //cout<<fileName_base;
    inp.setFittingFileBase(fileName_base);
    
    
    //output directory settings
    buffer = output_flag("-op", argc, argv);
    if (buffer.length()>0) {
        inp.setOutputDir(buffer);
        cout << "Output file directory."<<endl;
        cout << inp.getOutputDir()<<endl;
    }
    if (inp.getOutputDir().length()==0) {
        inp.setOutputDirFromDialog("Set output file directory.\n");
    }
    MKDIR(inp.getOutputDir().c_str());
    
    
    //output file base settings
    buffer = output_flag("-ob", argc, argv);
    if (buffer.length()>0) {
        inp.setOutputFileBase(buffer);
        cout << "Output file name base."<<endl;
        cout << inp.getOutputFileBase()<<endl;
    }
    if (inp.getOutputFileBase().length()==0) {
        inp.setOutputFileBaseFromDialog("Set output file name base.\n");
    }
    
    
    //processing energy No range
    buffer = output_flag("-enr", argc, argv);
    if (buffer.length()>0) {
        inp.setEnergyNoRange(buffer);
        cout << "Energy num range:"<<endl;
        cout << "   "<<inp.getStartEnergyNo()<<" - "<<inp.getEndEnergyNo()<<endl;
    }
    if ((inp.getStartEnergyNo()<0)|(inp.getEndEnergyNo()<0)) {
        inp.setEnergyNoRangeFromDialog("Set energy num range (ex. 1-100).\n");
    }
    
    
    //processing angle range
    buffer = output_flag("-ar", argc, argv);
    if (buffer.length()>0) {
        inp.setAngleRange(buffer);
        cout << "Angle num range:"<<endl;
        cout << "   "<<inp.getStartAngleNo()<<" - "<<inp.getEndAngleNo()<<endl;
    }
    if ((inp.getStartAngleNo()<0)|(inp.getEndAngleNo()<0)) {
        inp.setAngleRangeFromDialog("Set angle num range (ex. 1-1600).\n");
    }
    
    
    time_t start,end;
    time(&start);
    
    
    //binning
    imageBatchBinning_OCL(inp,plat_dev_list);
    
    
    time(&end);
    double delta_t = difftime(end,start);
    int day = (int)floor(delta_t/24/60/60);
    int hour = (int)floor((delta_t-(double)(day*24*60*60))/60/60);
    int min = (int)floor((delta_t-(double)(day*24*60*60+hour*60*60))/60);
    double sec = delta_t-(double)(day*24*60*60+hour*60*60+min*60);
    if (day > 0 ) {
        cout << "process time: "<<day<<" day "<<hour<<" hr "<<min<<" min "<<sec<<" sec "<<endl;
    }else if (hour>0){
        cout <<"process time: "<<hour<<" hr "<<min<<" min "<<sec<<" sec "<<endl;
    }else if (min>0){
        cout <<"process time: "<< min << " min " << sec << " sec "<<endl;
    }else{
        cout <<"process time: "<< sec << " sec "<<endl;
    }
    
    /*cout << endl << "Press 'Enter' to quit." << endl;
     string dummy;
     getline(cin,dummy);*/
    
    return 0;
}
