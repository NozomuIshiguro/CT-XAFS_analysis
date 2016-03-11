//
//  main.cpp
//  imaging2DXAFS_registration
//
//  Created by Nozomu Ishiguro on 2015/05/28.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "imaging2DXAFS.hpp"

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
    
    OCL_platform_device plat_dev_list(inp.getPlatDevList(),false);
    cout << endl;
    
    
    // Input directory settings
    if (argc>1) {
        inp.setInputDir(output_flag("-ip", argc, argv));
        cout << "raw his file directory."<<endl;
        cout <<inp.getInputDir()<<endl;
    }
    if (inp.getInputDir().length()==0) {
        inp.setInputDirFromDialog("Set input raw his file directory.\n");
    }
    string fileName_base = inp.getInputDir();
    fileName_base += "/";
    
    DIR *dir;
    struct dirent *dp;
    dir=opendir(inp.getInputDir().c_str());
    if (dir==NULL) {
        cout <<"Directory not found."<<endl;
        return -1;
    }
    
    
    for(dp=readdir(dir);dp!=NULL;dp=readdir(dir)){
        string darkname = dp->d_name;
        
        if (darkname.find("dark.his")!=string::npos) {
            cout << "his file found: " << darkname <<endl;
            fileName_base += darkname;
            fileName_base.erase(fileName_base.size()-8);
            break;
        }
    }
    if (dp==NULL) {
        cout <<"No his file found."<<endl;
        return -1;
    }
    closedir(dir);
    //printf("%s\n",fileName_base);
    
    //output directory settings
    if (argc>1) {
        inp.setOutputDir(output_flag("-op", argc, argv));
        cout << "Output file directory."<<endl;
        cout << inp.getOutputDir()<<endl;
    }
    if (inp.getOutputDir().length()==0) {
        inp.setOutputDirFromDialog("Set output file directory.\n");
    }
    MKDIR(inp.getOutputDir().c_str());
    
    //output file base settings
    if (argc>1) {
        inp.setOutputDir(output_flag("-ob", argc, argv));
        cout << "Output file name base."<<endl;
        cout << inp.getOutputFileBase()<<endl;
    }
    if (inp.getOutputFileBase().length()==0) {
        inp.setOutputFileBaseFromDialog("Set output file name base.\n");
    }
    MKDIR(inp.getOutputDir().c_str());

    
    //processing energy No range
    if (argc>1) {
        inp.setEnergyNoRange(output_flag("-enr", argc, argv));
        cout << "Energy num range:"<<endl;
        cout << "   "<<inp.getStartEnergyNo()<<" - "<<inp.getEndEnergyNo()<<endl;
    }
    if ((inp.getStartEnergyNo()==NAN)|(inp.getEndEnergyNo()==NAN)) {
        inp.setEnergyNoRangeFromDialog("Set energy num range (ex. 1-100).\n");
    }
    
    //reference energy No
    if (argc>1) {
        inp.setTargetEnergyNo(output_flag("-re", argc, argv));
        cout << "reference energy No. for image registration: "<<endl;
        cout << "   " << inp.getTargetEnergyNo() << endl;
    }
    if (inp.getTargetEnergyNo()==NAN) {
        inp.setTargetEnergyNoFromDialog("Set reference energy No. for image registration.\n");
    }
    
    //processing loop range (treating angle No. as loop No.)
    if (argc>1) {
        inp.setAngleRange(output_flag("-ar", argc, argv));
        cout << "Loop num range:"<<endl;
        cout << "   "<<inp.getStartAngleNo()<<" - "<<inp.getEndAngleNo()<<endl;
    }
    if ((inp.getStartAngleNo()==NAN)|(inp.getEndAngleNo()==NAN)) {
        inp.setAngleRangeFromDialog("Set loop num range (ex. 1-3).\n");
    }
    
    //reference loop No
    if (argc>1) {
        inp.setTargetAngleNo(output_flag("-rln", argc, argv));
        cout << "Loop num range:"<<endl;
        cout << "   "<<inp.getStartAngleNo()<<" - "<<inp.getEndAngleNo()<<endl;
    }
    if (inp.getTargetAngleNo()==NAN) {
        inp.setTargetAngleNoFromDialog("Set reference loop No. for image registration.\n");
    }
    
    //select regmode
    string buffer;
    regMode regmode(0,1);
    buffer = output_flag("-rm", argc, argv);
    if (buffer.length()>0) {
        inp.setRegMode(buffer);
        regmode=regMode(inp.getRegMode(),1);
        cout << "Registration mode: "<< regmode.get_regModeName()<<"("<< regmode.get_regModeNo()<<")"<<endl;
        
    }else if ((inp.getRegMode()==NAN)) {
        inp.setRegModeFromDialog("Set Registration mode. \n\
                                 (0:xy shift, 1:rotation+xy shift, 2:scale+xy shift, \
                                 3:rotation+scale + xy shift, 4:affine + xy shift,-1:none)\n");
        regmode=regMode(inp.getRegMode(),1);
        
    }else regmode=regMode(inp.getRegMode(),1);
    
    
    time_t start,end;
    time(&start);
    
    
    /*Image Registration*/
    imageRegistlation_2D_ocl(fileName_base,inp,plat_dev_list,regmode);
    
    
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
