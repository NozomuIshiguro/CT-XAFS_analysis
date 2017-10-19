//
//  main.cpp
//  imgReg_XANESfit
//
//  Created by Nozomu Ishiguro on 2017/09/28.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#include "imgReg_XANESfit.hpp"

mutex m1,m2;
vector<thread> input_th,fitting_th, output_th_fit;

int main(int argc, const char * argv[]) {
    
    cout << "-----------------------------------------------"<<endl<<endl;
    cout << "            Imaging XANES fittting" <<endl<<endl;
    cout << "         First version: Jan. 13th, 2015"<<endl;
    cout << "         Last update: Sep. 14th, 2017"<<endl<<endl;
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
        //fp_str="C:/Users/CT/Desktop/XANES_fitting.inp";
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
    string fileName_base = inp.getInputDir();
    fileName_base += "/001";
    DIR *dir;
    struct dirent *dp;
    dir=opendir(fileName_base.c_str());
    if (dir==NULL) {
        cout <<"Directory not found."<<endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(-1);
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
        return -1;
    }
    closedir(dir);
    //cout<<fileName_base;
    inp.setFittingFileBase(fileName_base);
    
    
    //output directory settings
    buffer = output_flag("-op", argc, argv);
    if (buffer.length()>0) {
        inp.setFittingOutputDir(buffer);
    }
    if (inp.getFittingOutputDir().length()==0) {
        if (inp.getOutputDir().length() == 0) {
            inp.setFittingOutputDirFromDialog("Set output file directory.\n");
        }
    }
    //cout << inp.getFittingOutputDir() << endl;
    MKDIR(inp.getFittingOutputDir().c_str());
    cout <<endl;
    
    
    //energy file path
    buffer = output_flag("-ep", argc, argv);
    if (buffer.length()>0) {
        inp.setEnergyFilePath(buffer);
    }
    if (inp.getEnergyFilePath().length()==0) {
        inp.setEnergyFilePathFromDialog("Set energy file path.\n");
    }
    ifstream energy_ifs(inp.getEnergyFilePath(),ios::in);
    if(!energy_ifs) {
        cerr << "Failed to load energy file" << endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(-1);
    }
    
    
    //processing E0
    buffer = output_flag("-e0", argc, argv);
    if (buffer.length()>0) {
        inp.setE0(buffer);
    }
    if (inp.getE0()==NAN) {
        inp.setE0FromDialog("Set E0 /eV (ex. 11559).\n");
    }else{
        cout << "E0 = ";
        cout << inp.getE0()<<" eV."<<endl;
    }
    cout<<endl;
    
    
    //processing energy range
    buffer = output_flag("-er", argc, argv);
    if (buffer.length()>0) {
        inp.setEnergyRange(buffer);
    }
    if ((inp.getStartEnergy()==NAN)|(inp.getEndEnergy()==NAN)) {
        inp.setEnergyRangeFromDialog("Set fitting energy range /eV (ex. 11540.0-11600.0).\n");
    }
    cout<<endl;
    
    
    //processing angle range
    buffer = output_flag("-ar", argc, argv);
    if (buffer.length()>0) {
        inp.setAngleRange(buffer);
    }
    if ((inp.getStartAngleNo()<0)|(inp.getEndAngleNo()<0)) {
        inp.setAngleRangeFromDialog("Set angle num range (ex. 1-1600)\n.");
    }
    cout <<endl;
    
    
    //fitting parameter setting
    buffer = output_flag("-fitp", argc, argv);
    if (buffer.length()>0) {
        inp.setFittingPara(buffer);
    }
    if (inp.getFittingPara().size()==0) {
        inp.setFittingParaFromDialog("Set Initial fitting parameters (ex. 1.1,0.2,.....)\n");
        cout<<endl;
    }
    
    
    //free/fix parameter setting
    buffer = output_flag("-freep", argc, argv);
    if (buffer.length()>0) {
        inp.setFreeFixPara(buffer);
    }
    if (inp.getFreeFixPara().size()==0) {
        inp.setFreeFixParaFromDialog("Set Free(1)/Fix(0) of fitting parameters (ex. 1,1,0,.....)\n");
        cout<<endl;
    }
    
    
    //Valid Para Lower/Upper Limit setting
    buffer = output_flag("-vpl", argc, argv);
    if (buffer.length()>0) {
        inp.setValidParaLimit(buffer);
    }
    if (inp.getFreeFixPara().size()==0) {
        inp.setValidParaLowerLimitFromDialog("Set valid parameter lower limit (ex. 0,-1.0,0,.....)\n");
        cout<<endl;
        
        inp.setValidParaLowerLimitFromDialog("Set valid parameter upper limit (ex. 1.0,1.0,10.0,.....)\n");
        cout<<endl;
    }
    
    
    //parameter name
    buffer = output_flag("-fpn", argc, argv);
    if (buffer.length()>0) {
        inp.setFittingParaName(buffer);
    }
    if (inp.getFittingParaName().size()==0) {
        inp.setFittingParaNameFromDialog("Set fitting parameter names (ex. a0,a1,.....)\n");
    }
    
    fitting_eq fiteq(inp);
    
    time_t start,end;
    time(&start);
    
    //XANES fitting
    XANES_fit_ocl(fiteq, inp, plat_dev_list);
    
    
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
