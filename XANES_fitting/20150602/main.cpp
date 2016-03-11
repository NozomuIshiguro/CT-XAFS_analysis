//
//  main.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/01/13.
//  Copyright (c) 2014-2015 Nozomu Ishiguro. All rights reserved.
//

#include "XANES_fitting.hpp"
#include "atan_lor_linear_fitting.hpp"

int main(int argc, const char * argv[]) {
    string fp_str;
    if (argc>1) {
        fp_str=output_flag("-p", argc, argv);
    }else{
        cout<<"Set inut file path, if existed."<<endl;
        getline(cin, fp_str);
        //fp_str="./image_reg.inp";
    }
    
    input_parameter inp(fp_str);
    
    OCL_platform_device plat_dev_list(inp.getPlatDevList());
    cout << endl;
    
    /* Input directory settings*/
    if (argc>1) {
        inp.setInputDir(output_flag("-ip", argc, argv));
    }
    if (inp.getInputDir().length()==0) {
        inp.setInputDirFromDialog("Set input mt raw file directory.");
    }else{
        cout << "mt raw file directory.\n";
        cout <<inp.getInputDir()<<endl;
    }
    string fileName_base = inp.getInputDir();
    fileName_base += "/001";
    
    DIR *dir;
    struct dirent *dp;
    dir=opendir(fileName_base.c_str());
    if (dir==NULL) {
        cout <<"Directory not found.\n\n";
        return -1;
    }
    for(dp=readdir(dir);dp!=NULL;dp=readdir(dir)){
        string Edirname = dp->d_name;
        
        //Image registration by OpenCL pr
        if (Edirname.find("0001.raw")!=string::npos) {
            cout << "    raw file found: " << Edirname << "\n\n";
            fileName_base += +"/"+Edirname;
            fileName_base.erase(fileName_base.size()-8);
            fileName_base.erase(0,inp.getInputDir().size()+4);
            break;
        }
    }
    if (dp==NULL) {
        cout <<"No raw file found.\n\n";
        return -1;
    }
    closedir(dir);
    //cout<<fileName_base;
    
    
    
    /*output directory settings*/
    if (argc>1) {
        inp.setOutputDir(output_flag("-op", argc, argv));
    }
    if (inp.getOutputDir().length()==0) {
        inp.setOutputDirFromDialog("Set output file directory.");
    }else{
        cout << "Output file directory.\n";
        cout << inp.getOutputDir()<<endl;
    }
    MKDIR(inp.getOutputDir().c_str());
    cout << "\n";
    
    //energy file path
    if (argc>1) {
        inp.setEnergyFilePath(output_flag("-ep", argc, argv));
    }
    if (inp.getEnergyFilePath().length()==0) {
        inp.setEnergyFilePathFromDialog("Set energy file path.");
    }else{
        cout << "Energy file path.\n";
        cout << inp.getEnergyFilePath()<<endl;
    }
    ifstream energy_ifs(inp.getEnergyFilePath(),ios::in);
    if(!energy_ifs) {
        cerr << "   Failed to load energy file \n\n" << endl;
        return -1;
    }else{
        cout<<"   Energy file found.\n\n";
    }
    
    
    //processing E0
    if (argc>1) {
        inp.setE0(output_flag("-e0", argc, argv));
    }
    if (inp.getE0()==NAN) {
        inp.setE0FromDialog("Set E0 /eV (ex. 11559).");
    }else{
        cout << "E0 = ";
        cout << inp.getE0()<<" eV."<<endl;
    }
    cout<<endl;
    
    //processing energy range
    if (argc>1) {
        inp.setEnergyRange(output_flag("-er", argc, argv));
    }
    if ((inp.getStartEnergy()==NAN)|(inp.getEndEnergy()==NAN)) {
        inp.setEnergyRangeFromDialog("Set fitting energy range /eV (ex. 11540.0-11600.0).");
    }else{
        cout << "Energy range:"<<endl;
        cout << "   "<< inp.getStartEnergy() <<" - "<<inp.getEndEnergy()<<" eV."<<endl;
    }
    cout<<endl;
    
    //processing angle range
    if (argc>1) {
        inp.setAngleRange(output_flag("-ar", argc, argv));
    }
    if ((inp.getStartAngleNo()==NAN)|(inp.getEndAngleNo()==NAN)) {
        inp.setAngleRangeFromDialog("Set angle num range (ex. 1-1600).");
    }else{
        cout << "Angle num range:"<<endl;
        cout << "   "<<inp.getStartAngleNo()<<" - "<<inp.getEndAngleNo()<<endl;
    }
    cout << "\n";
    
    
    //fitting parameter setting
    if (argc>1) {
        inp.setFittingPara(output_flag("-fitp", argc, argv));
    }
    if (inp.getFittingPara().size()==0) {
        inp.setFittingParaFromDialog("Set Initial fitting parameters (ex. 1.1,0.2,.....)");
        cout<<endl;
    }else{
        cout<<"Set Initial fitting parameters"<<endl<<"   ";
         for (int i=0; i<inp.getFittingPara().size(); i++) {
             cout.width(6);
             cout<<inp.getFittingPara()[i];
             if(i+1!=inp.getFittingPara().size()) cout<<",";
         }
        cout<<endl;
    }
    
    //free/fix parameter setting
    if (argc>1) {
        inp.setFreeFixPara(output_flag("-freep", argc, argv));
    }
    if (inp.getFreeFixPara().size()==0) {
        inp.setFreeFixParaFromDialog("Set Free(1)/Fix(0) of fitting parameters (ex. 1,1,0,.....)");
        cout<<endl;
    }else{
        cout<<"Free(1)/Fix(0) of fitting parameters"<<endl<<"   ";
        for (int i=0; i<inp.getFreeFixPara().size(); i++) {
            cout.width(6);
            cout<<inp.getFreeFixPara()[i];
            if(i+1!=inp.getFreeFixPara().size()) cout<<",";
        }
        cout<<endl;
    }
    
    //Valid Para Lower/Upper Limit setting
    if (argc>1) {
        inp.setValidParaLimit(output_flag("-vpl", argc, argv));
    }
    if (inp.getFreeFixPara().size()==0) {
        inp.setValidParaLowerLimitFromDialog("Set valid parameter lower limit (ex. 0,-1.0,0,.....)");
        cout<<endl;
        
        inp.setValidParaLowerLimitFromDialog("Set valid parameter upper limit (ex. 1.0,1.0,10.0,.....)");
        cout<<endl;
    }else{
        cout<<"Valid parameter ranges"<<endl<<"   ";
        for (int i=0; i<inp.getParaLowerLimit().size(); i++) {
            cout.width(6);
            cout<<inp.getParaLowerLimit()[i]<<"-"<<inp.getParaUpperLimit()[i];
            if(i+1!=inp.getParaLowerLimit().size()) cout<<",";
        }
        cout<<endl;
    }
    
    
    //parameter name
    if (argc>1) {
        inp.setFittingParaName(output_flag("-fpn", argc, argv));
    }
    if (inp.getFittingParaName().size()==0) {
        inp.setFittingParaNameFromDialog("Set fitting parameter names (ex. a0,a1,.....)");
    }else{
        cout<<"Fitting parameters name"<<endl<<"   ";
        for (int i=0; i<inp.getFreeFixPara().size(); i++) {
            cout.width(6);
            cout<<inp.getFittingParaName()[i];
            if(i+1!=inp.getFittingPara().size()) cout<<",";
        }
        cout<<endl;
    }
    cout<<endl;
    
    fitting_eq fiteq(inp, atanlorlinear_preprocessor);
    
    
    time_t start,end;
    time(&start);
    
    
    //XANES fitting
    XANES_fit_ocl(fiteq, inp, plat_dev_list,fileName_base);
    
    
    time(&end);
    double delta_t = difftime(end,start);
    int day = (int)floor(delta_t/24/60/60);
    int hour = (int)floor((delta_t-(double)(day*24*60*60))/60/60);
    int min = (int)floor((delta_t-(double)(day*24*60*60+hour*60*60))/60);
    double sec = delta_t-(double)(day*24*60*60+hour*60*60+min*60);
    if (day > 0 ) {
        cout << "process time: "<<day<<" day "<<hour<<" hr "<<min<<" min "<<sec<<" sec \n";
    }else if (hour>0){
        cout <<"process time: "<<hour<<" hr "<<min<<" min "<<sec<<" sec \n";
    }else if (min>0){
        cout <<"process time: "<< min << " min " << sec << " sec \n";
    }else{
        cout <<"process time: "<< sec << " sec \n";
    }
    
	cout << endl << "Press 'Enter' to quit." << endl;
	string dummy;
	getline(cin, dummy);

    return 0;
}