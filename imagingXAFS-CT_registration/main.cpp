//
//  main.cpp
//  imagingXAFS-CT_registration
//
//  Created by Nozomu Ishiguro on 2016/10/10.
//  Copyright © 2016年 Nozomu Ishiguro. All rights reserved.
//

#include "imagingXAFS-CT_registration.hpp"

mutex m1,m2;
vector<thread> input_th, imageReg_th, output_th;

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
    OCL_platform_device plat_dev_list(inp.getPlatDevList(),true); //true:非共有context
    //inp.setLambda_t(0.00001f);
    inp.setNumtrial(5);
    cout <<endl;
	if (inp.getNumParallel() == NAN) {
		inp.setNumParallel(6);
	}
    
    
    // Input directory settings
    string buffer;
    buffer = output_flag("-ip", argc, argv);
    if (buffer.length()>0) {
        inp.setInputDir(buffer);
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
            cout << "his file found: " << darkname <<endl<<endl;
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
    
    
    /*output directory settings*/
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
    
    
    /*output file base settings*/
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
    
    
    //processing target energy No
    buffer = output_flag("-re", argc, argv);
    if (buffer.length()>0){
        inp.setTargetEnergyNo(buffer);
        cout << "reference energy No. for image registration: "<<endl;
        cout << "   " << inp.getTargetEnergyNo() << endl;
    }
    if (inp.getTargetEnergyNo()<0) {
        inp.setTargetEnergyNoFromDialog("Set reference energy No. for image registration.\n");
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
    
    
    //select regmode
    regMode regmode(0);
    buffer = output_flag("-rm", argc, argv);
    if (buffer.length()>0) {
        inp.setRegMode(buffer);
        
        cout << "Registration mode: "<< inp.getRegMode()<<endl;
        
    }
    if ((inp.getRegMode()==NAN)) {
        inp.setRegModeFromDialog("Set Registration mode. \n\
                                 (0:xy shift, 1:rotation+xy shift, 2:scale+xy shift, \
                                 3:rotation+scale + xy shift, 4:affine + xy shift,-1:none)\n");
        regmode=regMode(inp.getRegMode());
    }
    regmode=regMode(inp.getRegMode());
    
    //set image reg fixed parameter
    if(inp.getReg_fixpara().size()==0){
        inp.setFreeFixParaFromDialog("Input Free(1)/Fix(0) paramater\n");
    }
	regmode.set_pfix(inp);
    
    //set image reg initial parameter
    if(inp.getReg_inipara().size()==0){
        inp.setReg_iniparaFromDialog("Input initial fitting paramater\n");
    }
    for (int i=0; i<min(regmode.get_p_num(), (int)inp.getReg_inipara().size()); i++) {
        regmode.p_ini[i]=inp.getReg_inipara()[i];
    }
    
    
    time_t start,end;
    time(&start);
    
    
    /*Image Registration*/
    imXAFSCT_Registration_ocl(fileName_base,inp,plat_dev_list,regmode);
    
    
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
