//
//  main.cpp
//  CT_imageReg_XANESfitting
//
//  Created by Nozomu Ishiguro on 2015/11/21.
//  Copyright © 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "CTXAFS.hpp"
#include "XANES_fitting.hpp"
#include "atan_lor_linear_fitting.hpp"

mutex m1,m2;
vector<thread> input_th, imageReg_th, fitting_th, output_th, output_th_fit;
fitting_eq fiteq(atanlorlinear_preprocessor1,atanlorlinear_preprocessor2);

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
    inp.setNumtrial(5); //only image Reg
    cout <<endl;
    
    if (inp.getNumParallel()==NAN) {
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
    
    
    //image Reg output directory settings
    buffer = output_flag("-op", argc, argv);
    if (buffer.length()>0) {
        inp.setOutputDir(buffer);
        cout << "Image reg output file directory."<<endl;
        cout << inp.getOutputDir()<<endl;
    }
    if (inp.getOutputDir().length()==0) {
        inp.setOutputDirFromDialog("Set image reg output file directory.\n");
    }
    MKDIR(inp.getOutputDir().c_str());
    
    
    //image Reg output file base settings
    buffer = output_flag("-ob", argc, argv);
    if (buffer.length()>0) {
        inp.setOutputFileBase(buffer);
        cout << "Image reg output file name base."<<endl;
        cout << inp.getOutputFileBase()<<endl;
    }
    if (inp.getOutputFileBase().length()==0) {
        inp.setOutputFileBaseFromDialog("Set image reg output file name base.\n");
    }
    
    //XANES fitting output directory settings
    buffer = output_flag("-xfop", argc, argv);
    if (buffer.length()>0) {
        inp.setFittingOutputDir(buffer);
        cout << "Output file directory."<<endl;
        cout << inp.getFittingOutputDir()<<endl;
    }
    if (inp.getFittingOutputDir().length()==0) {
        inp.setFittingOutputDirFromDialog("Set XANES fitting output file directory.\n");
    }
    MKDIR(inp.getFittingOutputDir().c_str());
    
    /*XANES fitting output file base settings*/
    /*buffer = output_flag("-ob", argc, argv);
    if (buffer.length()>0) {
        inp.setFittingFileBase(buffer);
        cout << "Image reg output file name base."<<endl;
        cout << inp.getFittingFileBase()<<endl;
    }
    if (inp.getFittingFileBase().length()==0) {
        inp.setFittingFileBaseFromDialog("Set image reg output file name base.\n");
    }*/
    
    //processing energy No range
    buffer = output_flag("-enr", argc, argv);
    if (buffer.length()>0) {
        inp.setEnergyNoRange(buffer);
        cout << "Energy num range:"<<endl;
        cout << "   "<<inp.getStartEnergyNo()<<" - "<<inp.getEndEnergyNo()<<endl;
    }
    if ((inp.getStartEnergyNo()==NAN)|(inp.getEndEnergyNo()==NAN)) {
        inp.setEnergyNoRangeFromDialog("Set energy num range (ex. 1-100).\n");
    }
    
    
    //processing target energy No
    buffer = output_flag("-re", argc, argv);
    if (buffer.length()>0){
        inp.setTargetEnergyNo(buffer);
        cout << "reference energy No. for image registration: "<<endl;
        cout << "   " << inp.getTargetEnergyNo() << endl;
    }
    if (inp.getTargetEnergyNo()==NAN) {
        inp.setTargetEnergyNoFromDialog("Set reference energy No. for image registration.\n");
    }
    
    //energy file path
    buffer = output_flag("-ep", argc, argv);
    if (buffer.length()>0) {
        inp.setEnergyFilePath(buffer);
    }
    if (inp.getEnergyFilePath().length()==0) {
        inp.setEnergyFilePathFromDialog("Set energy file path.");
    }
    ifstream energy_ifs(inp.getEnergyFilePath(),ios::in);
    if(!energy_ifs) {
        cerr << "   Failed to load energy file" << endl;
        return -1;
    }else{
        cout<<"   Energy file found."<<endl<<endl;
    }
    
    
    //processing E0
    buffer = output_flag("-e0", argc, argv);
    if (buffer.length()>0) {
        inp.setE0(buffer);
    }
    if (inp.getE0()==NAN) {
        inp.setE0FromDialog("Set E0 /eV (ex. 11559).");
    }
    
    
    //processing energy range
    buffer = output_flag("-er", argc, argv);
    if (buffer.length()>0) {
        inp.setEnergyRange(buffer);
    }
    if ((inp.getStartEnergy()==NAN)|(inp.getEndEnergy()==NAN)) {
        inp.setEnergyRangeFromDialog("Set fitting energy range /eV (ex. 11540.0-11600.0).");
    }
    cout<<endl;
    
    
    //processing angle range
    buffer = output_flag("-ar", argc, argv);
    if (buffer.length()>0) {
        inp.setAngleRange(buffer);
        cout << "Angle num range:"<<endl;
        cout << "   "<<inp.getStartAngleNo()<<" - "<<inp.getEndAngleNo()<<endl;
    }
    if ((inp.getStartAngleNo()==NAN)|(inp.getEndAngleNo()==NAN)) {
        inp.setAngleRangeFromDialog("Set angle num range (ex. 1-1600).\n");
    }
    
    
    //select regmode
    regMode regmode(0,1);
    buffer = output_flag("-rm", argc, argv);
    if (buffer.length()>0) {
        inp.setRegMode(buffer);
        
        cout << "Registration mode: "<< inp.getRegMode()<<endl;
        
    }
    buffer = output_flag("-cm", argc, argv);
    if (buffer.length()>0) {
        inp.setCntMode(buffer);
        regmode=regMode(inp.getRegMode(),2);
        cout << "Registration contrast factor mode: "<< inp.getCntMode()<<endl;
    }
    if ((inp.getRegMode()==NAN)) {
        inp.setRegModeFromDialog("Set Registration mode. \n\
                                 (0:xy shift, 1:rotation+xy shift, 2:scale+xy shift, \
                                 3:rotation+scale + xy shift, 4:affine + xy shift,-1:none)\n");
        regmode=regMode(inp.getRegMode(),2);
    }
    regmode=regMode(inp.getRegMode(),inp.getCntMode());
    
    //fitting parameter setting
    buffer = output_flag("-fitp", argc, argv);
    if (buffer.length()>0) {
        inp.setFittingPara(buffer);
    }
    if (inp.getFittingPara().size()==0) {
        inp.setFittingParaFromDialog("Set Initial fitting parameters (ex. 1.1,0.2,.....)");
        cout<<endl;
    }
    
    
    //free/fix parameter setting
    buffer = output_flag("-freep", argc, argv);
    if (buffer.length()>0) {
        inp.setFreeFixPara(buffer);
    }
    if (inp.getFreeFixPara().size()==0) {
        inp.setFreeFixParaFromDialog("Set Free(1)/Fix(0) of fitting parameters (ex. 1,1,0,.....)");
        cout<<endl;
    }
    
    
    //Valid Para Lower/Upper Limit setting
    buffer = output_flag("-vpl", argc, argv);
    if (buffer.length()>0) {
        inp.setValidParaLimit(buffer);
    }
    if (inp.getFreeFixPara().size()==0) {
        inp.setValidParaLowerLimitFromDialog("Set valid parameter lower limit (ex. 0,-1.0,0,.....)");
        cout<<endl;
        
        inp.setValidParaLowerLimitFromDialog("Set valid parameter upper limit (ex. 1.0,1.0,10.0,.....)");
        cout<<endl;
    }
    
    
    //parameter name
    buffer = output_flag("-fpn", argc, argv);
    if (buffer.length()>0) {
        inp.setFittingParaName(buffer);
    }
    if (inp.getFittingParaName().size()==0) {
        inp.setFittingParaNameFromDialog("Set fitting parameter names (ex. a0,a1,.....)");
    }
    
    fiteq=fitting_eq(inp, atanlorlinear_preprocessor1,atanlorlinear_preprocessor2);
    
    //set image reg initial parameter
	for (int i=0; i<min(regmode.get_p_num()+regmode.get_cp_num(), (int)inp.getRegCnt_inipara().size()); i++) {
        regmode.p_ini[i]=inp.getRegCnt_inipara()[i];
    }
    
    time_t start,end;
    time(&start);
    
    
    //Image Registration
    imageRegistlation_ocl(fileName_base,inp,plat_dev_list,regmode);
    
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
