//
//  main.cpp
//  imaging2DXAFS_registration
//
//  Created by Nozomu Ishiguro on 2015/05/28.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "imaging2DXAFS.hpp"
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
        //fp_str="/Users/ishiguro/Desktop/ImageReg2.inp";
    }
    
    
    input_parameter inp(fp_str);
    OCL_platform_device plat_dev_list(inp.getPlatDevList(),false);
    cout << endl;
    
    
    // Input file path settings
    string buffer;
    buffer = output_flag("-ip", argc, argv);
    if (buffer.length()>0) {
        inp.setInputDir(buffer);
        cout << "Input raw file path."<<endl;
        cout <<inp.getInputDir()<<endl;
    }
    if (inp.getInputDir().length()==0) {
        inp.setInputDirFromDialog("Set input mt raw file path.\n");
    }
    string fileName_path = inp.getInputDir();
    ifstream inputstream(fileName_path,ios::in|ios::binary);
    if(!inputstream) {
        cerr << "   Failed to load raw file: " <<endl;
        cerr<< fileName_path << endl;
        return -1;
    }
    
    
    //output path settings
    buffer = output_flag("-op", argc, argv);
    if (buffer.length()>0) {
        inp.setOutputDir(buffer);
        cout << "Output file path."<<endl;
        cout << inp.getOutputDir()<<endl;
    }
    if (inp.getOutputDir().length()==0) {
        inp.setOutputDirFromDialog("Set output file path.\n");
    }

    
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
    if(inp.getFreeFixPara().size()>0){
        for (int i=0; i<inp.getFreeFixPara().size(); i++) {
            char a = inp.getFreeFixPara()[i];
            regmode.p_fix[i]=atof(&a);
        }
    }
    
    
    //set image reg initial parameter
    for (int i=0; i<min(regmode.get_p_num(), (int)inp.getReg_inipara().size()); i++) {
        regmode.p_ini[i]=inp.getReg_inipara()[i];
    }
    
    
    time_t start,end;
    time(&start);
    
    
    //Image Registration
    imageRegistlation_2D_ocl(inp,plat_dev_list,regmode);
    
    
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
