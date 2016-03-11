//
//  main.cpp
//  CT_imageRegistration_multi
//
//  Created by Nozomu Ishiguro on 2015/02/11.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "CTXAFS.hpp"


int main(int argc, const char * argv[]) {
    
    string fp_str;
    if (argc>1) {
        fp_str=argv[1];
    }else{
        cout<<"Set inut file path, if existed."<<endl;
        getline(cin, fp_str);
        //fp_str="./image_reg.inp";
    }
    
    
    
    input_parameter inp(fp_str);
    
    OCL_platform_device plat_dev_list(inp.getPlatDevList());
    cout << "\n";
    
    
    /* Input directory settings*/
    if (argc>1) {
        inp.setInputDir(output_flag("-ip", argc, argv));
    }
    if (inp.getOutputDir().length()==0) {
            inp.setOutputDirFromDialog("Set input raw his file directory.\n");
    }else{
        cout << "raw his file directory.\n";
        cout <<inp.getInputDir()<<endl;
    }
    string fileName_base = inp.getInputDir();
    fileName_base += "/";
    
    DIR *dir;
    struct dirent *dp;
    dir=opendir(inp.getInputDir().c_str());
    if (dir==NULL) {
        cout <<"Directory not found.\n\n";
        return -1;
    }
    
    
    for(dp=readdir(dir);dp!=NULL;dp=readdir(dir)){
        string darkname = dp->d_name;
        
        if (darkname.find("dark.his")!=string::npos) {
            cout << "    his file found: " << darkname << "\n\n";
            fileName_base += darkname;
            fileName_base.erase(fileName_base.size()-8);
            break;
        }
    }
    if (dp==NULL) {
        cout <<"No his file found.\n\n";
        return -1;
    }
    closedir(dir);
    //printf("%s\n",fileName_base);
    
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
    
    //processing energy No range
    if (argc>1) {
        inp.setEnergyNoRange(output_flag("-enr", argc, argv));
    }
    if ((inp.getStartEnergyNo()==NAN)|(inp.getEndEnergyNo()==NAN)) {
        inp.setEnergyNoRangeFromDialog("Set energy num range (ex. 1-100).");
    }else{
        cout << "Energy num range:"<<endl;
        cout << "   "<<inp.getStartEnergyNo()<<" - "<<inp.getEndEnergyNo()<<endl;
    }
    cout<<endl;
    
    if (inp.getTargetEnergyNo()==NAN) {
        inp.setTargetEnergyNoFromDialog("Set reference energy No. for image registration.");
    }else{
        cout << "reference energy No. for image registration: "<<endl;
		cout << "   " << inp.getTargetEnergyNo() << endl;
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
    
    
    time_t start,end;
    time(&start);
    
    
    /*Image Registration*/
    imageRegistlation_ocl(fileName_base,inp,plat_dev_list);
    
    
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
	getline(cin,dummy);
    
    return 0;
}
