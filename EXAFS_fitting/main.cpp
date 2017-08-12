//
//  main.cpp
//  EXAFS_fitting
//
//  Created by Nozomu Ishiguro on 2016/01/04.
//  Copyright © 2016年 Nozomu Ishiguro. All rights reserved.
//

#include "EXAFS.hpp"

mutex m1,m2;
vector<thread> input_th,fitting_th, output_th_fit;

int main2(int argc, const char * argv[]) {
    string fp_str;
    if (argc>1) {
        fp_str=argv[1];
    }else{
        /*string dummy;
        cout<<"Set input file path, if existed."<<endl;
        getline(cin,dummy);
        istringstream iss(dummy);
        iss>>fp_str;*/
        fp_str="/Users/ishiguro/Desktop/XAFS_tools/feff_inp/Pt_feff/feff0001.dat";
    }
    
    OCL_platform_device plat_dev_list("2"/*inp.getPlatDevList()*/,false);
    
    vector<FEFF_shell> shell;
    shell.push_back(FEFF_shell::FEFF_shell(fp_str));
    testEXAFS(shell,plat_dev_list);
    
    return 0;
}


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
        inp.setInputDirFromDialog("Set input chi raw file directory.\n");
    }
    string fileName_base = inp.getInputDir();
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
            fileName_base.erase(0,inp.getInputDir().size()+1);
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

	string fileName_base_ej = inp.getInputDir_EJ();
	dir = opendir(fileName_base_ej.c_str());
	if (dir == NULL) {
		cout << "Directory not found." << endl;
		return -1;
	}
	for (dp = readdir(dir); dp != NULL; dp = readdir(dir)) {
		string Edirname = dp->d_name;

		//Image registration by OpenCL pr
		if (Edirname.find("0001.raw") != string::npos) {
			cout << "    raw file found: " << Edirname << endl << endl;
			fileName_base_ej += +"/" + Edirname;
			fileName_base_ej.erase(fileName_base_ej.size() - 8);
			fileName_base_ej.erase(0, inp.getInputDir_EJ().size() + 1);
			break;
		}
	}
	if (dp == NULL) {
		cout << "No raw file found." << endl;
		return -1;
	}
	closedir(dir);
	//cout<<fileName_base;
	inp.setEJFileBase(fileName_base_ej);
    
    
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
    
    
    //processing angle range
    buffer = output_flag("-ar", argc, argv);
    if (buffer.length()>0) {
        inp.setAngleRange(buffer);
    }
    if ((inp.getStartAngleNo()<0)|(inp.getEndAngleNo()<0)) {
        inp.setAngleRangeFromDialog("Set angle num range (ex. 1-1600).\n");
    }
    cout <<endl;
    
    
    //input feffxxxx.dat files
    vector<FEFF_shell> shell;
    for (int i=0; i<inp.getShellNum(); i++) {
        shell.push_back(FEFF_shell(inp.getFeffxxxxdatPath()[i]));
    }

	inp.setLambda_t_fit(0.1f);
    
    
    time_t start,end;
    time(&start);
    
    //EXAFS fitting
    EXAFS_fit_ocl(inp,plat_dev_list,shell);
    
    
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
