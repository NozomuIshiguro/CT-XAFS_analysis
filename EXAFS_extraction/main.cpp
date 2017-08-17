//
//  main.cpp
//  EXAFS_extraction
//
//  Created by Nozomu Ishiguro on 2017/08/12.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#include "EXAFS_extraction.hpp"
mutex m1,m2;
vector<thread> input_th,fitting_th, output_th_fit;

int main(int argc, const char * argv[]) {
    
    cout << "-----------------------------------------------"<<endl<<endl;
    cout << "            Imaging EXAFS extraction" <<endl<<endl;
    cout << "         First version: Aug. 17th, 2017"<<endl;
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
	dir = opendir(fileName_base.c_str());
	if (dir == NULL) {
		cout << "Directory not found." << endl;
		return -1;
	}
	for (dp = readdir(dir); dp != NULL; dp = readdir(dir)) {
		string Edirname = dp->d_name;

		//Image registration by OpenCL pr
		if (Edirname.find("0001.raw") != string::npos) {
			cout << "    raw file found: " << Edirname << endl << endl;
			fileName_base += +"/" + Edirname;
			fileName_base.erase(fileName_base.size() - 8);
			fileName_base.erase(0, inp.getInputDir().size() + 4);
			break;
		}
	}
	if (dp == NULL) {
		cout << "No raw file found." << endl;
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
    
    
    //processing angle range
    buffer = output_flag("-ar", argc, argv);
    if (buffer.length()>0) {
        inp.setAngleRange(buffer);
    }
    if ((inp.getStartAngleNo()<0)|(inp.getEndAngleNo()<0)) {
        inp.setAngleRangeFromDialog("Set angle num range (ex. 1-1600).\n");
    }
    cout <<endl;
    
    
    inp.setLambda_t_fit(0.01f);
    
    
    time_t start,end;
    time(&start);
    
    //EXAFS extraction
    EXAFS_extraction_ocl(inp,plat_dev_list);
    
    
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
    
    
    return 0;
}
