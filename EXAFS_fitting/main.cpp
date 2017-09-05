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

int main(int argc, const char * argv[]) {
    
    /*vector<string> contrain_eqs;
    vector<string> fparaName;
    vector<vector<float>> C_matrix;
    vector<float> D_vector;
    contrain_eqs.push_back("CN_Pt-Pt - 3*CN_Pt-O = 12 - 2*CN_Pt-Co");
    contrain_eqs.push_back("R_Pt-Pt > 2.70");
    fparaName.push_back("S02");
    fparaName.push_back("CN_Pt-Pt");
    fparaName.push_back("CN_Pt-O");
    fparaName.push_back("CN_Pt-Co");
    fparaName.push_back("R_Pt-Pt");
    int num_contrain = createContrainMatrix(contrain_eqs,fparaName,&C_matrix,&D_vector);
    int num_fpara = (int)fparaName.size();
    correctBondDistanceContrain(&C_matrix, &D_vector,4,2.7713f);
    cout<<endl<<"C_matrix | D_vector"<<endl;
    for (int i=0; i<num_contrain; i++) {
        for (int j=0; j<num_fpara-1; j++) {
            cout << C_matrix[i][j] <<" ";
        }
        cout<< C_matrix[i][num_fpara-1] <<" | "<< D_vector[i] << endl;
    }*/
    
    cout << "-----------------------------------------------"<<endl<<endl;
    cout << "             Imaging EXAFS fittting" <<endl<<endl;
    cout << "         First version: Aug. 10th, 2017"<<endl;
    cout << "         Last update: Sep. 5th, 2017"<<endl<<endl;
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
        inp.setInputDirFromDialog("Set input chi raw file directory.\n");
    }
    string fileName_base = inp.getInputDir();
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
            cout << "    raw file found: " << Edirname <<endl<<endl;
            fileName_base += +"/"+Edirname;
            fileName_base.erase(fileName_base.size()-8);
            fileName_base.erase(0,inp.getInputDir().size()+1);
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

	string fileName_base_ej = inp.getInputDir_EJ();
	dir = opendir(fileName_base_ej.c_str());
	if (dir == NULL) {
		cout << "Directory not found." << endl;
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(-1);
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
        cout <<  "Press 'Enter' to quit." << endl;
        string dummy;
        getline(cin,dummy);
        exit(-1);
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
