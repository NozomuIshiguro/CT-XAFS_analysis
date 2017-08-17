//
//  input_parameter_fitting_class.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/11/22.
//  Copyright © 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"

static int buffsize=512;

int input_parameter_fitting::getPreEdgeFittingMode(){
    return bkgFittingMode;
}

float input_parameter_fitting::getRbkg(){
    return Rbkg;
}

int input_parameter_fitting::getPreEdgeStartEnergyNo(){
    return preEdgeStartEnergyNo;
}

int input_parameter_fitting::getPreEdgeEndEnergyNo(){
    return preEdgeEndEnergyNo;
}

int input_parameter_fitting::getPostEdgeStartEnergyNo(){
    return postEdgeStartEnergyNo;
}

int input_parameter_fitting::getPostEdgeEndEnergyNo(){
    return postEdgeEndEnergyNo;
}

float input_parameter_fitting::getPreEdgeStartEnergy(){
    return preEdgeStartEnergy;
}

float input_parameter_fitting::getPreEdgeEndEnergy(){
    return preEdgeEndEnergy;
}

float input_parameter_fitting::getPostEdgeStartEnergy(){
    return postEdgeStartEnergy;
}

float input_parameter_fitting::getPostEdgeEndEnergy(){
    return postEdgeEndEnergy;
}


void input_parameter_fitting::setPreEdgeEnergyNoRange(int startEnergyNo,int endEnergyNo){
    preEdgeStartEnergyNo = startEnergyNo;
    preEdgeEndEnergyNo = endEnergyNo;
}

void input_parameter_fitting::setPostEdgeEnergyNoRange(int startEnergyNo,int endEnergyNo){
    postEdgeStartEnergyNo = startEnergyNo;
    postEdgeEndEnergyNo = endEnergyNo;
}
void input_parameter_fitting::setFittingEnergyNoRange(int startEnergyNo,int endEnergyNo){
    fittingStartEnergyNo = startEnergyNo;
    fittingEndEnergyNo = endEnergyNo;
}

//get
string input_parameter_fitting::getFittingOutputDir(){
    return fitting_output_dir;
}

string input_parameter_fitting::getFittingFileBase(){
    return fitting_filebase;
}

string input_parameter_fitting::getEJFileBase() {
	return ej_filebase;
}

int input_parameter_fitting::getFittingStartEnergyNo(){
    return fittingStartEnergyNo;
}

int input_parameter_fitting::getFittingEndEnergyNo(){
    return fittingEndEnergyNo;
}

string input_parameter_fitting::getEnergyFilePath(){
    return energyFilePath;
}

float input_parameter_fitting::getE0(){
    return E0;
}

float input_parameter_fitting::getStartEnergy(){
    return startEnergy;
}

float input_parameter_fitting::getEndEnergy(){
    return endEnergy;
}

vector<float> input_parameter_fitting::getFittingPara(){
    return  fitting_para;
}

vector<bool> input_parameter_fitting::getFreeFixPara(){
    return free_para;
}

vector<string> input_parameter_fitting::getFittingParaName(){
    return parameter_name;
}

vector<float> input_parameter_fitting::getParaUpperLimit(){
    return  para_upperLimit;
}

vector<float> input_parameter_fitting::getParaLowerLimit(){
    return para_lowerLimit;
}

vector<float> input_parameter_fitting::getParaAttenuator(){
    return para_attenuator;
}

bool input_parameter_fitting::getCSbool(){
    return CSbool;
}

vector<float> input_parameter_fitting::getCSepsilon(){
    return CSepsilon;
}

vector<float> input_parameter_fitting::getCSalpha(){
    return CSalpha;
}

int input_parameter_fitting::getCSit(){
    return CSit;
}


//set from dialog
void input_parameter_fitting::setFittingOutputDirFromDialog(string message){
    cout << message;
    cin >> fitting_output_dir;
}

void input_parameter_fitting::setFittingFileBaseFromDialog(string message){
    cout << message;
    cin >> fitting_filebase;
}

void input_parameter_fitting::setEnergyFilePathFromDialog(string message){
    cout << message;
    cin >> energyFilePath;
}

void input_parameter_fitting::setE0FromDialog(string message){
    cout << message;
    cin >> E0;
}

void input_parameter_fitting::setEnergyRangeFromDialog(string message){
    cout << message;
    cin >> startEnergy; cin.ignore() >> endEnergy;
}

void input_parameter_fitting::setFittingParaFromDialog(string message){
    cout << message;
    char *buffer;
    buffer = new char[buffsize];
    cin.getline(buffer, buffsize);
    istringstream iss(buffer);
    float a;
    for (iss>>a; !iss.eof(); iss.ignore()>>a) {
        fitting_para.push_back(a);
    }
    fitting_para.push_back(a);
}

void input_parameter_fitting::setFreeFixParaFromDialog(string message){
    cout << message;
    char *buffer;
    buffer = new char[buffsize];
    cin.getline(buffer, buffsize);
    istringstream iss(buffer);
    char b;
    for (iss>>b; !iss.eof(); iss.ignore()>>b) {
        if (atoi(&b)==1) {
            free_para.push_back(true);
        }else{
            free_para.push_back(false);
        }
    }
    if (atoi(&b)==1) {
        free_para.push_back(true);
    }else{
        free_para.push_back(false);
    }
    
    while (free_para.size()<fitting_para.size()) {
        free_para.push_back(true);
    }
}

void input_parameter_fitting::setFittingParaNameFromDialog(string message){
    cout << message;
    char *buffer;
    buffer = new char[buffsize];
    cin.getline(buffer, buffsize);
    istringstream iss(buffer);
    for (;!iss.eof();) {
        iss.get(buffer, buffsize,',');
        //cout<<buffer<<endl;
        parameter_name.push_back(buffer);
        iss.ignore();
    }
}

void input_parameter_fitting::setValidParaLowerLimitFromDialog(string message){
    cout << message;
    char *buffer;
    buffer = new char[buffsize];
    cin.getline(buffer, buffsize);
    istringstream iss(buffer);
    float a;
    for (iss>>a; !iss.eof(); iss.ignore()>>a) {
        para_lowerLimit.push_back(a);
    }
    para_lowerLimit.push_back(a);
    
    while (para_lowerLimit.size()<fitting_para.size()) {
        para_lowerLimit.push_back(0);
    }
}

void input_parameter_fitting::setValidParaUpperLimitFromDialog(string message){
    cout << message;
    char *buffer;
    buffer = new char[buffsize];
    cin.getline(buffer, buffsize);
    istringstream iss(buffer);
    float a;
    for (iss>>a; !iss.eof(); iss.ignore()>>a) {
        para_upperLimit.push_back(a);
    }
    para_upperLimit.push_back(a);
    
    while (para_upperLimit.size()<fitting_para.size()) {
        para_upperLimit.push_back(0);
    }
}



//set
void input_parameter_fitting::setFittingOutputDir(string outputDir){
    fitting_output_dir=outputDir;
}

void input_parameter_fitting::setFittingFileBase(string outputfilebase){
    fitting_filebase=outputfilebase;
}

void input_parameter_fitting::setEJFileBase(string outputfilebase) {
	ej_filebase = outputfilebase;
}

void input_parameter_fitting::setFittingStartEnergyNo(int energyNo){
    fittingStartEnergyNo=energyNo;
}

void input_parameter_fitting::setFittingEndEnergyNo(int energyNo){
    fittingEndEnergyNo=energyNo;
}

void input_parameter_fitting::setEnergyFilePath(string energy_path){
    energyFilePath=energy_path;
}

void input_parameter_fitting::setE0(string E0_str){
    istringstream iss(E0_str);
    iss >> E0;
}

void input_parameter_fitting::setEnergyRange(string E_range){
    istringstream iss(E_range);
    iss >> startEnergy; iss.ignore() >> endEnergy;
}

void input_parameter_fitting::setFittingPara(string fitpara_input){
    istringstream iss(fitpara_input);
    float a;
    for (iss>>a; !iss.eof(); iss.ignore()>>a) {
        fitting_para.push_back(a);
    }
    fitting_para.push_back(a);
}

void input_parameter_fitting::setParaAttenuator(string para_input){
    istringstream iss(para_input);
    float a;
    for (iss>>a; !iss.eof(); iss.ignore()>>a) {
        para_attenuator.push_back(a);
    }
    para_attenuator.push_back(a);
}

void input_parameter_fitting::setFreeFixPara(string freepara_inp){
    istringstream iss(freepara_inp);
    char b;
    for (iss>>b; !iss.eof(); iss.ignore()>>b) {
        if (atoi(&b)==1) {
            free_para.push_back(true);
        }else{
            free_para.push_back(false);
        }
    }
    if (atoi(&b)==1) {
        free_para.push_back(true);
    }else{
        free_para.push_back(false);
    }
    
    while (free_para.size()<fitting_para.size()) {
        free_para.push_back(true);
    }
}

void input_parameter_fitting::setFittingParaName(string fitparaname_inp){
    char *buffer;
    buffer = new char[buffsize];
    istringstream iss(fitparaname_inp);
    for (;!iss.eof();) {
        iss.get(buffer, buffsize,',');
        //cout<<buffer<<endl;
        parameter_name.push_back(buffer);
        iss.ignore();
    }
}

void input_parameter_fitting::setValidParaLimit(string limit){
    char *buffer;
    buffer = new char[buffsize];
    istringstream iss(limit);
    float a,b;
    for (iss>>a; !iss.eof(); iss.ignore()>>a) {
        iss.ignore()>>b;
        para_lowerLimit.push_back(a);
        para_upperLimit.push_back(b);
    }
    para_lowerLimit.push_back(a);
    para_upperLimit.push_back(b);
    
    while (para_lowerLimit.size()<fitting_para.size()) {
        para_lowerLimit.push_back(0);
    }
    while (para_upperLimit.size()<fitting_para.size()) {
        para_upperLimit.push_back(0);
    }
}

int input_parameter_fitting::getNumTrial_fit() {
	return num_trial_fit;
}

float input_parameter_fitting::getLambda_t_fit() {
	return lamda_t_fit;
}

void input_parameter_fitting::setNumtrial_fit(int numTrial_inp) {
	num_trial_fit = numTrial_inp;
}

void input_parameter_fitting::setLambda_t_fit(float lambda_t_inp) {
	lamda_t_fit = lambda_t_inp;
}

void input_parameter_fitting::setCSbool(bool CSbool_inp) {
    CSbool = CSbool_inp;
}

void input_parameter_fitting::setCSepsilon(string para_input){
    istringstream iss(para_input);
    float a;
    for (iss>>a; !iss.eof(); iss.ignore()>>a) {
        CSepsilon.push_back(a);
    }
    CSepsilon.push_back(a);
}


void input_parameter_fitting::setCSalpha(string para_input){
    istringstream iss(para_input);
    float a;
    for (iss>>a; !iss.eof(); iss.ignore()>>a) {
        CSalpha.push_back(a);
    }
    CSalpha.push_back(a);
}

void input_parameter_fitting::setCSit(int CSit_inp) {
    CSit = CSit_inp;
}


//constructor
input_parameter_fitting::input_parameter_fitting(){

    fitting_output_dir="";
    fitting_filebase="";
    fittingStartEnergyNo=-1;
    fittingEndEnergyNo=-1;
    
    energyFilePath="";
    E0=NAN;
    startEnergy=NAN;
    endEnergy=NAN;

	num_trial_fit = 20;
	lamda_t_fit = 0.01f;
    
    CSbool=false;
    //CSepsilon = 1.0e-5f;
    //CSalpha = 0.01f;
    CSit = 5;
    
    numLCF = 0;
    numGauss = 0;
    numLor= 0;
    numAtan = 0;
    numErf = 0;
    numParameter = 0;
    
    numShell=-1;
    kstart=NAN;
    kend=NAN;
    Rstart=NAN;
    Rend=NAN;
    qstart=NAN;
    qend=NAN;
    kw=3;
    EXAFSfittingMode=1; //Rfit;
    ini_S02 = 1.0f;
    S02_freefix = false;
    edgeJ_dir_path="";
    useEdgeJ=false;
    Rbkg = 1.0f;
    preEdgeStartEnergyNo=-1;
    preEdgeEndEnergyNo=-1;
    postEdgeStartEnergyNo=-1;
    postEdgeEndEnergyNo=-1;
    preEdgeStartEnergy=NAN;
    preEdgeEndEnergy=NAN;
    postEdgeStartEnergy=NAN;
    postEdgeEndEnergy=NAN;
    bkgFittingMode=1;
    
}

float input_parameter_fitting::get_kstart(){
    return kstart;
}

float input_parameter_fitting::get_kend(){
    return kend;
}

void input_parameter_fitting::set_kend(float val){
    kend = val;
}

float input_parameter_fitting::get_Rstart(){
    return Rstart;
}

float input_parameter_fitting::get_Rend(){
    return Rend;
}

float input_parameter_fitting::get_qstart(){
    return qstart;
}

float input_parameter_fitting::get_qend(){
    return qend;
}

int input_parameter_fitting::get_kw(){
    return kw;
}
vector<string> input_parameter_fitting::getFeffxxxxdatPath(){
    return feffxxxxdat_path;
}

int input_parameter_fitting::getEXAFSfittingMode(){
    return EXAFSfittingMode;
}

int input_parameter_fitting::getShellNum() {
	return numShell;
}

vector<string> input_parameter_fitting::getShellName(){
    return shellname;
}

float input_parameter_fitting::getIniS02(){
    return ini_S02;
}

bool input_parameter_fitting::getS02freeFix(){
    return S02_freefix;
}
vector<vector<float>> input_parameter_fitting::getEXAFSiniPara(){
    return EXAFS_iniPara;
}

vector<vector<bool>> input_parameter_fitting::getEXAFSfreeFixPara(){
    return EXAFS_freeFixPara;
}

bool input_parameter_fitting::getUseEJ(){
    return useEdgeJ;
}

string input_parameter_fitting::getInputDir_EJ(){
    return edgeJ_dir_path;
}


input_parameter_fitting::input_parameter_fitting(string inputfile_path){
    
    input_parameter_fitting();
    
    ifstream inp_ifs(inputfile_path,ios::in);
    if(inp_ifs) {
        cout<<"input file found."<<endl<<endl;
        while (!inp_ifs.eof()) {
            char *buffer;
            buffer = new char[buffsize];
            inp_ifs.getline(buffer, buffsize);
            inputFromFile_fitting(buffer, &inp_ifs);
            
        }
        inp_ifs.close();
    }
}


void input_parameter_fitting::inputFromFile_fitting(char *buffer, ifstream *inp_ifs){
    if((string)buffer=="#Output directory path for XANES fitting"){
        cout<<buffer<<endl;
        (*inp_ifs).getline(buffer, buffsize);
        fitting_output_dir=buffer;
        cout<<"  "<<fitting_output_dir<<endl;
    }else if((string)buffer=="#Output file base name for XANES fitting"){
        cout<<buffer<<endl;
        (*inp_ifs).getline(buffer, buffsize);
        fitting_filebase=buffer;
        cout<<"  "<<fitting_filebase<<endl;
    }else if((string)buffer=="#Energy data file path"){
        cout<<buffer<<endl;
        (*inp_ifs).getline(buffer, buffsize);
        energyFilePath=buffer;
        cout<<"  "<<energyFilePath<<endl;
    }else if((string)buffer=="#E0"){
        cout<<buffer<<endl;
        (*inp_ifs)>>E0;
        cout<<"  "<<E0<<endl;
    }else if((string)buffer=="#Energy range"){
        cout<<buffer<<endl;
        (*inp_ifs)>>startEnergy; (*inp_ifs).ignore() >> endEnergy;
        cout<<"  "<<startEnergy<<"-"<<endEnergy<<endl;
    }else if((string)buffer=="#Fitting parameter name"){
        cout<<buffer<<endl;
        (*inp_ifs).getline(buffer, buffsize);
        istringstream iss(buffer);
        cout<<"  ";
        for (int i=0;(!iss.eof())&&(i<numParameter);i++) {
            iss.get(buffer, buffsize,',');
            parameter_name.push_back(buffer);
            cout<<buffer;
            iss.ignore();
            if(iss.eof()) cout<<endl;
            else cout<<",";
        }
    }else if((string)buffer=="#Processing parameter name"){
        cout<<buffer<<endl;
        (*inp_ifs).getline(buffer, buffsize);
		//cout << buffer << endl;
        istringstream iss(buffer);
        cout<<"  ";
        for (int i=0;!iss.eof();i++) {
            iss.get(buffer, buffsize,',');
            parameter_name.push_back(buffer);
            cout<<buffer;
            iss.ignore();
            if(iss.eof()) cout<<endl;
            else cout<<",";
        }
    }else if((string)buffer=="#Initial fitting parameter"){
        cout<<buffer<<endl;
        (*inp_ifs).getline(buffer, buffsize);
        istringstream iss(buffer);
        float a;
        int i;
        cout<<"  ";
        for (iss>>a,i=0; (!iss.eof())&&(i+1<numParameter); iss.ignore()>>a,i++) {
            fitting_para.push_back(a);
            cout<<a<<",";
        }
        fitting_para.push_back(a);
        cout<<a<<endl;
    }else if((string)buffer=="#Free/fix parameter for XANES fitting"){
        cout<<buffer<<endl;
        (*inp_ifs).getline(buffer, buffsize);
        istringstream iss(buffer);
        char b;
        int i;
        cout<<"  ";
        for (iss>>b,i=0; (!iss.eof())&&(i<numParameter); iss.ignore()>>b,i++) {
            if (atoi(&b)==1) {
                free_para.push_back(true);
            }else{
                free_para.push_back(false);
            }
			if (iss.eof() || (i + 1==numParameter)) cout << b;
			else cout<<b<<",";
        }
        
        while (free_para.size()<fitting_para.size()) {
            free_para.push_back(true);
            cout<<","<<1;
        }
        cout<<endl;
    }
	else if ((string)buffer == "#Free/fix parameter") {
		cout << buffer << endl;
		(*inp_ifs).getline(buffer, buffsize);
		istringstream iss(buffer);
		char b;
		int i;
		cout << "  ";
		for (iss >> b, i = 0; (!iss.eof()) && (i<numParameter); iss.ignore() >> b, i++) {
            if (atoi(&b)==1) {
                free_para.push_back(true);
            }else{
                free_para.push_back(false);
            }
			if (iss.eof() || (i + 1 == numParameter)) cout << b;
			else cout << b << ",";
		}

		while (free_para.size()<fitting_para.size()) {
			free_para.push_back(true);
			cout << "," << 1;
		}
		cout << endl;
	}
	else if((string)buffer=="#Valid parameter lower limit"){
        cout<<buffer<<endl;
        (*inp_ifs).getline(buffer, buffsize);
        istringstream iss(buffer);
        float a;
        int i;
        cout<<"  ";
        for (iss>>a,i=0; (!iss.eof())&&(i<numParameter); iss.ignore()>>a,i++) {
            para_lowerLimit.push_back(a);
            cout<<a<<",";
        }
        para_lowerLimit.push_back(a);
        cout<<a;
        
        while (para_lowerLimit.size()<fitting_para.size()) {
            para_lowerLimit.push_back(-INFINITY);
            cout<<",-INF";
        }
        cout<<endl;
    }else if((string)buffer=="#Valid parameter upper limit"){
        cout<<buffer<<endl;
        (*inp_ifs).getline(buffer, buffsize);
        istringstream iss(buffer);
        float a;
        int i;
        cout<<"  ";
        for (iss>>a,i=0; (!iss.eof())&&(i<numParameter); iss.ignore()>>a,i++) {
            para_upperLimit.push_back(a);
            cout<<"  "<<a<<",";
        }
        para_upperLimit.push_back(a);
        cout<<a;
        
        while (para_upperLimit.size()<fitting_para.size()) {
            para_upperLimit.push_back(INFINITY);
            cout<<",INF";
        }
        cout<<endl;
    }else if((string)buffer=="#Parameter attenuator for reconstruction"){
        cout<<buffer<<endl;
        (*inp_ifs).getline(buffer, buffsize);
        istringstream iss(buffer);
        float a;
        int i;
        cout<<"  ";
        for (iss>>a,i=0; (!iss.eof())&&(i<numParameter); iss.ignore()>>a,i++) {
            para_attenuator.push_back(a);
            cout<<a<<",";
        }
        para_attenuator.push_back(a);
        cout<<a<<endl;
    }else if((string)buffer=="#CS-based iteration for XANES fitting"){
        cout<<buffer<<endl;
        int dummy;
        (*inp_ifs)>>dummy;
        CSbool = (dummy==1) ? true:false;
        cout<<"  "<< boolalpha <<CSbool<<endl;
    }else if((string)buffer=="#Noise factor of CS-based iteration for XANES fitting"){
        cout<<buffer<<endl;
        (*inp_ifs).getline(buffer, buffsize);
        istringstream iss(buffer);
        float a;
        int i;
        cout<<"  ";
        for (iss>>a,i=0; (!iss.eof())&&(i<numParameter); iss.ignore()>>a,i++) {
            CSepsilon.push_back(a);
            cout<<a<<",";
        }
        CSepsilon.push_back(a);
        cout<<a<<endl;
    }else if((string)buffer=="#Update factor CS-based iteration for XANES fitting"){
        cout<<buffer<<endl;
        (*inp_ifs).getline(buffer, buffsize);
        istringstream iss(buffer);
        float a;
        int i;
        cout<<"  ";
        for (iss>>a,i=0; (!iss.eof())&&(i<numParameter); iss.ignore()>>a,i++) {
            CSalpha.push_back(a);
            cout<<a<<",";
        }
        CSalpha.push_back(a);
        cout<<a<<endl;
    }else if((string)buffer=="#Iteration number of CS-based iteration for XANES fitting"){
        cout<<buffer<<endl;
        (*inp_ifs)>>CSit;
        cout<<"  "<<CSit<<endl;
    }else if((string)buffer=="#XANES fitting equation"){
        cout<<buffer<<endl;
        (*inp_ifs).getline(buffer, buffsize);
        istringstream iss(buffer);
        cout<<"  ";
        string funcname;
        for (;!iss.eof();) {
            iss.get(buffer, buffsize,'+');
            funcname = buffer;
			size_t pos;
			while ((pos = funcname.find_first_of(" 　\t")) != string::npos) {
				funcname.erase(pos, 1);
			}
            if(funcname == "line"){
                funcNameList.push_back("line");
                numParameter += 2;
                cout<<"line";
            }else if(funcname == "Gaussian"){
                funcNameList.push_back("Gaussian");
				cout << "Gaussian("<<numGauss+1<<")";
                numGauss++;
                numParameter += 3;
            }else if(funcname == "Lorentzian"){
                funcNameList.push_back("Lorentzian");
				cout << "Lorentzian("<<numLor+1<<")";
                numLor++;
                numParameter += 3;
            }else if(funcname == "atanE"){
                funcNameList.push_back("atanE");
				cout << "atanE("<<numAtan+1<<")";
                numAtan++;
                numParameter += 3;
            }else if(funcname == "erfE"){
                funcNameList.push_back("erfE");
				cout << "erfE("<<numErf+1<<")";
                numErf++;
                numParameter += 3;
            }else if(funcname == "stdXANES"){
                funcNameList.push_back("stdXANES");
				cout << "stdXANES("<<numLCF+1<<")";
                numLCF++;
                numParameter += 2;
            }

            iss.ignore();
            if(iss.eof()) cout<<endl;
            else cout<<"+";
        }
    }else if((string)buffer=="#Standard XANES spectra data file paths for LCF"){
        cout<<buffer<<endl;
        for (int i=0; i<numLCF; i++) {
			(*inp_ifs).getline(buffer, buffsize);
            LCFstd_paths.push_back(buffer);
            cout<<"("<<i+1<<")  "<<LCFstd_paths[i]<<endl;
        }
    }else if((string)buffer=="#Output directory path for EXAFS fitting"){
        cout<<buffer<<endl;
        (*inp_ifs).getline(buffer, buffsize);
        fitting_output_dir=buffer;
        cout<<"  "<<fitting_output_dir<<endl;
    }else if((string)buffer=="#Use edge jump images"){
        cout<<buffer<<endl;
        int dummy;
        (*inp_ifs)>>dummy;
        useEdgeJ = (dummy==1) ? true:false;
        cout<<"  "<< boolalpha <<useEdgeJ<<endl;
    }else if((string)buffer=="#Input directory path of edge jump image"){
        cout<<buffer<<endl;
        (*inp_ifs).getline(buffer, buffsize);
        edgeJ_dir_path=buffer;
        cout<<"  "<<edgeJ_dir_path<<endl;
    }else if((string)buffer=="#EXAFS fitting mode"){
        cout<<buffer<<endl;
        (*inp_ifs)>>EXAFSfittingMode;
        cout<<"  "<<EXAFSfittingMode<<endl;
    }else if((string)buffer=="#k-weight"){
        cout<<buffer<<endl;
        (*inp_ifs)>>kw;
        cout<<"  "<<kw<<endl;
    }else if((string)buffer=="#k-range"){
        cout<<buffer<<endl;
        (*inp_ifs)>>kstart; (*inp_ifs).ignore() >> kend;
        cout<<"  "<<kstart<<"-"<<kend<<endl;
    }else if((string)buffer=="#R-range"){
        cout<<buffer<<endl;
        (*inp_ifs)>>Rstart; (*inp_ifs).ignore() >> Rend;
        cout<<"  "<<Rstart<<"-"<<Rend<<endl;
    }else if((string)buffer=="#q-range"){
        cout<<buffer<<endl;
        (*inp_ifs)>>qstart; (*inp_ifs).ignore() >> qend;
        cout<<"  "<<qstart<<"-"<<qend<<endl;
    }else if((string)buffer=="#Number of EXAFS fitting shells"){
        cout<<buffer<<endl;
        (*inp_ifs)>>numShell;
        cout<<"  "<<numShell<<endl;
    }else if((string)buffer=="#Shell names"){
        cout<<buffer<<endl;
        for (int i=0; i<numShell; i++) {
            (*inp_ifs).getline(buffer, buffsize);
            shellname.push_back(buffer);
            cout<<"("<<i+1<<")  "<<shellname[i]<<endl;
        }
    }else if((string)buffer=="#feffxxxx.dat file paths"){
        cout<<buffer<<endl;
        for (int i=0; i<numShell; i++) {
            (*inp_ifs).getline(buffer, buffsize);
            feffxxxxdat_path.push_back(buffer);
            cout<<"("<<i+1<<")  "<<feffxxxxdat_path[i]<<endl;
        }
    }else if((string)buffer=="#EXAFS initial fitting parameter"){
        cout<<buffer<<endl;
        for (int i=0; i<numShell; i++) {
            vector<float> EXAFS_iniPara_atSh;
            (*inp_ifs).getline(buffer, buffsize);
            istringstream iss(buffer);
            float a;
            int j;
            cout<<"  ";
            for (iss>>a,j=0; (!iss.eof())||(j+1<4); iss.ignore()>>a,j++) {
                EXAFS_iniPara_atSh.push_back(a);
                cout<<a<<",";
            }
            EXAFS_iniPara_atSh.push_back(a);
            cout<<a<<endl;
            EXAFS_iniPara.push_back(EXAFS_iniPara_atSh);
        }
    }else if((string)buffer=="#EXAFS free/fix parameter"){
        cout<<buffer<<endl;
        for (int i=0; i<numShell; i++) {
            vector<bool> EXAFS_freeFixPara_atSh;
            (*inp_ifs).getline(buffer, buffsize);
            istringstream iss(buffer);
            char b;
            int j;
            cout<<"  ";
            for (iss>>b,j=0; (!iss.eof())||(j<4); iss.ignore()>>b,j++) {
                if (b=='1') {
                    EXAFS_freeFixPara_atSh.push_back(true);
                }else{
                    EXAFS_freeFixPara_atSh.push_back(false);
                }
                if (iss.eof() || (j + 1==4)) cout << boolalpha << EXAFS_freeFixPara_atSh[j];
                else cout<< boolalpha << EXAFS_freeFixPara_atSh[j]<<",";
            }
            
            while (EXAFS_freeFixPara_atSh.size()<EXAFS_iniPara[i].size()) {
                EXAFS_freeFixPara_atSh.push_back(true);
                cout<<","<< boolalpha << true;
            }
            cout<<endl;
            EXAFS_freeFixPara.push_back(EXAFS_freeFixPara_atSh);
        }
    }else if((string)buffer=="#initial S02 value"){
        cout<<buffer<<endl;
        (*inp_ifs)>>ini_S02;
        cout<<"  "<<ini_S02<<endl;
    }else if((string)buffer=="#Free/fix S02"){
        cout<<buffer<<endl;
        int dummy;
        (*inp_ifs)>>dummy;
        S02_freefix = (dummy==1) ? true:false;
        cout<<"  "<< boolalpha <<S02_freefix<<endl;
    }else if((string)buffer=="#Rbkg"){
        cout<<buffer<<endl;
        (*inp_ifs)>>Rbkg;
        cout<<"  "<<Rbkg<<endl;
    }else if((string)buffer=="#Pre-edge fitting energy range"){
        cout<<buffer<<endl;
        (*inp_ifs)>>preEdgeStartEnergy; (*inp_ifs).ignore() >> preEdgeEndEnergy;
        cout<<"  "<<preEdgeStartEnergy<<"-"<<preEdgeEndEnergy<<endl;
    }else if((string)buffer=="#Post-edge fitting energy range"){
        cout<<buffer<<endl;
        (*inp_ifs)>>postEdgeStartEnergy; (*inp_ifs).ignore() >> postEdgeEndEnergy;
        cout<<"  "<<postEdgeStartEnergy<<"-"<<postEdgeEndEnergy<<endl;
    }else if((string)buffer=="#Pre-edge fitting equation"){
        cout<<buffer<<endl;
        (*inp_ifs)>>bkgFittingMode;
        cout<<"  "<<bkgFittingMode<<endl;
    }
}
