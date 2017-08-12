//
//  main.cpp
//  XAFSreconst_fitting
//
//  Created by Nozomu Ishiguro on 2017/02/16.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#include "XANES_fitting.hpp"
#include "CT_reconstruction.hpp"

mutex m1,m2;
vector<thread> input_th,fitting_th, output_th_fit;

string	g_d1 = "input";         /* 入力ファイルディレクトリ */
string	g_f1 = "i****.raw";     /* 入力ファイル */
string	g_d2 = "output";        /* 出力ファイルディレクトリ */
string	g_f2 = "r****.img";     /* 出力ファイル */
string	g_f3 = "log.dat";       /* logファイル */
string	g_d4 = ".";             /* 初期画像ファイルディレクトリ */
string	g_f4 = "1st.img";       /* 初期画像ファイル */
string	g_d5 = "temp";          /* 一時ファイルディレクトリ */
int		g_fst = 2;				/* 初期画像の使用 */
double	g_ini = 1;				/* 初期値 */
int		g_px = 2048;			/* カメラピクセルサイズ */
int		g_pa = 1600;			/* 投影数 */
int		g_nx = 2048;			/* 再構成ピクセルサイズ */
int		g_ny = 2048;			/* 再構成ピクセルサイズ */
int		g_ox = 2048;			/* 再構成ピクセルサイズ */
int		g_oy = 2048;			/* 再構成ピクセルサイズ */
int		g_mode = 6;				/* 再構成法 */
int		g_it = 1;				/* 反復回数 */
int		g_num = 1;              /* レイヤー数 */
int		g_st = 1;               /* 開始ナンバー */
int		g_x = 0;                /* 回転中心のずれ */
double	g_wt1 = 1.0;			/* AARTファクター */
double	g_wt2 = 0.01;			/* ASIRTファクター */
int		g_ss = 20;				/* サブセット */
int		g_zp = 4096;			/* ゼロパディングサイズ */
float	*g_ang;					/* 投影角度 */
int		g_cover = 2;			/* 非投影領域の推定 */
time_t	g_t0;					/* 時間 */
string   g_devList="1"; //OpenCLデバイスリスト
int numParallel=10; //OpenCLデバイス当たりの並列数
int correctionMode = 1; //投影像補正 0:なし,1:x方向,2:θ方向,3:x+θ方向
float amp = 1.0f; //強度増幅因子

int XAFSreconstRit_ocl(fitting_eq fiteq, input_parameter inp,
                       OCL_platform_device plat_dev_list);

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
	getparameter_inp(fp_str);
    OCL_platform_device plat_dev_list(inp.getPlatDevList(),true);
    cout << endl;
    
    
    // Input directory settings
    string buffer;
    buffer = output_flag("-ip", argc, argv);
    if (buffer.length()>0) {
        inp.setInputDir(buffer);
    }
    if (inp.getInputDir().length()==0) {
        inp.setInputDirFromDialog("Set input mt raw file directory.");
        cout<<endl;
    }
    string fileName_base = inp.getInputDir();
    fileName_base += "/001";
    
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
            inp.setFittingOutputDirFromDialog("Set output file directory.");
            cout<<endl;
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
        inp.setEnergyFilePathFromDialog("Set energy file path.");
        cout<<endl;
    }
    ifstream energy_ifs(inp.getEnergyFilePath(),ios::in);
    if(!energy_ifs) {
        cerr << "   Failed to load energy file" << endl;
        return -1;
    }
    
    
    //processing E0
    buffer = output_flag("-e0", argc, argv);
    if (buffer.length()>0) {
        inp.setE0(buffer);
    }
    if (inp.getE0()==NAN) {
        inp.setE0FromDialog("Set E0 /eV (ex. 11559).");
        cout<<endl;
    }
    
    //processing energy range
    buffer = output_flag("-er", argc, argv);
    if (buffer.length()>0) {
        inp.setEnergyRange(buffer);
    }
    if ((inp.getStartEnergy()==NAN)|(inp.getEndEnergy()==NAN)) {
        inp.setEnergyRangeFromDialog("Set fitting energy range /eV (ex. 11540.0-11600.0).");
        cout<<endl;
    }
    
    
    //processing angle range
    buffer = output_flag("-ar", argc, argv);
    if (buffer.length()>0) {
        inp.setAngleRange(buffer);
    }
    if ((inp.getStartAngleNo()==NAN)|(inp.getEndAngleNo()==NAN)) {
        inp.setAngleRangeFromDialog("Set angle num range (ex. 1-1600).");
        cout<<endl;
    }
    
    
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
    
    fitting_eq fiteq(inp,"","");
    vector<string> function_name{"linear","atan edge","Lorentzian"};
    vector<int> function_num{1,1,1};
    fiteq.set_fitting_preprocessor(function_name, function_num);
    
    time_t start,end;
    time(&start);
    
    //inp.setLambda_t_fit(1.0f);
    
    //XANES fitting
    //XANES_fit_ocl(fiteq, inp, plat_dev_list);
    XAFSreconstRit_ocl(fiteq, inp, plat_dev_list);
    
    
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
