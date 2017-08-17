//
//  main.cpp
//  RotationCenterSearch
//
//  Created by Nozomu Ishiguro on 2016/02/16.
//  Copyright © 2016年 Nozomu Ishiguro. All rights reserved.
//


#include "CTXAFS.hpp"
#include "CT_reconstruction.hpp"

string	g_d1 = "input";	/* 入力ファイルディレクトリ */
string	g_f1 = "i****.raw";	/* 入力ファイル */
string	g_d2 = "output";	/* 出力ファイルディレクトリ */
string	g_f2 = "r****.img";	/* 出力ファイル */
string	g_f3 = "log.dat";	/* logファイル */
string	g_d4 = ".";	/* 初期画像ファイルディレクトリ */
string	g_f4 = "1st.img";	/* 初期画像ファイル */
string	g_d5 = "temp";	/* 一時ファイルディレクトリ */
int		g_fst = 2;				/* 初期画像の使用 */
double	g_ini = 1;				/* 初期値 */
int		g_px = 2048;				/* カメラピクセルサイズ */
int		g_pa = 1600;				/* 投影数 */
int		g_nx = 2048;				/* 再構成ピクセルサイズ X*/
int		g_ny = 2048;				/* 再構成ピクセルサイズ Y*/
int		g_ox = 2048;				/* 再構成出力ピクセルサイズ X*/
int		g_oy = 2048;				/* 再構成出力ピクセルサイズ Y*/
int		g_mode = 6;				/* 再構成法 */
int		g_it = 1;				/* 反復回数 */
int		g_num = 1;			/* レイヤー数 */
int		g_st = 1;			/* 開始ナンバー */
int		g_x = 0;			/* 回転中心のずれ */
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
int baseupOrder =4;//baseup 減少速度次数
bool CSitBool = false; //圧縮センシング逐次計算
float CSepsilon = 1.0e-8f; //圧縮センシング逐次計算ノイズファクター
float CSalpha = 0.1f; //圧縮センシング逐次計算加算ファクター
int CSit = 5; //圧縮センシング逐次計算回数

mutex m1,m2;
int rotationCenterSearch(string fileName_base, input_parameter inp, float *ang);

int main(int argc, const char * argv[]) {
    
    cout << "-----------------------------------------------"<<endl<<endl;
    cout << "         Rotation center search for CT" <<endl<<endl;
    cout << "         First version: Feb. 16th, 2016"<<endl;
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
        //fp_str="./image_reg.inp";
    }
    
    
    input_parameter inp(fp_str);
    OCL_platform_device plat_dev_list(inp.getPlatDevList(),true); //true:非共有context
    //inp.setLambda_t(0.00001f);
    inp.setNumtrial(5);
    cout <<endl;
    
    if (inp.getNumParallel()==NAN) {
        inp.setNumParallel(10);
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
            cout << "his file found: " << darkname <<endl;
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
    
    
    //output directory settings (optional)
    buffer = output_flag("-op", argc, argv);
    if (buffer.length()>0) {
        inp.setOutputDir(buffer);
        cout << "Output file directory."<<endl;
        cout << inp.getOutputDir()<<endl;
    }
    if (inp.getOutputDir().length()!=0) {
        MKDIR(inp.getOutputDir().c_str());
    }
    
    
    
    //output file base settings (optional)
    buffer = output_flag("-ob", argc, argv);
    if (buffer.length()>0) {
        inp.setOutputFileBase(buffer);
        cout << "Output file name base."<<endl;
        cout << inp.getOutputFileBase()<<endl;
    }
    if (inp.getOutputFileBase().length()==0) {
        inp.setOutputFileBaseFromDialog("Set output file name base.\n");
    }
    
    
    //processing target energy No
    buffer = output_flag("-re", argc, argv);
    if (buffer.length()>0){
        inp.setTargetEnergyNo(buffer);
        cout << "target energy No. for rotation center search: "<<endl;
        cout << "   " << inp.getTargetEnergyNo() << endl;
    }
    if (inp.getTargetEnergyNo()<0) {
        inp.setTargetEnergyNoFromDialog("target energy No. for rotation center search.\n");
    }
    
    //searching layer No
    buffer = output_flag("-ly", argc, argv);
    if (buffer.length()>0){
        inp.setLayerN(buffer);
        cout << "target layer No. for rotation center search: "<<endl;
        cout << "   " << inp.getTargetEnergyNo() << endl;
    }
    if (inp.getLayerN()==NAN) {
        inp.setLayerNFromDialog("target layer No. for rotation center search.\n");
    }
    
    
    //searching rotation shift start
    buffer = output_flag("-rss", argc, argv);
    if (buffer.length()>0){
        inp.setRotCenterShiftStart(buffer);
        cout << "start shift for rotation center search: "<<endl;
        cout << "   " << inp.getRotCenterShiftStart() << endl;
    }
    if (inp.getRotCenterShiftStart()==NAN) {
        inp.setRotCenterShiftStartFromDialog("start shift for rotation center search.\n");
    }
    
    //searching rotation shift No
    buffer = output_flag("-rsn", argc, argv);
    if (buffer.length()>0){
        inp.setRotCenterShiftStart(buffer);
        cout << "number of shift step for rotation center search: "<<endl;
        cout << "   " << inp.getRotCenterShiftN() << endl;
    }
    if (inp.getRotCenterShiftN()==NAN) {
        inp.setRotCenterShiftNFromDialog("start shift for rotation center search.\n");
    }
    
    //searching rotation shift step
    buffer = output_flag("-rsst", argc, argv);
    if (buffer.length()>0){
        inp.setRotCenterShiftStep(buffer);
        cout << "shift step for rotation center search: "<<endl;
        cout << "   " << inp.getRotCenterShiftStep() << endl;
    }
    if (inp.getRotCenterShiftStep()==NAN) {
        inp.setRotCenterShiftStepFromDialog("start shift for rotation center search.\n");
    }
    
	//CT_reconst parameter
	getparameter_inp(fp_str);
    g_pa=inp.getEndAngleNo()-inp.getStartAngleNo()+1;
    
    time_t start,end;
    time(&start);

	g_ang = new float[(unsigned long)g_pa];
	read_log(g_f3, g_pa);
    
	rotationCenterSearch(fileName_base, inp, g_ang);
    
    
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
    
    cout << endl << "Press 'Enter' to quit." << endl;
    string dummy;
    getline(cin,dummy);
    
    return 0;
}
