//
//  main.cpp
//  Reslice_CTreconst
//
//  Created by Nozomu Ishiguro on 2016/03/10.
//  Copyright © 2016年 Nozomu Ishiguro. All rights reserved.
//

#include <iostream>

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
int correctionMode = 0; //投影像補正 0:なし,1:x方向,2:θ方向,3:x+θ方向
float amp = 1.0f; //強度増幅因子
int baseupOrder =4;//baseup 減少速度次数
bool CSitBool = false; //圧縮センシング逐次計算
float CSepsilon = 1.0e-8f; //圧縮センシング逐次計算ノイズファクター
float CSalpha = 0.1f; //圧縮センシング逐次計算加算ファクター
int CSit = 5; //圧縮センシング逐次計算回数

int GPU();

vector<thread> input_th, reconst_th, output_th, reslice_th, output_th1, output_th2;

int reslice_CTreconst_ocl(input_parameter inp,string fileName_base,float *ang);

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
    
    OCL_platform_device plat_dev_list(inp.getPlatDevList(),true);
    cout << endl;
    
    /* Input directory settings*/
    string buffer;
    buffer = output_flag("-ip", argc, argv);
    if (buffer.length()>0) {
        inp.setInputDir(buffer);
    }
    if (inp.getInputDir().length()==0) {
        inp.setInputDirFromDialog("Set input mt raw file directory.\n");
    }
    
    //processing energy No range
    buffer = output_flag("-enr", argc, argv);
    if (buffer.length()>0) {
        inp.setEnergyNoRange(buffer);
    }
    //parameter name
    buffer = output_flag("-fpn", argc, argv);
    if (buffer.length()>0) {
        inp.setFittingParaName(buffer);
    }
    //Input processing energy range from dialog
    if ((inp.getStartEnergyNo()<0)|(inp.getEndEnergyNo()<0)) {
        inp.setEnergyNoRangeFromDialog("Set energy num range (ex. 1-100).\n");
    }
    //Input processing parameter name from dialog
    if ((inp.getStartEnergyNo()<0)|(inp.getEndEnergyNo()<0)) {
        if (inp.getFittingParaName().size()==0) {
            inp.setFittingParaNameFromDialog("Set fitting parameter names (ex. a0,a1,.....)\n");
        }
    }
    
    //check if input data exist
    string fileName_base = inp.getInputDir();
    if (!(inp.getStartEnergyNo()<0)) {
        fileName_base += EnumTagString(inp.getStartEnergyNo(),"/","");
    }else if(inp.getFittingParaName().size()!=0){
        fileName_base += "/"+inp.getFittingParaName()[0];
    }
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
    
    
    
    /*output directory settings*/
    buffer = output_flag("-op", argc, argv);
    if (buffer.length()>0) {
        inp.setOutputDir(buffer);
    }
    if (inp.getOutputDir().length()==0) {
        inp.setOutputDirFromDialog("Set output file directory.\n");
	}
    MKDIR(inp.getOutputDir().c_str());
    cout <<endl;
    

    //processing angle No range
    buffer = output_flag("-ar", argc, argv);
    if (buffer.length()>0) {
        inp.setAngleRange(buffer);
    }
    if ((inp.getStartAngleNo()==NAN)|(inp.getEndAngleNo()==NAN)) {
        inp.setAngleRangeFromDialog("Set angle num range (ex. 1-1600).\n");
    }
    
    //image baseup
    if (inp.getBaseup()==NAN) {
        inp.setBaseupFromDialog("Set baseup value of image.\n");
    }

    
    getparameter_inp(fp_str);
    
    time(&g_t0);
    g_ang = new float[(unsigned long)g_pa];
    read_log(g_f3, g_pa);
    
    reslice_CTreconst_ocl(inp,fileName_base,g_ang);
    
    return 0;
}

