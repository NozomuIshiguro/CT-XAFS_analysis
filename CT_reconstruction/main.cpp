//
//  main.cpp
//  CT_reconstruction
//
//  Created by Nozomu Ishiguro on 2015/06/19.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//


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
int		g_nx = 2048;				/* 再構成ピクセルサイズ */
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

//int CPU();
int GPU();

vector<thread> input_th, reconst_th, output_th;

int main(int argc, const char * argv[]) {
    string inputfile_path;
    cout << "Inputファイルがあればパスを入力:"<<endl;
    string dummy;
    getline(cin,dummy);
    istringstream iss(dummy);
    iss>>inputfile_path;
    getparameter_inp(inputfile_path);
    
    time(&g_t0);
    g_ang = new float[(unsigned long)g_pa];
    read_log(g_f3, g_pa);
    MKDIR(g_d2.c_str());  //reconstディレクトリ
    
    //CPU();
    GPU();
}
