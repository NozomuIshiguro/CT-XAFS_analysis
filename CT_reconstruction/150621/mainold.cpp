//
//  main.cpp
//  CT-XANES_analysis
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
float	*g_prj;					/* 投影データ */
float	*g_ang;					/* 投影角度 */
float	*g_img;					/* 再構成データ */
int		*g_cx;					/* 検出位置 */
float	*g_c0;					/* 検出確率(-1) */
float	*g_c1;					/* 検出確率(0) */
float	*g_c2;					/* 検出確率(+1) */
int		g_cover = 2;			/* 非投影領域の推定 */
float	*g_prjf;				/* 仮想投影データ */
time_t	g_t0;					/* 時間 */
string   g_devList="1";

int CPU();
int GPU();

int main(int argc, const char * argv[]) {
    char inputfile_path[256];
    cout << "Inputファイルがあればパスを入力:"<<endl;
    cin.getline(inputfile_path, 256);
    getparameter_inp((string)inputfile_path);
    
    time(&g_t0);
    g_ang = new float[(unsigned long)g_pa];
    read_log(g_f3, g_pa);
    MKDIR(g_d2.c_str());  //reconstディレクトリ
    
    //CPU();
    GPU();
}
