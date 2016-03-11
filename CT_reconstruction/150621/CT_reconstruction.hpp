
//
//  CT_reconstruction.hpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/06/13.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#ifndef CT_XANES_analysis_CT_reconstruction_hpp
#define CT_XANES_analysis_CT_reconstruction_hpp

#include <time.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;
#define _USE_MATH_DEFINES


#if defined (__APPLE__)  //Mac OS X, iOS?
#define MKDIR(c) \
mkdir((const char*)(c), 0755)


#elif defined (_M_X64)  //Windows 64 bit
#include "stdafx.h"
#include <direct.h>
#include <windows.h>
#include <mmsystem.h>
#pragma comment (lib, "winmm.lib")

#define MKDIR(c) \
_mkdir((const char*)(c))


#elif defined (_WIN32)  //Windows 32 bit
#include "stdafx.h"
#include <direct.h>
#include <windows.h>
#include <mmsystem.h>
#pragma comment (lib, "winmm.lib")

#define MKDIR(c) \
_mkdir((const char*)(c))

#endif

#include "OpenCL_analysis.hpp"

void getparameter();
void getparameter_inp(string inputfile_path);

void read_log(string fi, int);
void read_data_input(string di, string fi, float *, int);
void write_data_output(string di, string fi, float *, int);
void write_data_temp(string di, string fi, float *, int);
void first_image(string fi, float *, int);
void detection_probability(int, int, int, float *);
void projection(float *, float *, int, int, int, int);
void backprojection(float *, float *, int, int, int, float *, int);
void backprojection_em(float *, float *, int, int, int, int);
void make_pj(float *, float *, int, int, int, float *, float *, int, int);
void AART(float *, float *, int, int, int, float *, int, int, double);
void MART(float *, float *, int, int, int, float *, int, int);
void ASIRT(float *, float *, int, int, int, float *, int, int, double);
void MSIRT(float *, float *, int, int, int, float *, int, int);
void MLEM(float *, float *, int, int, int, float *, int, int);
void OSEM(float *, float *, int, int, int, float *, int, int, int);

void FBP(float *, float *, int, int, int, float *, int);
void zero_padding(float *, float *, int, int, int);
void FFT_filter(float *, int, int);
void FFT(int, int, float *, float *, float *, float *, unsigned short *);
void bitrev(int, float *, float *, unsigned short *);
void FFT_init(int, float *, float *, unsigned short *);
int br(int, unsigned);
void Filter(float *, int);

int OSEM_ocl(OCL_platform_device plat_dev_list, float *ang, int ss);
int OSEM_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                cl::Buffer angle_buffer, /*cl::Buffer*/int *sub/*_buffers[j]*/,
                int startN, int endN, int it);

extern string	g_d1;	/* 入力ファイルディレクトリパス */
extern string	g_f1;	/* 入力ファイル名 */
extern string	g_d2;	/* 出力ファイルディレクトリパス */
extern string	g_f2;	/* 出力ファイル名 */
extern string	g_f3;	/* logファイルパス */
extern string	g_f4;	/* 初期画像ファイル名*/
extern string	g_d5;	/* 一時ファイルディレクトリパス */
extern int		g_fst;				/* 初期画像の使用 */
extern double	g_ini;				/* 初期値 */
extern int		g_px;				/* カメラピクセルサイズ */
extern int		g_pa;				/* 投影数 */
extern int		g_nx;				/* 再構成ピクセルサイズ */
extern int		g_mode;				/* 再構成法 */
extern int		g_it;				/* 反復回数 */
extern int		g_num;			/* レイヤー数 */
extern int		g_st;			/* 開始ナンバー */
extern int		g_x;			/* 回転中心のずれ */
extern double	g_wt1;			/* AARTファクター */
extern double	g_wt2;			/* ASIRTファクター */
extern int		g_ss;				/* サブセット */
extern float	*g_prj;					/* 投影データ */
extern float	*g_ang;					/* 投影角度 */
extern float	*g_img;					/* 再構成データ */
extern int		*g_cx;					/* 検出位置 */
extern float	*g_c0;					/* 検出確率(-1) */
extern float	*g_c1;					/* 検出確率(0) */
extern float	*g_c2;					/* 検出確率(+1) */
extern int		g_cover;			/* 非投影領域の推定 */
extern float	*g_prjf;				/* 仮想投影データ */
extern time_t	g_t0;					/* 時間 */
extern string   g_devList;            //OpenCLデバイスリスト
#endif
