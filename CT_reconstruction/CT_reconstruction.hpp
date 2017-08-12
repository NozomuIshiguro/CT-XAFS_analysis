
//
//  CT_reconstruction.hpp
//  CT_reconstruction
//
//  Created by Nozomu Ishiguro on 2015/06/13.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#ifndef CT_XANES_analysis_CT_reconstruction_hpp
#define CT_XANES_analysis_CT_reconstruction_hpp

#include "OpenCL_analysis.hpp"

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
//#include "stdafx.h"
#include <direct.h>
#include <windows.h>
#include <mmsystem.h>
#pragma comment (lib, "winmm.lib")

#define MKDIR(c) \
_mkdir((const char*)(c))


#elif defined (_WIN32)  //Windows 32 bit
//#include "stdafx.h"
#include <direct.h>
#include <windows.h>
#include <mmsystem.h>
#pragma comment (lib, "winmm.lib")

#define MKDIR(c) \
_mkdir((const char*)(c))

#endif



void getparameter();
void getparameter_inp(string inputfile_path);

void read_log(string fi, int);
void read_data_input(string di, string fi, float *, int);
void write_data_output(string di, string fi, float *, int);
void write_data_temp(string di, string fi, float *, int);
void first_image(string fi, float *, int);

int OSEM_ocl(OCL_platform_device plat_dev_list, float *ang, int ss);
int FBP_ocl(OCL_platform_device plat_dev_list, float *ang);
int AART_ocl(OCL_platform_device plat_dev_list, float *ang, int ss);
int FISTA_ocl(OCL_platform_device plat_dev_list, float *ang, int ss);
int hybrid_ocl(OCL_platform_device plat_dev_list, float *ang, int ss);
int OSEM_programBuild(cl::Context context,vector<cl::Kernel> *kernels);
int FBP_programBuild(cl::Context context,vector<cl::Kernel> *kernels);
int AART_programBuild(cl::Context context,vector<cl::Kernel> *kernels);
int FISTA_programBuild(cl::Context context,vector<cl::Kernel> *kernels);
int OSEM_execution(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,cl::Buffer angle_buffer, int *sub,cl::Image2DArray reconst_img, cl::Image2DArray prj_img, int dN, int it, bool prjCorrection);
int AART_execution(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,cl::Buffer angle_buffer, int *sub,cl::Image2DArray reconst_img, cl::Image2DArray prj_img,int dN,int it);
int FISTA_execution(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,cl::Buffer angle_buffer, cl::Buffer L2norm_buffer, int *sub,cl::Image2DArray reconst_img, cl::Image2DArray prj_img,int dN,int it);

int AART_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                cl::Buffer angle_buffer, int *sub, vector<float*> imgs, vector<float*> prjs,
                int startN, int endN, int it, int thread_id);
int OSEM_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
            cl::Buffer angle_buffer, int *sub, vector<float*> imgs, vector<float*> prjs,
                int startN, int endN, int it, int thread_id);
int FBP_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
               cl::Buffer angle_buffer,vector<float*> imgs, vector<float*> prjs,
               int startN, int endN,int thread_id);
int FISTA_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernel,
                cl::Buffer angle_buffer, cl::Buffer L2norm_buffer, int *sub,
                vector<float*> imgs, vector<float*> prjs,
                int startN, int endN, int it, int thread_id);

extern int numParallel;  //OpenCLデバイス当たりの並列数

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
extern int		g_ny;				/* 再構成ピクセルサイズ */
extern int		g_ox;				/* 再構成ピクセルサイズ */
extern int		g_oy;				/* 再構成ピクセルサイズ */
extern int		g_mode;				/* 再構成法 */
extern int		g_it;				/* 反復回数 */
extern int		g_num;			/* レイヤー数 */
extern int		g_st;			/* 開始ナンバー */
extern int		g_x;			/* 回転中心のずれ */
extern double	g_wt1;			/* AARTファクター */
extern double	g_wt2;			/* ASIRTファクター */
extern int		g_ss;				/* サブセット */
extern int		g_zp;				/* ゼロパディングサイズ */
extern float	*g_ang;					/* 投影角度 */
extern int		g_cover;			/* 非投影領域の推定 */
extern time_t	g_t0;					/* 時間 */
extern string   g_devList;            //OpenCLデバイスリスト
extern int numParallel; //OpenCLデバイス当たりの並列数
extern int correctionMode; //投影像補正 0:なし,1:x方向,2:θ方向,3:x+θ方向
extern float amp; //強度増幅因子
extern int baseupOrder;//baseup 減少速度次数
extern bool CSitBool; //圧縮センシング逐次計算
extern float CSepsilon; //圧縮センシング逐次計算ノイズファクター
extern float CSalpha; //圧縮センシング逐次計算加算ファクター
extern int CSit; //圧縮センシング逐次計算回数


#endif
