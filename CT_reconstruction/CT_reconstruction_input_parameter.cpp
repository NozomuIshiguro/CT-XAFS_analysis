﻿//
//  CT_reconstruction_input_parameter.cpp
//  CT_reconstruction
//
//  Created by Nozomu Ishiguro on 2015/06/13.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "CT_reconstruction.hpp"

string menu[] = {
    " Open CL デバイス番号 (ex. 1,2-4)         ",
    " デバイス当たりの並列数                     ",
    " 測定ファイルディレクトリパス               ",
    " 測定ファイル名                           ",
    " 出力ファイルディレクトリパス               ",
    " 出力ファイル名                           ",
    " ログファイルパス                         ",
    " 一時ファイルディレクトリパス                ",
    " レイヤー数                              ",
    " 開始ナンバー                             ",
    " 初期画像の使用 1:Yes 2: No               ",
    " 初期画像ファイル名                        ",
    //	" ベタ塗の初期値                           ",
    " カメラピクセル数                         ",
    " 投影数                                 ",
    " 再構成画像サイズ                         ",
    //	" 回転中心のズレ                           ",
    "再構成法 1:AART 2:MART 3:ASIRT 4:MSIRT 5:MLEM 6:OSEM 7:FBP ",
    " 反復回数                                ",
    //	" ファクター(AART)                        ",
    //	" ファクター(ASIRT)                       ",
    " サブセット(OSEM)                        ",
    " ゼロパディングサイズ(FBP)                 ",
    //	" 非投影領域の推定 1:Yes 2:No              ",
    " 投影像補正(OSEM) 0:なし 1: x方向のみ 2:θ方向のみ 3:x,θ方向両方",
    " 再構成像強度増幅因子",
};

void getparameter_inp(string inputfile_path){
    ifstream inp_ifs(inputfile_path,ios::in);
    if(inp_ifs) {
        cout<<"Input file found."<<endl<<endl;
        while (!inp_ifs.eof()) {
            string str;
            str=ifs_getline(&inp_ifs);
            if(str=="#Open CL Platform & Device No"){
                cout << str << endl;
                str=ifs_getline(&inp_ifs);
                g_devList=str;
                cout << g_devList << endl << endl;
            }else if(str=="#Number of Parallel processing per one device"){
                cout << str << endl;
                inp_ifs>>numParallel;
                cout << numParallel << endl << endl;
            }else if(str=="#測定ファイルディレクトリパス"){
				cout << str << endl;
                str=ifs_getline(&inp_ifs);
                g_d1=str;
				cout << g_d1 << endl << endl;
            }else if(str=="#測定ファイル名"){
				cout << str << endl;
                str=ifs_getline(&inp_ifs);
                g_f1=str;
				cout << g_f1 << endl << endl;
            }else if(str=="#出力ファイルディレクトリパス"){
				cout << str << endl;
				str=ifs_getline(&inp_ifs);
                g_d2=str;
				cout << g_d2 << endl << endl;
            }else if(str=="#出力ファイル名"){
				cout << str << endl;
				str=ifs_getline(&inp_ifs);
                g_f2=str;
				cout << g_f2 << endl << endl;
            }else if(str=="#ログファイルパス"){
				cout << str << endl;
				str=ifs_getline(&inp_ifs);
                g_f3=str;
				cout << g_f3 << endl << endl;
            }else if(str=="#初期画像ファイル名"){
				cout << str << endl;
				str=ifs_getline(&inp_ifs);
                g_f4=str;
				cout << g_f4 << endl << endl;
            }else if(str=="#一時ファイルディレクトリパス"){
				cout << str << endl;
				str=ifs_getline(&inp_ifs);
                g_d5=str;
				cout << g_d5 << endl << endl;
            }else if(str=="#レイヤー数"){
				cout << str << endl;
				inp_ifs>>g_num;
				cout << g_num << endl;
            }else if(str=="#開始ナンバー"){
				cout << str << endl;
				inp_ifs>>g_st;
				cout << g_st << endl << endl;
            }else if(str=="#初期画像の使用 1:Yes 2: No"){
				cout << str << endl;
				inp_ifs>>g_fst;
				cout << g_fst << endl << endl;
            }else if(str=="#カメラピクセル数"){
				cout << str << endl;
				inp_ifs>>g_px;
				cout << g_px << endl << endl;
            }else if(str=="#投影数"){
				cout << str << endl;
				inp_ifs>>g_pa;
				cout << g_pa << endl << endl;
            }else if(str=="#再構成画像サイズ"){
				cout << str << endl;
				inp_ifs>>g_nx;
                g_ny = g_nx;
                g_ox = g_nx;
                g_oy = g_nx;
				cout << g_nx <<" x "<< g_ny << endl << endl;
            }else if(str=="#再構成画像サイズ X"){
                cout << str << endl;
                inp_ifs>>g_nx;
                cout << g_nx << endl << endl;
            }else if(str=="#再構成画像サイズ Y"){
                cout << str << endl;
                inp_ifs>>g_ny;
                cout << g_ny << endl << endl;
            }else if(str=="#出力画像サイズ X"){
                cout << str << endl;
                inp_ifs>>g_ox;
                cout << g_ox << endl << endl;
            }else if(str=="#出力画像サイズ Y"){
                cout << str << endl;
                inp_ifs>>g_oy;
                cout << g_oy << endl << endl;
            }else if(str=="#再構成法"){
				cout << str << endl;
				inp_ifs>>g_mode;
				cout << g_mode << endl << endl;
            }else if(str=="#反復回数"){
				cout << str << endl;
				inp_ifs>>g_it;
				cout << g_it << endl << endl;
            }else if (str == "#AARTファクター") {
				cout << str << endl;
				inp_ifs >> g_wt1;
				cout << g_wt1 << endl << endl;
			}else if(str=="#サブセット(OSEM)"){
				cout << str << endl;
				inp_ifs>>g_ss;
				cout << g_ss << endl << endl;
            }else if(str=="#ゼロパディングサイズ(FBP)"){
				cout << str << endl;
				inp_ifs>>g_zp;
				cout << g_zp << endl << endl;
            }else if(str=="#投影像補正 0:なし 1: x方向のみ 2:θ方向のみ 3:x,θ方向両方"){
                cout << str << endl;
                inp_ifs>>correctionMode;
                cout << correctionMode << endl << endl;
            }else if(str=="#再構成像強度増幅因子"){
                cout << str << endl;
                inp_ifs>>amp;
                cout << amp << endl << endl;
            }else if(str=="#Base up 減少速度次数 (Hybrid)"){
                cout << str << endl;
                inp_ifs>>baseupOrder;
                cout << baseupOrder << endl << endl;
            }else if(str=="#逐次圧縮センシング計算 (FBP適応外)"){
                cout << str << endl;
                int dummy;
                inp_ifs>>dummy;
                CSitBool= (dummy==1) ? true:false;
                cout << boolalpha <<CSitBool << endl << endl;
            }else if(str=="#逐次圧縮センシング計算加算ファクター"){
                cout << str << endl;
                inp_ifs>>CSlambda;
                cout << CSlambda << endl << endl;
            }else if(str=="#逐次圧縮センシング計算重複レイヤー数"){
                cout << str << endl;
                inp_ifs>>CSoverlap;
                cout << CSoverlap << endl << endl;
            }
            

        };
        inp_ifs.close();
    }else {
        cout<<"Input file not found."<<endl;
        cout<<"手動でパラメータを設定してください。"<<endl<<endl;
        
        getparameter();
    }
    
}

void getparameter(){
    int   i = 0;
    char  dat[256];
    cout << " "<< menu[i++] <<" ["<< g_devList <<"] :"; /* OpenCLデバイスリスト */
    cin.getline(dat, 256);
    cout << " "<< menu[i++] <<" ["<< numParallel <<"] :"; /* OpenCLデバイス当たりの並列数 */
    cin.getline(dat, 256);
    if (((string)dat).length()>0) numParallel = atoi(dat);
    cout << " "<< menu[i++] <<" ["<< g_d1 <<"] :"; /* 入力ファイルディレクトリパス */
    cin.getline(dat, 256);
    if (((string)dat).length()>0) g_d1 = dat;
    cout << " "<< menu[i++] <<" ["<< g_f1 <<"] :"; /* 入力ファイル名 */
    cin.getline(dat, 256);
    if (((string)dat).length()>0) g_f1 = dat;
    cout << " "<< menu[i++] <<" ["<< g_d2 <<"] :"; /* 出力ファイルディレクトリパス */
    cin.getline(dat, 256);
    if (((string)dat).length()>0) g_d2 = dat;
    cout << " "<< menu[i++] <<" ["<< g_f2 <<"] :"; /* 出力ファイル名 */
    cin.getline(dat, 256);
    if (((string)dat).length()>0) g_f2 = dat;
    cout << " "<< menu[i++] <<" ["<< g_f3 <<"] :"; /* logファイルパス */
    cin.getline(dat, 256);
    if (((string)dat).length()>0) g_f3 = dat;
    cout << " "<<menu[i++]<<" ["<<g_d5<<"] :";     /* 一時ファイルファイルディレクトリパス */
    cin.getline(dat, 256);
    if (((string)dat).length()>0) g_d5 = dat;
    cout << " "<<menu[i++]<<" ["<<g_num<<"] :";    /* レイヤー数 */
    cin.getline(dat, 256);
    if (((string)dat).length()>0) g_num = atoi(dat);
    cout << " "<<menu[i++]<<" ["<<g_st<<"] :";     /* 開始ナンバー */
    cin.getline(dat, 256);
    if (((string)dat).length()>0)  g_st = atoi(dat);
    cout << " "<<menu[i++]<<" ["<<g_fst<<"] :";    /* 初期画像の使用 */
    cin.getline(dat, 256);
    if (((string)dat).length()>0)  g_fst = atoi(dat);
    cout << " "<<menu[i++]<<" ["<<g_f4<<"] :";     /* 初期画像ファイル名 */
    cin.getline(dat, 256);
    if (((string)dat).length()>0) g_f4 = dat;
    //	cout << " "<<menu[i++]<<" ["<<g_ini<<"] :";/* 初期値 */
    //	cin.getline(dat, 256);
    //  if (((string)dat).length()>0)  g_ini = atof(dat);
    cout << " "<<menu[i++]<<" ["<<g_px<<"] :";     /* カメラピクセルサイズ */
    cin.getline(dat, 256);
    if (((string)dat).length()>0)  g_px = atoi(dat);
    cout << " "<<menu[i++]<<" ["<<g_pa<<"] :";     /* 投影数 */
    cin.getline(dat, 256);
    if (((string)dat).length()>0)  g_pa = atoi(dat);
    cout << " "<<menu[i++]<<" ["<<g_nx<<"] :";     /* 再構成ピクセルサイズ */
    cin.getline(dat, 256);
    if (((string)dat).length()>0)  {
        g_nx = atoi(dat);
        g_ny = atoi(dat);
    }
    //	cout << " "<<menu[i++]<<" ["<<g_x<<"] :";  /* 回転中心のずれ */
    //	cin.getline(dat, 256);
    //  if (((string)dat).length()>0)  g_x = atoi(dat);
    cout << " "<<menu[i++]<<" ["<<g_mode<<"] :";/* 再構成法 */
    cin.getline(dat, 256);
    if (((string)dat).length()>0)  g_mode = atoi(dat);
    cout << " "<<menu[i++]<<" ["<<g_it<<"] :";      /* 反復回数 */
    cin.getline(dat, 256);
    if (((string)dat).length()>0)  g_it = atoi(dat);
    //	cout << " "<<menu[i++]<<" ["<<g_wt1<<"] :"; /* AARTファクター */
    //	cin.getline(dat, 256);
    //  if (((string)dat).length()>0)  g_wt1 = atoi(dat);
    //	cout << " "<<menu[i++]<<" ["<<g_wt2<<"] :"; /* ASIRTファクター */
    //	cin.getline(dat, 256);
    //  if (((string)dat).length()>0)  g_wt2 = atoi(dat);
    cout << " "<<menu[i++]<<" ["<<g_ss<<"] :";      /* サブセット */
    cin.getline(dat, 256);
    if (((string)dat).length()>0)  g_ss = atoi(dat);
    cout << " "<<menu[i++]<<" ["<<g_zp<<"] :";      /* ゼロパディングサイズ */
    cin.getline(dat, 256);
    if (((string)dat).length()>0)  g_zp = atoi(dat);
    //	cout << " "<<menu[i++]<<" ["<<g_cover<<"] :"; /* 非投影領域の推定 */
    //	cin.getline(dat, 256);
    //  if (((string)dat).length()>0)  g_cover = atoi(dat);
    cout << " "<<menu[i++]<<" ["<<correctionMode<<"] :";      /* 投影像補正 */
    cin.getline(dat, 256);
    if (((string)dat).length()>0)  correctionMode = atoi(dat);
}
