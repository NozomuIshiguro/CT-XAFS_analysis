//
//  CT_reconstruction_input_parameter.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/06/13.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "CT_reconstruction.hpp"

string menu[] = {
    " Open CL デバイス番号 (ex. 1,2-4)         ",
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
    //	" 1:AART 2:MART 3:ASIRT 4:MSIRT 5:MLEM 6:OSEM 7:FBP ",
    " 反復回数                                ",
    //	" ファクター(AART)                        ",
    //	" ファクター(ASIRT)                       ",
    " サブセット(OSEM)                        ",
    //	" 非投影領域の推定 1:Yes 2:No              ",
};

void getparameter_inp(string inputfile_path){
    
    ifstream inp_ifs(inputfile_path,ios::in);
    if(inp_ifs) {
        cout<<"Input file found."<<endl<<endl;
        while (!inp_ifs.eof()) {
            char *buffer;
            buffer = new char[256];
            inp_ifs.getline(buffer, 256);
            if((string)buffer=="#Open CL Platform & Device No"){
                inp_ifs.getline(buffer, 256);
                g_devList=buffer;
            }else if((string)buffer=="#測定ファイルディレクトリパス"){
                inp_ifs.getline(buffer, 256);
                g_d1=buffer;
            }else if((string)buffer=="#測定ファイル名"){
                inp_ifs.getline(buffer, 256);
                g_f1=buffer;
            }else if((string)buffer=="#出力ファイルディレクトリパス"){
                inp_ifs.getline(buffer, 256);
                g_d2=buffer;
            }else if((string)buffer=="#出力ファイル名"){
                inp_ifs.getline(buffer, 256);
                g_f2=buffer;
            }else if((string)buffer=="#ログファイルパス"){
                inp_ifs.getline(buffer, 256);
                g_f3=buffer;
            }else if((string)buffer=="#初期画像ファイル名"){
                inp_ifs.getline(buffer, 256);
                g_f4=buffer;
            }else if((string)buffer=="#一時ファイルディレクトリパス"){
                inp_ifs.getline(buffer, 256);
                g_d5=buffer;
            }else if((string)buffer=="#レイヤー数"){
                inp_ifs>>g_num;
            }else if((string)buffer=="#開始ナンバー"){
                inp_ifs>>g_st;
            }else if((string)buffer=="#初期画像の使用 1:Yes 2: No"){
                inp_ifs>>g_fst;
            }else if((string)buffer=="#カメラピクセル数"){
                inp_ifs>>g_px;
            }else if((string)buffer=="#投影数"){
                inp_ifs>>g_pa;
            }else if((string)buffer=="#再構成画像サイズ"){
                inp_ifs>>g_nx;
            }else if((string)buffer=="#反復回数"){
                inp_ifs>>g_it;
            }else if((string)buffer=="#サブセット(OSEM)"){
                inp_ifs>>g_ss;
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
    if (((string)dat).length()>0)  g_nx = atoi(dat);
    //	cout << " "<<menu[i++]<<" ["<<g_x<<"] :";  /* 回転中心のずれ */
    //	cin.getline(dat, 256);
    //  if (((string)dat).length()>0)  g_x = atoi(dat);
    //	cout << " "<<menu[i++]<<" ["<<g_mode<<"] :";/* 再構成法 */
    //	cin.getline(dat, 256);
    //  if (((string)dat).length()>0)  g_mode = atoi(dat);
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
    //	cout << " "<<menu[i++]<<" ["<<g_cover<<"] :"; /* 非投影領域の推定 */
    //	cin.getline(dat, 256);
    //  if (((string)dat).length()>0)  g_cover = atoi(dat);
}
