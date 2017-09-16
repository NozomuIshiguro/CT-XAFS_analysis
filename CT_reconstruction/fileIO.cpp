//
//  fileIO.cpp
//  CT_reconstruction
//
//  Created by Nozomu Ishiguro on 2015/06/21.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "CT_reconstruction.hpp"

void read_log(string fi, int num){
    int k;
	//cout << fi << endl;
    ifstream inp_ifs(fi,ios::in);
    if(!inp_ifs) {
        cerr << "file open error ["<< fi <<"]."<<endl;
        exit(1);
    }
    for (k = 0; k < num; k++) {
        string str=ifs_getline(&inp_ifs);
        istringstream iss(str);
        iss>>g_ang[k];
    }
}
void read_data_input(string di, string fi, float *img, int size){
    string fi2=di+"/"+fi;
    //cout<<fi2<<endl;
    ifstream inputstream(fi2,ios::in|ios::binary);
    if(!inputstream) {
        cerr << "file open error ["<< fi <<"]."<<endl;
        exit(1);
    }
    inputstream.read((char*)img, sizeof(float)*size);
}
void write_data_output(string di, string fi, float *img, int size){
    string fi2=di+"/"+fi;
    ofstream fout;
    fout.open(fi2, ios::out|ios::binary|ios::trunc);
    if(!fout){
        cerr << "file open error ["<< fi <<"]."<<endl;
        exit(1);
    }
    fout.write((char*)img,sizeof(float)*size);
}
void write_data_temp(string di, string fi, float *img, int size){
    string fi2=di+"/"+fi;
    ofstream fout;
    fout.open(fi2, ios::out|ios::binary|ios::trunc);
    if(!fout){
        cerr << "file open error ["<< fi <<"]."<<endl;
        exit(1);
    }
    fout.write((char*)img,sizeof(float)*size);
}
void first_image(string fi, float *img, int size){
    int i;
    switch (g_fst){
        case 1:
            read_data_input("", fi, img, size);
            break;
        case 2:
            for (i = 0; i < size; i++)
                img[i] = (float)g_ini;
            break;
        default:
            break;
    }
}
