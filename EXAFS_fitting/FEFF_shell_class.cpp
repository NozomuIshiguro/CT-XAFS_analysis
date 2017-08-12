//
//  FEFF_shell_class.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2017/07/28.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#include "EXAFS.hpp"
static int buffsize=512;

FEFF_shell::FEFF_shell(string path){
    ifstream feff_ifs(path,ios::in);
    
    //seeking header
    while (!feff_ifs.eof()) {
        char *buffer;
        buffer = new char[buffsize];
        feff_ifs.getline(buffer, buffsize);
        if((string)buffer==" -----------------------------------------------------------------------"){
            break;
        }/*else{
            cout<<buffer<<endl;;
        }*/
    }
    
    //reading Reff
    string nleg_s, deg_s, reff_s, rnrmav_s, edge_s;
    float nleg, rnrmav, edge;
    feff_ifs >> nleg >> degen >> reff >> rnrmav >> edge;
    //cout << "nleg, deg, reff, rnrmav, edge" << endl;
    //cout << nleg << "," << degen << "," << reff <<"," << rnrmav << "," << edge;
    
    //seeking header
    while (!feff_ifs.eof()) {
        char *buffer;
        buffer = new char[buffsize];
        feff_ifs.getline(buffer, buffsize);
        
        if((string)buffer=="    k   real[2*phc]   mag[feff]  phase[feff] red factor   lambda     real[p]@#"){
            break;
        }/*else{
            cout<<buffer<<endl;;
        }*/
    }
    
    
    numPnts=0;
    int j = 0;
    //cout << "k, real2_phc, mag, phase, red factor, lambda, real_p" << endl;
    do {
        
        if (feff_ifs.eof()) break;
        string a1,a2,a3,a4,a5,a6,a7;
        float aa1,aa2,aa3,aa4,aa5,aa6,aa7;
        feff_ifs >> aa1 >> aa2 >> aa3 >>aa4 >> aa5 >> aa6 >> aa7;
        kw.push_back(aa1);
        real2phc.push_back(aa2);
        mag.push_back(aa3);
        phase.push_back(aa4);
        redFactor.push_back(aa5);
        lambda.push_back(aa6);
        real_p.push_back(aa7);
        //cout<<kw[j]<<","<<real2phc[j]<<","<<mag[j]<<","<<phase[j];
        //cout<<redFactor[j]<<","<<lambda[j]<<","<<real_p[j]<<endl;
        numPnts++;
        j++;
    } while (!feff_ifs.eof());
    feff_ifs.close();
}

vector<float> FEFF_shell::getk(){
    return kw;
}

vector<float> FEFF_shell::getReal2phc(){
    return real2phc;
}

vector<float> FEFF_shell::getMag(){
    return mag;
}

vector<float> FEFF_shell::getPhase(){
    return phase;
}

vector<float> FEFF_shell::getRedFactor(){
    return redFactor;
}

vector<float> FEFF_shell::getLambda(){
    return lambda;
}

vector<float> FEFF_shell::getReal_p(){
    return real_p;
}

float FEFF_shell::getReff(){
    return reff;
}

float FEFF_shell::getDegen(){
    return degen;
}

int FEFF_shell::getNumPnts(){
    return numPnts;
}
