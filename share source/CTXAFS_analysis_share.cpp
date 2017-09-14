//
//  CTXAFS_analysis_share.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/03/06.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"

static int buffersize=512;

string ifs_getline(ifstream *inp_ifs){
    
    char *buffer;
    buffer = new char[buffersize];
    (*inp_ifs).getline(buffer, buffersize);
    
    string str=buffer;
    if (str.find('\r')!=string::npos) {
        str.erase(str.end()-1);
    }
    
    
    return str;
}

string output_flag(string flag, int argc, const char * argv[]){
    
    string output;
    for (int i=1; i<argc-1; i++) {
        string argv_str=argv[i];
        if(argv_str==flag){
            output=argv[i+1];
            break;
        }
    }
    
    return output;
}

string IntToString(int number)
{
    stringstream ss;
    ss << number;
    return ss.str();
}

string LnumTagString(int LoopNo,string preStr, string postStr){
    
    string LnumTag;
    if (LoopNo<10) {
        LnumTag=preStr+"0"+IntToString(LoopNo)+postStr;
    } else {
        LnumTag=preStr+IntToString(LoopNo)+postStr;
    }
    return LnumTag;
}

string numTagString(int tagNum, string preStr, string postStr, int degit) {

	string tagStr,zeroStr;
	
	int tagDegit = 0;
	int degitNum = 1;
	do {
		degitNum *= 10;
		tagDegit++;
	} while (tagNum >= degitNum);

	if (tagDegit >= degit) {
		tagStr = preStr + IntToString(tagNum) + postStr;
	} else {
		zeroStr = "";
		for (int j = 0; j < degit-tagDegit; j++) {
			zeroStr += "0";
		}
		tagStr = preStr + zeroStr + IntToString(tagNum) + postStr;
	}

	return tagStr;
}

string EnumTagString(int EnergyNo,string preStr, string postStr){
    
    string EnumTag;
    if (EnergyNo<10) {
        EnumTag=preStr+"00"+IntToString(EnergyNo)+postStr;
    } else if(EnergyNo<100){
        EnumTag=preStr+"0"+IntToString(EnergyNo)+postStr;
    } else {
        EnumTag=preStr+IntToString(EnergyNo)+postStr;
    }
    return EnumTag;
}

string AnumTagString(int angleNo,string preStr, string postStr){
    
    string angleNumTag;
    if (angleNo<10) {
        angleNumTag = preStr+"000"+IntToString(angleNo)+postStr;
    } else if(angleNo<100){
        angleNumTag = preStr+"00"+IntToString(angleNo)+postStr;
    } else if(angleNo<1000){
        angleNumTag = preStr+"0"+IntToString(angleNo)+postStr;
    } else {
        angleNumTag = preStr+IntToString(angleNo)+postStr;
    }
    return angleNumTag;
}

CL_objects::CL_objects(){
    
}

cl::Kernel CL_objects::getKernel(string kernelName){
    
	int i;
    for (i=0; i<kernels.size(); i++) {
        string name = kernels[i].getInfo<CL_KERNEL_FUNCTION_NAME>();
		//cout << name << endl;
		//cout << name.find(kernelName) << endl;
		if (name.find(kernelName)==0) {
			//cout << name << endl;
			return kernels[i];
			break;
		}
    }
    
    return kernels[i-1];
}


void CL_objects::addKernel(cl::Program program, string kernelName){
    kernels.push_back(cl::Kernel(program,kernelName.c_str()));
}


int createContrainMatrix(vector<string> contrain_eqs, vector<string> fparaName,
                         vector<vector<float>> *C_matrix, vector<float> *D_vector,int cotrainOffset){
    
    int num_contrain=cotrainOffset;
    int num_fpara = (int)fparaName.size();
    for (int i=0; i<contrain_eqs.size(); i++) {
        string eqstr = contrain_eqs[i];
        
        //replace paraName string
        for (int j=0; j<num_fpara; j++) {
            ostringstream pname;
            pname <<"p"<<j;
            string pnamestr =pname.str();
            
            size_t  pos = eqstr.find(fparaName[j]);
            while (pos!=string::npos) {
                eqstr.replace(pos, fparaName[j].size(), pnamestr);
                pos = eqstr.find(fparaName[j], pos+pnamestr.size());
            }
        }
        
        
        // erase " "
        size_t pos = eqstr.find(" ");
        while (pos!=string::npos) {
            eqstr.replace(pos, 1, "");
            pos = eqstr.find(" ", pos);
        }
        
        
        //separate eqstr at "<>="
        string eqstr1, eqstr2;
        int eqsymbol=-1;
        pos = eqstr.find("<");
        if (pos!=string::npos) {
            eqstr1 = eqstr.substr(0,pos);
            eqstr2 = eqstr.substr(pos+1,eqstr.size());
            eqsymbol=0;
        }
        pos = eqstr.find(">");
        if (pos!=string::npos) {
            eqstr1 = eqstr.substr(0,pos);
            eqstr2 = eqstr.substr(pos+1,eqstr.size());
            eqsymbol=1;
        }
        pos = eqstr.find("=");
        if (pos!=string::npos) {
            eqstr1 = eqstr.substr(0,pos);
            eqstr2 = eqstr.substr(pos+1,eqstr.size());
            eqsymbol=2;
        }
        
        
        //replace "-" with "+-"
        pos = eqstr1.find("-");
        while (pos!=string::npos) {
            eqstr1.replace(pos, 1, "+-");
            pos = eqstr1.find("-", pos+2);
        }
        pos = eqstr2.find("-");
        while (pos!=string::npos) {
            eqstr2.replace(pos, 1, "+-");
            pos = eqstr2.find("-", pos+2);
        }
        
        
        //separate eqstr1, eqstr2 into terms
        vector<string> terms1, terms2;
        pos = eqstr1.find("+");
        while (pos!=string::npos) {
            terms1.push_back(eqstr1.substr(0,pos));
            eqstr1.erase(0,pos+1);
            pos = eqstr1.find("+");
        }
        terms1.push_back(eqstr1.substr(0,eqstr1.size()));
        pos = eqstr2.find("+");
        while (pos!=string::npos) {
            terms2.push_back(eqstr2.substr(0,pos));
            eqstr2.erase(0,pos+1);
            pos = eqstr2.find("+");
        }
        terms2.push_back(eqstr2.substr(0,eqstr2.size()));
        
        
        //move terms if there is p* terms in terms2
        for (int j=0; j<terms2.size();j++) {
            for (int k=0; k<num_fpara; k++) {
                ostringstream pname;
                pname <<"p"<<k;
                string pnamestr =pname.str();
                
                pos = terms2[j].find(pnamestr);
                if (pos!=string::npos) {
                    string term="-"+terms2[j];
                    size_t pos2 = term.find("--");
                    while (pos2!=string::npos) {
                        term.erase(pos2,pos2+2);
                        pos2 = term.find("--");
                    }
                    
                    terms1.push_back(term);
                    terms2[j]="";
                }
            }
        }
        
        
        //symbol
        string symbol;
        switch (eqsymbol) {
            case 0:
                symbol="<";
                (*C_matrix).push_back(vector<float>(num_fpara,0.0f));
                (*D_vector).push_back(0.0f);
                break;
            case 1:
                symbol=">";
                (*C_matrix).push_back(vector<float>(num_fpara,0.0f));
                (*D_vector).push_back(0.0f);
                break;
            case 2:
                symbol="=";
                (*C_matrix).push_back(vector<float>(num_fpara,0.0f));
                (*D_vector).push_back(0.0f);
                (*C_matrix).push_back(vector<float>(num_fpara,0.0f));
                (*D_vector).push_back(0.0f);
                break;
                
            default:
                break;
        }
        
        //D vector
        for (int j=0; j<terms1.size(); j++) {
            for (int k=0; k<num_fpara; k++) {
                ostringstream pname;
                pname <<"p"<<k;
                string pnamestr =pname.str();
                
                string term=terms1[j];
                pos = term.find(pnamestr);
                if (pos!=string::npos) {
                    term.erase(pos, pos+pnamestr.size());
                    
                    float a=1.0f;
                    float b=1.0f;
                    size_t pos2 = term.find("*");
                    while (pos2!=string::npos) {
                        string term2 = term.substr(0,pos2);
                        term.erase(0,pos2+1);
                        
                        b=1.0f;
                        if(term2.size()==0){
                            b=1.0f;
                        }else if(term2=="-"){
                            b=-1.0f;
                        }else{
                            istringstream iss(term2);
                            iss>>b;
                        }
                        a *= b;
                        pos2 = term.find("*");
                    }
                    b=1.0f;
                    if(term.size()==0){
                        b=1.0f;
                    }else if(term=="-"){
                        b=-1.0f;
                    }else{
                        istringstream iss(term);
                        iss>>b;
                    }
                    a *= b;
                    
                    switch (eqsymbol) {
                        case 0:
                            (*C_matrix)[num_contrain][k]+=a;
                            break;
                        case 1:
                            (*C_matrix)[num_contrain][k]-=a;
                            break;
                        case 2:
                            (*C_matrix)[num_contrain][k]  +=a;
                            (*C_matrix)[num_contrain+1][k]-=a;
                            break;
                            
                        default:
                            break;
                    }
                    break;
                }
            }
        }
        //D vector
        for (int j=0; j<terms2.size(); j++) {
            istringstream iss(terms2[j]);
            float a=0.0f;
            iss>>a;
            
            switch (eqsymbol) {
                case 0:
                    (*D_vector)[num_contrain]+=a;
                    break;
                case 1:
                    (*D_vector)[num_contrain]-=a;
                    break;
                case 2:
                    (*D_vector)[num_contrain]  +=a;
                    (*D_vector)[num_contrain+1]-=a;
                    break;
                    
                default:
                    break;
            }
        }
        
        
        //output
        /*for (int j=0; j<terms1.size()-1; j++) {
         cout << terms1[j]<<"+";
         }
         cout << terms1[terms1.size()-1]<<symbol;
         for (int j=0; j<terms2.size()-1; j++) {
         if(terms2[j+1].length()>0) cout << terms2[j]<<"+";
         else cout << terms2[j];
         }
         cout << terms2[terms2.size()-1]<<endl;*/
        
        
        switch (eqsymbol) {
            case 0:
                num_contrain++;
                break;
            case 1:
                num_contrain++;
                break;
            case 2:
                num_contrain+=2;
                break;
                
            default:
                break;
        }
    }
    
    /*cout<<endl<<"C_matrix | D_vector"<<endl;
     for (int i=0; i<num_contrain; i++) {
     for (int j=0; j<num_fpara-1; j++) {
     cout << C_matrix[i][j] <<" ";
     }
     cout<< C_matrix[i][num_fpara-1] <<" | "<< D_vector[i] << endl;
     }*/
    
    return num_contrain;
}
