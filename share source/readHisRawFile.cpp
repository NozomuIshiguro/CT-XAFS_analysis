//
//  readHisFile.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2014/12/17.
//  Copyright (c) 2014 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"

int readRawFile(string filepath_input,float *binImgf){
    
    //cout<<filepath_input;
    ifstream inputstream(filepath_input,ios::in|ios::binary);
    if(!inputstream) {
        cerr << "   Failed to load raw file: " <<endl;
        cerr<< filepath_input << endl;
        return -1;
    }
    
    inputstream.read((char*)binImgf, sizeof(float)*IMAGE_SIZE_M);
    
    return 0;
}

int readHisFile_stream(string filename, int startnum, int endnum, unsigned short *binImgf)
{
    ifstream inputstream(filename,ios::in|ios::binary);
    if(!inputstream) {
        cerr << "   Failed to load his" << endl;
        return -1;
    }
 
    if ((int)inputstream.tellg()==0) {
        string header;
        for (int i=0; i<5; i++) {
            getline(inputstream,header);
        }
        inputstream.seekg(2,ios_base::cur);
        if(startnum>1) {
			int64_t offset = sizeof(unsigned short)*(IMAGE_SIZE_M + 32)*(int64_t)(startnum - 1);
            inputstream.seekg(offset,ios_base::cur);
        }
		//cout << inputstream.tellg() << endl;
    }

	int64_t bufferpnts = (IMAGE_SIZE_M + 32)*(int64_t)(endnum - startnum + 1);
	int64_t buffersize = bufferpnts*sizeof(unsigned short);
	inputstream.read((char*)binImgf, buffersize);
    
    //printf("%d\r",hisbin[0]);
    return 0;
}

int outputRawFile_stream(string filename,float *data, size_t pnt_size){
    ofstream fout;
    fout.open(filename, ios::out|ios::binary|ios::trunc);
    
    fout.write((char*)data,sizeof(float)*pnt_size);
    fout.close();
    
    return 0;
}
