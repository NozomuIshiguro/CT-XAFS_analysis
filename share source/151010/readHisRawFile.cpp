//
//  readHisFile.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2014/12/17.
//  Copyright (c) 2014 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"
//#include "CTXAFS.hpp"

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

int readHisFile_stream(string filename, int startnum, int endnum, float *binImgf, size_t shift)
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
			int64_t offset = (IMAGE_SIZE_M*sizeof(unsigned short) + 64)*(startnum - 1);
            inputstream.seekg(offset,ios_base::cur);
        }
		//cout << inputstream.tellg() << endl;
    }

	for (int j = 0; j < IMAGE_SIZE_M; j++) {
		binImgf[j] = 0.0;
	}

	int64_t bufferpnts = (IMAGE_SIZE_M + 32)*(endnum - startnum + 1)-32;
	int64_t buffersize = bufferpnts*sizeof(unsigned short);
	unsigned short *hisbin;
	hisbin = new unsigned short[bufferpnts];
	inputstream.read((char*)hisbin, buffersize);
    
    //printf("    offset: %d\n",(unsigned int)ftell(fp));
    for(int i=startnum; i<=endnum; i++) {
                
        for (int j=0; j<IMAGE_SIZE_M; j++) {
            float val=0.0;
			int64_t numpnts = j + (IMAGE_SIZE_M + 32)*(i - startnum);
			if (shift==0) {
				val = binImgf[j] + (float)hisbin[numpnts] / (endnum - startnum + 1);
            } else {
                val = (float)hisbin[numpnts];
            }
			//cout << "val:" << val << endl;
            binImgf[j+shift*(i-startnum)] = val;
        }
    }
	delete[] hisbin;
    
    
    
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
