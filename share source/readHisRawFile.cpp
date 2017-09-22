//
//  readHisFile.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2014/12/17.
//  Copyright (c) 2014 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"

int readRawFile(string filepath_input,float *binImgf,int imageSizeM){
    
    //cout<<filepath_input;
    ifstream inputstream(filepath_input,ios::in|ios::binary);
    if(!inputstream) {
        cerr << "   Failed to load raw file: " <<endl;
        cerr<< filepath_input << endl;
        return -1;
    }
    
    inputstream.read((char*)binImgf, sizeof(float)*imageSizeM);
    
    return 0;
}

int readRawFile(string filepath_input,float *binImgf, int startnum, int endnum,int imageSizeM){
    
    //cout<<filepath_input;
    ifstream inputstream(filepath_input,ios::in|ios::binary);
    if(!inputstream) {
        cerr << "   Failed to load raw file: " <<endl;
        cerr<< filepath_input << endl;
        return -1;
    }
    
    
    int64_t offset = sizeof(float)*imageSizeM*(int64_t)(startnum - 1);
    inputstream.seekg(offset,ios_base::cur);
    
    int64_t bufferpnts = imageSizeM*(int64_t)(endnum - startnum + 1);
    int64_t buffersize = bufferpnts*sizeof(float);
    
    inputstream.read((char*)binImgf, buffersize);
    
    return 0;
}

int readRawFile_offset(string filepath_input,float *binImgf, int64_t offset, int64_t size){
    
    //cout<<filepath_input;
    ifstream inputstream(filepath_input,ios::in|ios::binary);
    if(!inputstream) {
        cerr << "   Failed to load raw file: " <<endl;
        cerr<< filepath_input << endl;
        return -1;
    }
    
    
    inputstream.seekg(offset,ios_base::cur);
    inputstream.read((char*)binImgf, size);
    
    return 0;
}

int readHisFile_stream(string filename, int startnum, int endnum, unsigned short *binImgf,int imageSizeM)
{
    ifstream inputstream(filename,ios::in|ios::binary);
    if(!inputstream) {
        cerr << "   Failed to load his: "<< filename << endl;
        return -1;
    }
 
    if ((int)inputstream.tellg()==0) {
        string header;

		inputstream.seekg(2, ios_base::cur);
		unsigned short offset0;
		inputstream.read((char*)&offset0, sizeof(unsigned short));
		offset0 += 64;

        /*for (int i=0; i<5; i++) {
            getline(inputstream,header);
        }
        inputstream.seekg(2,ios_base::cur);*/
		inputstream.seekg(offset0, ios_base::cur);
        if(startnum>1) {
			int64_t offset = sizeof(unsigned short)*(imageSizeM + 32)*(int64_t)(startnum - 1);
            inputstream.seekg(offset,ios_base::cur);
        }
		//cout << inputstream.tellg() << endl;
    }

	int64_t bufferpnts = (imageSizeM + 32)*(int64_t)(endnum - startnum + 1);
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

int outputRawFile_stream_batch(string filename,vector<float*> data, size_t pnt_size){
    ofstream fout;
    fout.open(filename, ios::out|ios::binary|ios::trunc);
    
    for (int i=0; i<data.size(); i++) {
        fout.write((char*)data[i],sizeof(float)*pnt_size);
    }
    fout.close();
    
    return 0;
}
