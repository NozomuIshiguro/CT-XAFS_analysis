//
//  readHisFile.c
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2014/12/17.
//  Copyright (c) 2014 Nozomu Ishiguro. All rights reserved.
//
#include "CTXAFS.h"

int IMAGE_SIZE_X = 2048;
int IMAGE_SIZE_Y = 2048;

const int angleN=1600;

int readHisFile(FILE *fp, int startnum, int endnum, float *binImgf)
{
    
    if (ftell(fp)==0) {
        char header[2];
        int i=0;
        int kaigyo=0;
        do {
            fgets(&*header, 2, fp);
            if (*header=='\r') {
                kaigyo++;
            }
            i++;
        } while (kaigyo<5);
        fseek(fp,3,SEEK_CUR);
        if(startnum>1) fseek(fp,(IMAGE_SIZE_X*IMAGE_SIZE_Y*sizeof(short)+64)*(startnum-1),SEEK_CUR);
    }
    
    //printf("    offset: %d\n",(unsigned int)ftell(fp));
    short int *hisbin;
    hisbin = (short int *) malloc(IMAGE_SIZE_X*IMAGE_SIZE_Y*sizeof(short));
    for (int i=startnum-1; i<endnum; i++) {
        fread(hisbin, sizeof(short), IMAGE_SIZE_X*IMAGE_SIZE_Y, fp);
        if (feof(fp)) {
            //printf("%d\n",i);
            break;
        }
        
        for (int j=0; j<IMAGE_SIZE_X*IMAGE_SIZE_Y; j++) {
            float val;
            val = *(binImgf+j)+(float)hisbin[j]/(endnum-startnum+1);
            *(binImgf+j) = val;
            //binImgf[j] += (float)hisbin[j]/(endnum-startnum+1);
        }
        
        
        fseek(fp,64,SEEK_CUR);
        if (feof(fp)) {
            break;
        }
    }
    
    free(hisbin);
    //printf("%d\r",hisbin[0]);
    return 0;
}

int raw2mt(char *fileName_base, char *output_dir) {
    // insert code here...
    
	#if defined (WIN32) || defined (_M_X64)
		mkdir(output_dir);
	#else
		mkdir(output_dir, 0755);
	#endif
    
    /* dark input */
    printf("Processing dark...\n");
    /*his file read*/
    char fileName_dark[128];
    strcpy(fileName_dark, fileName_base);
    strcat(fileName_dark,"dark.his");
    FILE *fp_dark;
    //printf("%s\r",fileName_his);
    fp_dark = fopen(fileName_dark, "rb");
    printf("    loading %s\n",fileName_dark);
    if (!fp_dark) {
        fprintf(stderr, "   Failed to load his \n");
        exit(1);
    }
    float *dark_img;
    dark_img = (float *) malloc(IMAGE_SIZE_X*IMAGE_SIZE_Y*sizeof(float));
    readHisFile(fp_dark,1,30,dark_img);
    fclose(fp_dark);
    
    /* file output path*/
    /*char fileName_output[] = "/Volumes/Ishiguro's SSD/fresh_0.4V_2/fresh_0.4V_2_dark.raw";
     FILE *fp_output;
     printf("%s\r",fileName_output);
     fp_output = fopen(fileName_output, "wb");
     fwrite(dark_img, sizeof(float), 2048*2048, fp_output);
     fclose(fp_output);*/
    
    
    int i=1;
    do {
        char numTag[10];
        if (i<10) {
            sprintf(numTag, "00%d",i);
        } else if(i<100){
            sprintf(numTag, "0%d",i);
        } else {
            sprintf(numTag, "%d",i);
        }
        
        /* I0 input */
        char fileName_I0[128];
        strcpy(fileName_I0, fileName_base);
        strcat(fileName_I0, numTag);
        strcat(fileName_I0, "_I0.his");
        FILE *fp_I0;
        fp_I0 = fopen(fileName_I0, "rb");
        if (!fp_I0) {
            break;
        }
        printf("Processing I0 ...\n");
        printf("    loading %s\n",fileName_I0);
        float *I0_img;
        I0_img = (float *) malloc(IMAGE_SIZE_X*IMAGE_SIZE_Y*sizeof(float));
        readHisFile(fp_I0,1,20,I0_img);
        
        fclose(fp_I0);
        
        /*file output path*/
        char fileName_output[] = "/Volumes/Ishiguro's SSD/fresh_0.4V_2/fresh_0.4V_2_I0.raw";
         FILE *fp_output;
         printf("%s\r",fileName_output);
         fp_output = fopen(fileName_output, "wb");
         fwrite(I0_img, sizeof(float), IMAGE_SIZE_X*IMAGE_SIZE_Y, fp_output);
         fclose(fp_output);
        
        
        
        /*It input*/
        char fileName_It[128];
        strcpy(fileName_It, fileName_base);
        strcat(fileName_It, numTag);
        strcat(fileName_It, ".his");
        FILE *fp_It;
        fp_It = fopen(fileName_It, "rb");
        if (!fp_It) {
            break;
        }
        printf("Processing It ...\n");
        char outputAngleDir[128];
        strcpy(outputAngleDir, output_dir);
        strcat(outputAngleDir, numTag);
        strcat(outputAngleDir, "/");
        mkdir(outputAngleDir, 0755);
		#if defined (WIN32) || defined (_M_X64)
			mkdir(outputAngleDir);
		#else
			mkdir(outputAngleDir, 0755);
		#endif
        
        printf("    loading %s\n",fileName_It);
        float *mt_img;
        mt_img = (float *) malloc(IMAGE_SIZE_X*IMAGE_SIZE_Y*sizeof(float));
        for (int j=1; j<angleN+1; j++) {
            //printf("%d ",j);
            readHisFile(fp_It,j,j,mt_img);
            for (int k=0; k<IMAGE_SIZE_X*IMAGE_SIZE_Y; k++) {
                mt_img[k] = log((I0_img[k]-dark_img[k])/(mt_img[k]-dark_img[k]));
            }
            
            /*file output path*/
            char fileName_output[128];
            char angleNumTag[15];
            if (j<10) {
                sprintf(angleNumTag, "mt000%d.raw",j);
            } else if(j<100){
                sprintf(angleNumTag, "mt00%d.raw",j);
            } else if(i<1000){
                sprintf(angleNumTag, "mt0%d.raw",j);
            } else {
                sprintf(angleNumTag, "mt%d.raw",j);
            }
            strcpy(fileName_output, outputAngleDir);
            strcat(fileName_output, angleNumTag);
            FILE *fp_output;
            fp_output = fopen(fileName_output, "wb");
            fwrite(mt_img, sizeof(float), IMAGE_SIZE_X*IMAGE_SIZE_Y, fp_output);
            fclose(fp_output);
        }
        fclose(fp_It);
        
        i++;
    } while (1);
    
    return 0;
}
