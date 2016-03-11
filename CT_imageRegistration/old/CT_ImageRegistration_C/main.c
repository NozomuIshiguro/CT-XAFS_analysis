//
//  main.c
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2014/12/17.
//  Copyright (c) 2014-2015 Nozomu Ishiguro. All rights reserved.
//

#include "CTXAFS.h"

int main() {
    
    
    int plat_id_num;
    int dev_id_num;
    OCL_device_list(&plat_id_num,&dev_id_num);
    
    
    /* Input directory settings*/
    char input_dir_path[128];
    puts("Set input raw his file directory.");
    scanf("%s",input_dir_path); // /Volumes/IshiguroHDD/fresh_0.4V_2
    char fileName_base[128];
    strcpy(fileName_base, input_dir_path);
    strcat(fileName_base, "/");
    
    DIR *dir;
    struct dirent *dp;
    dir=opendir(input_dir_path);
    
    for(dp=readdir(dir);dp!=NULL;dp=readdir(dir)){
        //printf("%s\n",dp->d_name);
        char *darkname = dp->d_name;
        char *ret;
        ret=strstr(darkname, "dark.his");
        if (ret !=NULL) {
            printf("    his file found: %s\n",darkname);
            strncat(fileName_base, darkname, strlen(darkname)-8);
            break;
        }
    }
    if (strstr(dp->d_name, "dark.his")==NULL) {
        puts("No his file found.");
        return -1;
    }
    closedir(dir);
    //printf("%s\n",fileName_base);*
    
    /*output directory settings*/
    char output_dir_path[128];
    puts("Set output file directory.");
    scanf("%s",output_dir_path); // /Volumes/IshiguroHDD/fresh_0.4V_2/mtr
#if defined (WIN32) || defined (_M_X64)
    mkdir(output_dir_path);
#else
    mkdir(output_dir_path, 0755);
#endif
    /*processing energy and angles*/
    int start_E,end_E,start_A,end_A,target_E;
    puts("Set energy num range (ex. 1-100).");
    scanf("%d-%d",&start_E,&end_E);
    puts("Set reference energy No. for image registration.");
    scanf("%d",&target_E);
    puts("Set angle num range (ex. 1-1600).");
    scanf("%d-%d",&start_A,&end_A);
    //printf("%d,%d,%d,%d,%d",start_E,end_E,target_E,start_A,end_A);
    
    puts("");
    time_t start,end;
    time(&start);
    
    /*Image Registration*/
    imageRegistlation_ocl(fileName_base, output_dir_path, start_E,end_E,target_E,start_A,end_A,0,&plat_id_num,&dev_id_num);
    
    
    
    
    time(&end);
    double delta_t = difftime(end,start);
    int day, hour, min;
    double sec;
    day = (int)floor(delta_t/24/60/60);
    hour = (int)floor((delta_t-(double)(day*24*60*60))/60/60);
    min = (int)floor((delta_t-(double)(day*24*60*60)-(double)(hour*60*60))/60);
    sec = delta_t-(double)(day*24*60*60)-(double)(hour*60*60)-(double)(min*60);
    if (day > 0 ) {
        printf("process time: %d day %d hr %d min %.1f sec \n",day,hour,min,sec);
    }else if (hour>0){
        printf("process time: %d hr %d min %.1f sec \n",hour,min,sec);
    }else if (min>0){
        printf("process time: %d min %.1f sec \n",min,sec);
    }else{
        printf("process time: %.1f sec \n",sec);
    }
    return 0;
}


