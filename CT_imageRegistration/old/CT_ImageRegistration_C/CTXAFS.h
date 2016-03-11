//
//  CTXAFS.h
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/01/09.
//  Copyright (c) 2014-2015 Nozomu Ishiguro. All rights reserved.
//

#ifndef CT_XANES_analysis_CTXAFS_hpp
#define CT_XANES_analysis_CTXAFS_hpp



#if defined (__APPLE__)  //Mac OS X, iOS?
    #include <OpenCL/OpenCL.h>
    #include <dirent.h>
    #define WINDOWS 0

#elif defined (_M_X64)  //Windows 64 bit
    #define _CRT_SECURE_NO_WARNINGS
    #include <windows.h>
    #include <CL/cl.h>
    #include <direct.h>
    #include <dirent.h>
    #include <stdbool.h>
    #define mkdir _mkdir
    #define WINDOWS 1

#elif defined (_WIN32)  //Windows 32 bit
    #define _CRT_SECURE_NO_WARNINGS
    #include <windows.h>
    #include <CL/cl.h>
    #include <direct.h>
    #include <dirent.h>
    #include <stdbool.h>
    #define mkdir _mkdir
    #define WINDOWS 1

#elif defined (__linux__)   //Linux
    #include <CL/cl.h>
    #include <dirent.h>
    #define WINDOWS 0

#endif


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>




#define MAX_SOURCE_SIZE (0x100000)




extern int IMAGE_SIZE_X;
extern int IMAGE_SIZE_Y;

extern int imageRegistlation_ocl();
extern int OCL_device_list();
extern int imagingXANES_fitting();
extern int readHisFile();




#endif
