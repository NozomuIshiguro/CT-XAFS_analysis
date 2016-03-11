//
//  imageRegistration_ocl.c
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2014/12/24.
//  Copyright (c) 2014-2015 Nozomu Ishiguro. All rights reserved.
//


#include "CTXAFS.h"
#include "imageregistration_kernel_src_cl.h"

int OCL_device_list(int *plat_id_num,int *dev_id_num){
    cl_int ret;
    
    struct OCL_platform_device{
        int platform_num;
        int device_num;
    }ocl_plat_dev[50];
    
    cl_platform_id platform_ids[5];
    cl_uint ret_num_platforms;
    clGetPlatformIDs(5, platform_ids, &ret_num_platforms);
    
    puts("Select OpenCL device...");
    
    int platdevice_num = 0;
    for (unsigned int i=0; i<ret_num_platforms; i++) {
        char platform_pram[128];
        ret = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, 128, &platform_pram, NULL);
        printf("    OpenCL platform: %s\n",platform_pram);
        cl_device_id device_ids[10];
        cl_uint ret_num_devices;
        clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 10, device_ids, &ret_num_devices);
        char device_pram[128];
        for (unsigned int j=0; j<ret_num_devices; j++) {
            ret = clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME, 128, &device_pram, NULL);
            printf("        %d: %s\n",platdevice_num, device_pram);
            
            ocl_plat_dev[platdevice_num].platform_num = i;
            ocl_plat_dev[platdevice_num].device_num  =j;
            platdevice_num++;
        }
    }
     
    int select_devNum;
    scanf("%d", &select_devNum);
    *plat_id_num = ocl_plat_dev[select_devNum].platform_num;
    *dev_id_num = ocl_plat_dev[select_devNum].device_num;
    
    return 0;
}

int imageRegistlation_ocl(char *fileName_base, char *output_dir,
                          int startEnergyNo, int endEnergyNo, int targetEnergyNo,
                          int startAngleNo, int endAngleNo,int rotation,
                          int *platform_id_num, int *device_id_num)
{
    cl_int ret;
    
    
    /*OpenCL　Plattform check*/
    cl_platform_id platform_id[5];
    cl_uint ret_num_platforms;
    clGetPlatformIDs(1, platform_id, &ret_num_platforms);
    
    int plat_id_num;
    if (platform_id_num == NULL) {
        plat_id_num=0;
    }else{
        plat_id_num=*platform_id_num;
    }
    char CL_version[64];
    size_t r_size_version;
    clGetPlatformInfo(platform_id[plat_id_num], CL_PLATFORM_VERSION, sizeof(CL_version), &CL_version, &r_size_version);
    bool OCL_12 = (strcmp(CL_version, "OpenCL 1.2")>=0);
    printf("%s\n",CL_version);
    
    
    
    /*OpenCL　Device check*/
    cl_device_id device_id[10];
    cl_uint ret_num_devices;
    int dev_id_num;
    if (device_id_num==NULL) {
        clGetDeviceIDs(platform_id[plat_id_num], CL_DEVICE_TYPE_GPU, 10, device_id, &ret_num_devices);
        dev_id_num = 0;
    } else{
        clGetDeviceIDs(platform_id[plat_id_num], CL_DEVICE_TYPE_ALL, 10, device_id, &ret_num_devices);
        dev_id_num = *device_id_num;
    }
    
    //printf("p,d:,%d,%d",plat_id_num,dev_id_num);
    
    char device_pram[128];
    size_t device_pram_size[3]={0,0,0};
    ret = clGetDeviceInfo(device_id[dev_id_num], CL_DEVICE_NAME, 128, &device_pram, NULL);
    printf("CL DEVICE NAME: %s\n", device_pram);
    ret = clGetDeviceInfo(device_id[dev_id_num], CL_DEVICE_IMAGE2D_MAX_HEIGHT, 512, &device_pram_size, NULL);
    printf("CL DEVICE MAX 2D IMAGE HEIGHT: %d\n",(int)device_pram_size[0]);
    ret = clGetDeviceInfo(device_id[dev_id_num], CL_DEVICE_IMAGE2D_MAX_WIDTH, 512, &device_pram_size, NULL);
    printf("CL DEVICE MAX 2D IMAGE WIDTH: %d\n",(int)device_pram_size[0]);
    ret = clGetDeviceInfo(device_id[dev_id_num], CL_DEVICE_MAX_WORK_GROUP_SIZE, 512, &device_pram_size, NULL);
    printf("CL DEVICE MAX WORK GROUP SIZE: %d\n",(int)device_pram_size[0]);
    ret = clGetDeviceInfo(device_id[dev_id_num], CL_DEVICE_MAX_WORK_ITEM_SIZES, 512, &device_pram_size, NULL);
    printf("CL DEVICE MAX WORK ITEM SIZE: %d,%d,%d\n",(int)device_pram_size[0],(int)device_pram_size[1],(int)device_pram_size[2]);
    ret = clGetDeviceInfo(device_id[dev_id_num], CL_DEVICE_MAX_MEM_ALLOC_SIZE, 512, &device_pram_size, NULL);
    printf("CL DEVICE CL_DEVICE_MAX_MEM_ALLOC_SIZE: %d\n",(unsigned int)device_pram_size[0]);
    ret = clGetDeviceInfo(device_id[dev_id_num], CL_DEVICE_GLOBAL_MEM_SIZE, 512, &device_pram_size, NULL);
    printf("CL DEVICE CL_DEVICE_GLOBAL_MEM_SIZE: %d\n",(unsigned int)device_pram_size[0]);
    ret = clGetDeviceInfo(device_id[dev_id_num], CL_DEVICE_LOCAL_MEM_SIZE, 512, &device_pram_size, NULL);
    printf("CL DEVICE CL_DEVICE_LOCAL_MEM_SIZE: %d\n\n",(unsigned int)device_pram_size[0]);
    
    
    /*OpenCL image support check*/
    cl_bool support;
    size_t r_size;
    //for (int i=0; i<ret_num_devices; i++) {
        clGetDeviceInfo(device_id[dev_id_num], CL_DEVICE_IMAGE_SUPPORT, sizeof(support), &support, &r_size);
        if (support!=CL_TRUE) {
            printf("Device does not support image ");
            return 1;
        }
    //}
    
    
    
    /*OpenCL Context create*/
    cl_context context = NULL;
    context = clCreateContext(NULL, 1, &device_id[dev_id_num], NULL, NULL, &ret);
    
    
    
    
    /*OpenCL program object build*/
    cl_program program = NULL;
	
    char *kernel_src_str0;
    FILE *fp_kernel;
    fp_kernel = fopen("./imageregistration_kernel_src.cl", "r");
	if (!fp_kernel){
		fprintf(stderr, "Fail to load kernel source\n");
		exit(1);
	}
    kernel_src_str0 = (char*)malloc(MAX_SOURCE_SIZE);
    size_t kernel_code_size = fread(kernel_src_str0, 1, MAX_SOURCE_SIZE, fp_kernel);
	fclose(fp_kernel);
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src_str0, (const size_t *)&kernel_code_size, &ret);
	free(kernel_src_str0);
    
    /*const size_t kernel_code_size = strlen(kernel_src);
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_src, (const size_t *)&kernel_code_size, &ret);*/
    
    
    /*kernel build*/
    cl_kernel kernel_shift[2] = {NULL,NULL};
    cl_kernel kernel_mt = NULL;
    cl_kernel kernel_sumImg = NULL;
    cl_kernel kernel_jacob = NULL;
    //cl_kernel kernel_calctJJ = NULL;
    cl_kernel kernel_reduct1 = NULL;
    cl_kernel kernel_reduct2 = NULL;
    cl_kernel kernel_renew = NULL;
    cl_kernel kernel_SumImage_RenewLambda = NULL;
    cl_kernel kernel_convertImg2buff= NULL;
    clBuildProgram(program, 1, &device_id[dev_id_num], "", NULL, NULL);
    kernel_shift[0] = clCreateKernel(program, "imageconvshift1", &ret);
    kernel_shift[1] = clCreateKernel(program, "imageconvshift2", &ret);
    kernel_mt = clCreateKernel(program, "convert2mt_img", &ret);
    kernel_sumImg = clCreateKernel(program, "Sum_Image", &ret);
    kernel_jacob = clCreateKernel(program, "makeJacobian", &ret);
    //kernel_calctJJ = clCreateKernel(program, "calc_tJJ_tJDeltaImg", &ret);
    kernel_reduct1 = clCreateKernel(program, "reduction_tJJ_tJDeltaImg1", &ret);
    kernel_reduct2 = clCreateKernel(program, "reduction_tJJ_tJDeltaImg2", &ret);
    kernel_renew = clCreateKernel(program, "renew_taranspara", &ret);
    kernel_SumImage_RenewLambda = clCreateKernel(program, "SumImage_RenewLambda", &ret);
    kernel_convertImg2buff=clCreateKernel(program, "convertImage2Buffer", &ret);
    
    
    
    
    
    /*Open CL command que create*/
    cl_command_queue command_queue = NULL;
    command_queue = clCreateCommandQueue(context, device_id[dev_id_num], 0, &ret);
    
    
    /*Image object create*/
    cl_mem image_target, image_sample;
    cl_mem image_target_out,image_sample_out;
    /*image object data format defenition*/
    cl_image_format fmt;
    fmt.image_channel_order = CL_RA;
    fmt.image_channel_data_type = CL_FLOAT;
    
    
    if (OCL_12) {
        //puts("OpenCL 1.2\n");
        
        #ifdef CL_VERSION_1_2
        /*image object data type defenition*/
        cl_image_desc desc;
        desc.image_width = IMAGE_SIZE_X;
        desc.image_height = IMAGE_SIZE_Y;
        desc.image_depth = 1;
        desc.image_row_pitch = 0;
        desc.image_slice_pitch = 0;
        desc.buffer = NULL;
        desc.image_type =CL_MEM_OBJECT_IMAGE2D;
        desc.num_mip_levels =0;
        desc.num_samples =0;
        desc.image_array_size=1;
        
        image_target = clCreateImage(context, CL_MEM_READ_WRITE, &fmt, &desc, NULL, NULL);//OPENCL 1.2
        image_sample = clCreateImage(context, CL_MEM_READ_WRITE, &fmt, &desc, NULL, NULL);//OPENCL 1.2
        image_target_out = clCreateImage(context, CL_MEM_READ_WRITE, &fmt, &desc, NULL, NULL);//OPENCL 1.2
        image_sample_out = clCreateImage(context, CL_MEM_READ_WRITE, &fmt, &desc, NULL, NULL);//OPENCL 1.2
        #endif
        
        
    }else{
        //puts("OpenCL 1.1\n");
        
        
        image_target = clCreateImage2D(context, CL_MEM_READ_WRITE, &fmt, IMAGE_SIZE_X, IMAGE_SIZE_Y, 0, NULL, NULL);//OPENCL 1.1
        image_sample = clCreateImage2D(context, CL_MEM_READ_WRITE, &fmt, IMAGE_SIZE_X, IMAGE_SIZE_Y, 0, NULL, NULL);
        image_target_out = clCreateImage2D(context, CL_MEM_READ_WRITE, &fmt, IMAGE_SIZE_X, IMAGE_SIZE_Y, 0, NULL, NULL);
        image_sample_out = clCreateImage2D(context, CL_MEM_READ_WRITE, &fmt, IMAGE_SIZE_X, IMAGE_SIZE_Y, 0, NULL, NULL);//OPENCL 1.1
        
        
    }
    
    
    /*OCL event object create*/
    cl_event ev_dark;
    cl_event ev_transpara[2];
    
    
    
    /*resistration shift parameter buffer create / OCL transfer*/
    const float transpara_target_data[3]={0,0,0};
    float transpara_data[3]={0.0,0.0,0.0};
    cl_mem transpara_target, transpara;
    transpara_target = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*3, NULL, &ret);
    ret = clEnqueueWriteBuffer(command_queue, transpara_target, CL_TRUE, 0, sizeof(cl_float)*3, transpara_target_data, 0, NULL, &ev_transpara[0]);
    //printf("transpara target ini: %d\n",ret);
    transpara = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*3, NULL, &ret);
    ret = clEnqueueWriteBuffer(command_queue, transpara, CL_TRUE, 0, sizeof(cl_float)*3, transpara_data, 0, NULL, &ev_transpara[1]);
   // printf("transpara ini: %d\n\n",ret);
    
    
    /*dark buffer memory reservation*/
    cl_mem  dark_img_cl;
    dark_img_cl = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_X*IMAGE_SIZE_Y, NULL, &ret);
   
    
    
    
    /* dark data input / OCL transfer*/
    printf("Processing dark...\n");
    float *dark_img;
    dark_img = (float *) malloc(IMAGE_SIZE_X*IMAGE_SIZE_Y*sizeof(float));
    char fileName_dark[128];
    strcpy(fileName_dark, fileName_base);
    strcat(fileName_dark,"dark.his");
    FILE *fp_dark;
    fp_dark = fopen(fileName_dark, "rb");
    if (!fp_dark) {
        fprintf(stderr, "   Failed to load his \n\n");
        exit(1);
    }
    readHisFile(fp_dark,1,30,dark_img);
    fclose(fp_dark);
    ret = clEnqueueWriteBuffer(command_queue, dark_img_cl, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_X*IMAGE_SIZE_Y, dark_img, 0, NULL, &ev_dark);
    //printf("    dark input: %d\n", ret);
    free(dark_img);
    
    
    
    /* I0 data input */
    printf("Processing I0...\n\n");
	float *I0_img, *I0_img_target;
	I0_img = (float *)malloc(IMAGE_SIZE_X*IMAGE_SIZE_Y*sizeof(float)*(endEnergyNo - startEnergyNo + 1));
	I0_img_target = (float *)malloc(IMAGE_SIZE_X*IMAGE_SIZE_Y*sizeof(float));
	for (int i = startEnergyNo; i <= endEnergyNo; i++) {
        /*sample numTag settings*/
        char EnumTag[10];
        if (i<10) {
            sprintf(EnumTag, "00%d",i);
        } else if(i<100){
            sprintf(EnumTag, "0%d",i);
        } else {
            sprintf(EnumTag, "%d",i);
        }
        
        char fileName_I0[128];
        strcpy(fileName_I0, fileName_base);
        strcat(fileName_I0, EnumTag);
        strcat(fileName_I0, "_I0.his");
        FILE *fp_I0;
        fp_I0 = fopen(fileName_I0, "rb");
        if (!fp_I0) {
            break;
        }
        readHisFile(fp_I0,1,20,(I0_img+IMAGE_SIZE_X*IMAGE_SIZE_Y*(i-startEnergyNo)));
        fclose(fp_I0);
    }
	/*target numTag settings*/
	char EnumTag_target[10];
	if (targetEnergyNo<10) {
		sprintf(EnumTag_target, "00%d", targetEnergyNo);
	}
	else if (targetEnergyNo<100){
		sprintf(EnumTag_target, "0%d", targetEnergyNo);
	}
	else {
		sprintf(EnumTag_target, "%d", targetEnergyNo);
	}

	char fileName_I0_target[128];
	strcpy(fileName_I0_target, fileName_base);
	strcat(fileName_I0_target, EnumTag_target);
	strcat(fileName_I0_target, "_I0.his");
	FILE *fp_I0_target;
	fp_I0_target = fopen(fileName_I0_target, "rb");
	readHisFile(fp_I0_target, 1, 20, I0_img_target);
	fclose(fp_I0_target);
    
    
    /*kernel dimension declaration*/
    cl_uint work_dim =2;
    size_t global_item_size[3] = {IMAGE_SIZE_X,IMAGE_SIZE_Y,1};
    size_t local_item_size[3] = {1,1,1};
    
    cl_event ev_loopstart[3];
    ev_loopstart[0]=ev_dark;
    ev_loopstart[1]=ev_transpara[0];
    ev_loopstart[2]=ev_transpara[1];
    clWaitForEvents(3, ev_loopstart);
    
    //printf("end angle: %d\n",endAngleNo);
    for (int j=startAngleNo; j<=endAngleNo; j++) {
        printf("angle: %d\n",j);
        //printf("initial transpara: %.1f,%.1f,%.1f\n",transpara_data[0],transpara_data[1],transpara_data[2]);
        
        /*image object*/
        cl_event ev_I0It_target[2];
        cl_event ev_mt_target;
        
        /*angle tag settings*/
        char angleNumTag[15];
        if (j<10) {
            sprintf(angleNumTag, "/mtr000%d.raw",j);
        } else if(j<100){
            sprintf(angleNumTag, "/mtr00%d.raw",j);
        } else if(j<1000){
            sprintf(angleNumTag, "/mtr0%d.raw",j);
        } else {
            sprintf(angleNumTag, "/mtr%d.raw",j);
        }
        
        
        /*target It data input*/
        float *It_img;
        It_img = (float *) malloc(IMAGE_SIZE_X*IMAGE_SIZE_Y*sizeof(float));
        char fileName_It[128];
        strcpy(fileName_It, fileName_base);
        strcat(fileName_It, EnumTag_target);
        strcat(fileName_It, ".his");
        FILE *fp_It;
        fp_It = fopen(fileName_It, "rb");
        if (!fp_It) {
            break;
        }
        readHisFile(fp_It,j,j,It_img);
        fclose(fp_It);
        
        cl_mem I0_img_cl, It_img_cl;
        I0_img_cl = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_X*IMAGE_SIZE_Y, NULL, &ret);
        It_img_cl = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_X*IMAGE_SIZE_Y, NULL, &ret);
        
        
        /*target mt transform / OCL tranfer*/
        ret = clEnqueueWriteBuffer(command_queue, I0_img_cl, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_X*IMAGE_SIZE_Y, I0_img_target, 0, NULL, &ev_I0It_target[0]);
        //printf("    target I0 input: %d\n",ret);
        ret = clEnqueueWriteBuffer(command_queue, It_img_cl, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_X*IMAGE_SIZE_Y, It_img, 0, NULL, &ev_I0It_target[1]);
        //printf("    target It input: %d\n",ret);
        
        
        
        /*target mt transform kernel parameter settings*/
        clSetKernelArg(kernel_mt, 0, sizeof(cl_mem), (void*)&dark_img_cl);
        clSetKernelArg(kernel_mt, 1, sizeof(cl_mem), (void*)&I0_img_cl);
        clSetKernelArg(kernel_mt, 2, sizeof(cl_mem), (void*)&It_img_cl);
        clSetKernelArg(kernel_mt, 3, sizeof(cl_mem), (void*)&image_target);
        /*target mt transform kernel dimension settings*/
        work_dim =2;
        global_item_size[0] = IMAGE_SIZE_X;
        global_item_size[1] = IMAGE_SIZE_Y;
        local_item_size[0] = 1;
        local_item_size[1] = 1;
        /*target mt transform execution*/
        ret = clEnqueueNDRangeKernel(command_queue, kernel_mt, work_dim, NULL, global_item_size, local_item_size, 2, ev_I0It_target, &ev_mt_target);
        //printf("    target mt: %d\n",ret);
        free(It_img);
        
        
        for (int i=startEnergyNo; i<=endEnergyNo; i++) {
            printf("    energy no. %d\n",i);
            
            
            /* event object */
            cl_event ev_I0It_sample[2];
            cl_event ev_mt_sample;
            
            
            /* sample numTag settings*/
            char EnumTag[10];
            if (i<10) {
                sprintf(EnumTag, "00%d",i);
            } else if(i<100){
                sprintf(EnumTag, "0%d",i);
            } else {
                sprintf(EnumTag, "%d",i);
            }
            /* Sample It data input */
            float *It_img;
            It_img = (float *) malloc(IMAGE_SIZE_X*IMAGE_SIZE_Y*sizeof(float));
            char fileName_It[128];
            strcpy(fileName_It, fileName_base);
            strcat(fileName_It, EnumTag);
            strcat(fileName_It, ".his");
            FILE *fp_It;
            fp_It = fopen(fileName_It, "rb");
            if (!fp_It) {
                break;
            }
            readHisFile(fp_It,j,j,It_img);
            fclose(fp_It);
            

            /*sample mt transform OCL transfer*/
			ret = clEnqueueWriteBuffer(command_queue, I0_img_cl, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_X*IMAGE_SIZE_Y, I0_img + IMAGE_SIZE_X*IMAGE_SIZE_Y*(i-startEnergyNo), 1, &ev_mt_target, &ev_I0It_sample[0]);
            //printf("        sample I0 input: %d\n",ret);
            ret = clEnqueueWriteBuffer(command_queue, It_img_cl, CL_TRUE, 0, sizeof(cl_float)*IMAGE_SIZE_X*IMAGE_SIZE_Y, It_img, 1, &ev_mt_target, &ev_I0It_sample[1]);
            //printf("        sample It input: %d\n",ret);
            /*sample mt transform kernel parameter settings*/
            clSetKernelArg(kernel_mt, 0, sizeof(cl_mem), (void*)&dark_img_cl);
            clSetKernelArg(kernel_mt, 1, sizeof(cl_mem), (void*)&I0_img_cl);
            clSetKernelArg(kernel_mt, 2, sizeof(cl_mem), (void*)&It_img_cl);
            clSetKernelArg(kernel_mt, 3, sizeof(cl_mem), (void*)&image_sample);
            /**sample mt transform kernel dimension settings*/
            work_dim =2;
            global_item_size[0] = IMAGE_SIZE_X;
            global_item_size[1] = IMAGE_SIZE_Y;
            local_item_size[0] = 1;
            local_item_size[1] = 1;
            /*sample mt transform execution*/
            ret = clEnqueueNDRangeKernel(command_queue, kernel_mt, work_dim, NULL, global_item_size, local_item_size, 2, ev_I0It_sample, &ev_mt_sample);
            printf("        sample mt: %d\n", ret);
            free(It_img);
            
            /* Levenberg-Marquarrdt　damping parameter */
            cl_mem lambda;
            lambda = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &ret);
            float *lambda_data;
            lambda_data = (float *) malloc(sizeof(float));
            lambda_data[0] = -1;
            ret = clEnqueueWriteBuffer(command_queue, lambda, CL_TRUE, 0, sizeof(float), lambda_data, 1, &ev_mt_sample, NULL);
            
            /* registration loop*/
            int merge[2]={8,8};
            for (int k=4; k>=1; k--) {  
                printf("        registlation x%d\n",k);
                
                int imagesize = IMAGE_SIZE_X*IMAGE_SIZE_Y/merge[0]/merge[1];
                int imagesizeX = IMAGE_SIZE_X/merge[0];
                int imagesizeY = IMAGE_SIZE_Y/merge[1];
                int mergesize = merge[0];
                
                
                
                /*resistration Jacobian matrix, OCL object create*/
                cl_mem Fx_old, Fx_new, jacobian, del_image, dummy, delta_rho;
                Fx_old = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &ret);
                Fx_new = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &ret);
                jacobian = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imagesize*3, NULL, &ret);
                del_image = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imagesize, NULL, &ret);
                dummy = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*imagesize*9, NULL, &ret);
                delta_rho = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float), NULL, &ret);
                
                
                
                for (int t=0; t<3; t++) {
                    printf("            trial no. %d\n",t);
                    
                    /*event object*/
                    cl_event ev_convShift[2];
                    cl_event ev_jacob;
                    //cl_event  ev_calctJJ;
                    cl_event ev_reduction1,ev_reduction2, ev_renew;
                    cl_event ev_sample_convshift;
                    
                    
                    /*target convlution/shift kernel parameter settings*/
                    clSetKernelArg(kernel_shift[0], 0, sizeof(cl_mem), (void*)&image_target);
                    clSetKernelArg(kernel_shift[0], 1, sizeof(cl_mem), (void*)&image_target_out);
                    clSetKernelArg(kernel_shift[0], 2, sizeof(cl_mem), (void*)&transpara_target);
                    /* sample convlution/shift カkernel parameter settings*/
                    clSetKernelArg(kernel_shift[1], 0, sizeof(cl_mem), (void*)&image_sample);
                    clSetKernelArg(kernel_shift[1], 1, sizeof(cl_mem), (void*)&image_sample_out);
                    clSetKernelArg(kernel_shift[1], 2, sizeof(cl_mem), (void*)&transpara);
                    /* sample convlution/shift image integration kernel parameter settings (t=0)*/
                    clSetKernelArg(kernel_sumImg, 0, sizeof(cl_mem), (void*)&image_sample_out);
                    clSetKernelArg(kernel_sumImg, 1, sizeof(cl_mem), (void*)&Fx_new);
                    clSetKernelArg(kernel_sumImg, 2, sizeof(int), (void*)&imagesizeX);
                    clSetKernelArg(kernel_sumImg, 3, sizeof(int), (void*)&imagesizeY);
                    /* sample convlution/shift image integration kernel parameter settings
                     and Lambda update kernel parameter settings (t>0)*/
                    clSetKernelArg(kernel_SumImage_RenewLambda, 0, sizeof(cl_mem), (void*)&image_sample_out);
                    clSetKernelArg(kernel_SumImage_RenewLambda, 1, sizeof(int), (void*)&imagesizeX);
                    clSetKernelArg(kernel_SumImage_RenewLambda, 2, sizeof(int), (void*)&imagesizeY);
                    clSetKernelArg(kernel_SumImage_RenewLambda, 3, sizeof(cl_mem), (void*)&Fx_old);
                    clSetKernelArg(kernel_SumImage_RenewLambda, 4, sizeof(cl_mem), (void*)&Fx_new);
                    clSetKernelArg(kernel_SumImage_RenewLambda, 5, sizeof(cl_mem), (void*)&lambda);
                    clSetKernelArg(kernel_SumImage_RenewLambda, 6, sizeof(cl_mem), (void*)&delta_rho);
                    /* resistration Jacobian inverse matrix kernel parameter settings*/
                    clSetKernelArg(kernel_jacob, 0, sizeof(cl_mem), (void*)&image_target_out);
                    clSetKernelArg(kernel_jacob, 1, sizeof(cl_mem), (void*)&image_sample_out);
                    clSetKernelArg(kernel_jacob, 2, sizeof(cl_mem), (void*)&transpara);
                    clSetKernelArg(kernel_jacob, 3, sizeof(cl_mem), (void*)&jacobian);
                    clSetKernelArg(kernel_jacob, 4, sizeof(cl_mem), (void*)&del_image);
                    clSetKernelArg(kernel_jacob, 5, sizeof(cl_mem), (void*)&dummy);
                    clSetKernelArg(kernel_jacob, 6, sizeof(int), (void*)&rotation);
                    /* tJJ reduction1*/
                    clSetKernelArg(kernel_reduct1, 0, sizeof(cl_mem), (void*)&dummy);
                    clSetKernelArg(kernel_reduct1, 1, sizeof(int), (void*)&imagesizeY);
                    /* tJJ reduction2*/
                    clSetKernelArg(kernel_reduct2, 0, sizeof(cl_mem), (void*)&dummy);
                    clSetKernelArg(kernel_reduct2, 1, sizeof(int), (void*)&imagesizeX);
                    /* transpara update kernel parameter settings*/
                    clSetKernelArg(kernel_renew, 0, sizeof(cl_mem), (void*)&transpara);
                    clSetKernelArg(kernel_renew, 1, sizeof(cl_mem), (void*)&dummy);
                    clSetKernelArg(kernel_renew, 2, sizeof(cl_mem), (void*)&lambda);
                    clSetKernelArg(kernel_renew, 3, sizeof(cl_mem), (void*)&delta_rho);
                    clSetKernelArg(kernel_renew, 4, sizeof(int), (void*)&imagesize);
                    clSetKernelArg(kernel_renew, 5, sizeof(int), (void*)&mergesize);
                    
                    
                    
                    /* convlution/shift kernel dimension settings */
                    work_dim =2;
                    global_item_size[0] = IMAGE_SIZE_X;
                    global_item_size[1] = IMAGE_SIZE_Y;
                    local_item_size[0] = merge[0];
                    local_item_size[1] = merge[1];
                    
                    
                    /* target convlution/shift execution*/
                    ret = clEnqueueNDRangeKernel(command_queue, kernel_shift[0], work_dim, NULL, global_item_size, local_item_size, 1, &ev_mt_sample, &ev_convShift[0]);
                   // printf("            target conv shift: %d\n", ret);
                    /* sample convlution/shift execution*/
                    ret = clEnqueueNDRangeKernel(command_queue, kernel_shift[1], work_dim, NULL, global_item_size, local_item_size, 1, &ev_mt_sample, &ev_sample_convshift);
                   // printf("            sample conv shift: %d\n", ret);
                    
                    /*Fx update execution*/
                    if (t>0) {
                        cl_event ev_copyFx;
                        clEnqueueCopyBuffer(command_queue, Fx_new, Fx_old, 0, 0, sizeof(float), 1, &ev_sample_convshift, &ev_copyFx);
                        ret =  clEnqueueTask(command_queue, kernel_SumImage_RenewLambda, 1, &ev_copyFx, &ev_convShift[1]);
                       // printf("            Renew Lambda: %d\n", ret);
                        
                        clReleaseEvent(ev_copyFx);
                    }else{
                        ret =  clEnqueueTask(command_queue, kernel_sumImg, 1, &ev_sample_convshift, &ev_convShift[1]);
                       // printf("            Sum Image: %d\n", ret);
                        
                    }
                    
                    ret = clEnqueueReadBuffer(command_queue, lambda, CL_TRUE, 0, sizeof(float), lambda_data, 1, &ev_convShift[1], NULL);
                    //printf("                labmda: %f\n",lambda_data[0]);
                    
                    
                    /*resistration Jacobian create / tJJ matrix processing kernerl dimension settings*/
                    work_dim =2;
                    global_item_size[0] = imagesizeX;
                    global_item_size[1] = imagesizeY;
                    local_item_size[0] = 1;
                    local_item_size[1] = 1;
                    
                    
                    /* resistration Jacobian create execution*/
                    ret = clEnqueueNDRangeKernel(command_queue, kernel_jacob, work_dim, NULL, global_item_size, local_item_size, 2, ev_convShift, &ev_jacob);
                    //printf("            Jacobian: %d\n", ret);
                   
                    
                    /* tJJ reduction1 execution*/
                    work_dim =1;
                    global_item_size[0] = imagesizeX;
                    global_item_size[1] = 1;
                    if(imagesizeX<512){
                        local_item_size[0] = imagesizeX;
                    }else{
                        local_item_size[0] = 512;
                    }
                    local_item_size[1] = 1;
                    ret = clEnqueueNDRangeKernel(command_queue, kernel_reduct1, work_dim, NULL, global_item_size, local_item_size, 1, &ev_jacob,&ev_reduction1);
                    //printf("            tJJ reduction1: %d\n", ret);
                    /* tJJ reduction2 execution*/
                    work_dim =1;
                    global_item_size[0] = imagesizeY;
                    global_item_size[1] = 1;
                    if(imagesizeY<512){
                        local_item_size[0] = imagesizeY;
                    }else{
                        local_item_size[0] = 512;
                    }
                    local_item_size[1] = 1;
                    ret = clEnqueueNDRangeKernel(command_queue, kernel_reduct2, work_dim, NULL, global_item_size, local_item_size, 1,&ev_reduction1,&ev_reduction2);
                    //printf("            tJJ reduction2: %d\n", ret);
                    
                   
                    
                    
                    /* transpara update execution*/
                    float transpara_old[3]={transpara_data[0],transpara_data[1],transpara_data[2]};
                    ret =  clEnqueueTask(command_queue, kernel_renew, 1, &ev_reduction2, &ev_renew);
                    //printf("            renew transpara: %d\n", ret);
                    if(lambda_data[0]==-1) lambda_data[0]=1;
                    ret = clEnqueueReadBuffer(command_queue,transpara, CL_TRUE, 0, sizeof(cl_float)*3, transpara_data, 1, &ev_renew, NULL);
                    //printf("            renew transpara: %d\n", ret);
                    printf("                renew transpara: %.1f,%.1f,%.1f\n",transpara_data[0],transpara_data[1],transpara_data[2]);
                    float delta_transpara[3];
                    delta_transpara[0]=transpara_data[0]-transpara_old[0];
                    delta_transpara[1]=transpara_data[1]-transpara_old[1];
                    delta_transpara[2]=transpara_data[2]-transpara_old[2];
                    float deltasize = delta_transpara[1]*delta_transpara[1]+delta_transpara[2]*delta_transpara[2];
                    if (deltasize<1) break;
                    else if(rotation==1&&delta_transpara[0]<0.001) break;
                    
                   
                    clReleaseEvent(ev_jacob);
                    clReleaseEvent(ev_reduction1);
                    clReleaseEvent(ev_reduction2);
                    clReleaseEvent(ev_renew);
                    clReleaseEvent(ev_convShift[0]);
                    clReleaseEvent(ev_convShift[1]);
                    clReleaseEvent(ev_sample_convshift);
                    
                    
                }
                
                
                
                clReleaseMemObject(jacobian);
                clReleaseMemObject(del_image);
                clReleaseMemObject(dummy);
                clReleaseMemObject(Fx_old);
                clReleaseMemObject(Fx_new);
                clReleaseMemObject(delta_rho);
                merge[0]/=2;
                merge[1]/=2;
            }
            
            
            /*registration image tranfer*/
            /*image object data transform settings*/

            cl_mem image_sample_out_buffer;
            image_sample_out_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_X*IMAGE_SIZE_Y, NULL, &ret);
            float *mt_output;
            mt_output = (float *) malloc(sizeof(float)*IMAGE_SIZE_X*IMAGE_SIZE_Y);
            //size_t imagewidth;
            //clGetImageInfo(image_sample_out, CL_IMAGE_ROW_PITCH, 128, &imagewidth, NULL);
            //printf("image width: %d\n",(int)imagewidth);
            
            
            /* resistration image output settings and execution*/
            cl_event ev_convImg2Buff;
            work_dim =2;
            global_item_size[0] = IMAGE_SIZE_X;
            global_item_size[1] = IMAGE_SIZE_Y;
            local_item_size[0] = 1;
            local_item_size[1] = 1;
            clSetKernelArg(kernel_convertImg2buff, 0, sizeof(cl_mem), (void*)&image_sample_out);
            clSetKernelArg(kernel_convertImg2buff, 1, sizeof(cl_mem), (void*)&image_sample_out_buffer);
            ret = clEnqueueNDRangeKernel(command_queue, kernel_convertImg2buff, work_dim, NULL, global_item_size, local_item_size, 0, NULL, &ev_convImg2Buff);
            //printf("    convert image to buffer: %d\n", ret);
            ret = clEnqueueReadBuffer(command_queue, image_sample_out_buffer, CL_TRUE, 0, sizeof(float)*IMAGE_SIZE_X*IMAGE_SIZE_Y, mt_output, 1, &ev_convImg2Buff, NULL);
            //printf("    Read OCL buffer: %d\n",ret);
            
            /*file output*/
            char fileName_output[128];
            strcpy(fileName_output, output_dir);
            strcat(fileName_output, "/");
            strcat(fileName_output, EnumTag);
            #if defined (WIN32) || defined (_M_X64)
                mkdir(fileName_output);
            
            #else
                mkdir(fileName_output, 0755);
            #endif
            strcat(fileName_output, angleNumTag);
            FILE *fp_out;
            printf("    output file: %s\n\n",fileName_output);
            fp_out = fopen(fileName_output, "wb");
            fwrite(mt_output, sizeof(float), IMAGE_SIZE_X*IMAGE_SIZE_Y, fp_out);
            fclose(fp_out);
            free(mt_output);
            
            
            
            clReleaseMemObject(lambda);
            clReleaseMemObject(image_sample_out_buffer);
            
            
            clReleaseEvent(ev_I0It_sample[0]);
            clReleaseEvent(ev_I0It_sample[1]);
            clReleaseEvent(ev_mt_sample);
            
            
        }
        
        
        
        clReleaseEvent(ev_I0It_target[0]);
        clReleaseEvent(ev_I0It_target[1]);
        clReleaseEvent(ev_mt_target);
        
        clReleaseMemObject(I0_img_cl);
        clReleaseMemObject(It_img_cl);
        
    }
    
    clReleaseEvent(ev_loopstart[0]);
    clReleaseEvent(ev_loopstart[1]);
    clReleaseEvent(ev_loopstart[2]);
    //clReleaseEvent(ev_dark);
    //clReleaseEvent(ev_transpara[0]);
    //clReleaseEvent(ev_transpara[1]);
    
    
    
    
    free(I0_img);
    
    
    
    clReleaseMemObject(image_target);
    clReleaseMemObject(image_sample);
    clReleaseMemObject(image_target_out);
    clReleaseMemObject(image_sample_out);
    clReleaseMemObject(transpara);
    clReleaseMemObject(transpara_target);
    clReleaseMemObject(dark_img_cl);
    
    

    clReleaseKernel(kernel_shift[0]);
    clReleaseKernel(kernel_shift[1]);
    clReleaseKernel(kernel_mt);
    clReleaseKernel(kernel_jacob);
    //clReleaseKernel(kernel_calctJJ);
    clReleaseKernel(kernel_renew);
    clReleaseKernel(kernel_convertImg2buff);
    clReleaseKernel(kernel_sumImg);
    clReleaseKernel(kernel_reduct1);
    clReleaseKernel(kernel_reduct2);
    clReleaseKernel(kernel_SumImage_RenewLambda);
    
    
    clReleaseProgram(program);
    
    clReleaseCommandQueue(command_queue);
    
    //clReleaseDevice(device_id[dev_id_num]);
    
    clReleaseContext(context);
    
    return 0;
}