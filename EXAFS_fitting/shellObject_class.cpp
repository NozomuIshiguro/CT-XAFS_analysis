//
//  shellObject_class.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2017/07/30.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#include "EXAFS.hpp"

shellObjects::shellObjects(cl::CommandQueue commandQueue, cl::Program program, FEFF_shell shell,
                           int SizeX, int SizeY){
    imageSizeX = SizeX;
    imageSizeY = SizeY;
    imageSizeM = imageSizeX*imageSizeY;
    queue = commandQueue;
    context = queue.getInfo<CL_QUEUE_CONTEXT>();
    
    //raw image object from feffxxx.dat files
    int ksize = shell.getNumPnts();
    Reff = shell.getReff();
    cl::ImageFormat format(CL_R, CL_FLOAT);
    cl::Buffer k_raw(context,CL_MEM_READ_ONLY,sizeof(cl_float)*ksize,0,NULL);
    cl::Image1D real_2phc_raw(context,CL_MEM_READ_ONLY,format,ksize,0,NULL);
    cl::Image1D mag_raw(context,CL_MEM_READ_ONLY,format,ksize,0,NULL);
    cl::Image1D phase_raw(context,CL_MEM_READ_ONLY,format,ksize,0,NULL);
    cl::Image1D redFactor_raw(context,CL_MEM_READ_ONLY,format,ksize,0,NULL);
    cl::Image1D lambda_raw(context,CL_MEM_READ_ONLY,format,ksize,0,NULL);
    cl::Image1D real_p_raw(context,CL_MEM_READ_ONLY,format,ksize,0,NULL);
    
    real_2phc       = cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*MAX_KRSIZE,0,NULL);
    mag             = cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*MAX_KRSIZE,0,NULL);
    phase           = cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*MAX_KRSIZE,0,NULL);
    redFactor       = cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*MAX_KRSIZE,0,NULL);
    lambda          = cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*MAX_KRSIZE,0,NULL);
    real_p          = cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*MAX_KRSIZE,0,NULL);
    
    CN              = cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
    dR              = cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
    dE0             = cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
    ss              = cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
    E0imag          = cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
    C3              = cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
    C4              = cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
    
    
    kernel_feff = cl::Kernel(program,"redimension_feffShellPara");
    kernel_chiout = cl::Kernel(program,"outputchi");
    kernel_jacob_k = cl::Kernel(program,"Jacobian_k");
    kernel_CNw = cl::Kernel(program,"CNweighten");
    kernel_update = cl::Kernel(program,"updatePara");
    kernel_UR = cl::Kernel(program,"updateOrRestore");
    kernel_OBD = cl::Kernel(program,"outputBondDistance");
    kernel_contrain1 = cl::Kernel(program,"contrain_1");
    kernel_contrain2 = cl::Kernel(program,"contrain_2");
    
    cl::size_t<3> origin;
    cl::size_t<3> region;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    region[0] = ksize;
    region[1] = 1;
    region[2] = 1;
    queue.enqueueWriteBuffer(k_raw,CL_TRUE,0,sizeof(float)*ksize, &(shell.getk()[0]));
    queue.enqueueWriteImage(real_2phc_raw,CL_TRUE,origin,region,0,0,&(shell.getReal2phc()[0]));
    queue.enqueueWriteImage(mag_raw,CL_TRUE,origin,region,0,0,&(shell.getMag()[0]));
    queue.enqueueWriteImage(phase_raw,CL_TRUE,origin,region,0,0,&(shell.getPhase()[0]));
    queue.enqueueWriteImage(redFactor_raw,CL_TRUE,origin,region,0,0,&(shell.getRedFactor()[0]));
    queue.enqueueWriteImage(lambda_raw,CL_TRUE,origin,region,0,0,&(shell.getLambda()[0]));
    queue.enqueueWriteImage(real_p_raw,CL_TRUE,origin,region,0,0,&(shell.getReal_p()[0]));
    
    const cl::NDRange local_item_size(1,1,1);
    const cl::NDRange global_item_size(1,1,1);
    
    //real2_phc
    kernel_feff.setArg(0, real_2phc);
    kernel_feff.setArg(1, real_2phc_raw);
    kernel_feff.setArg(2, k_raw);
    kernel_feff.setArg(3, (cl_int)ksize);
    queue.enqueueNDRangeKernel(kernel_feff, NULL, global_item_size, local_item_size, NULL, NULL);
    
    //mag
    kernel_feff.setArg(0, mag);
    kernel_feff.setArg(1, mag_raw);
    queue.enqueueNDRangeKernel(kernel_feff, NULL, global_item_size, local_item_size, NULL, NULL);
    
    //phase
    kernel_feff.setArg(0, phase);
    kernel_feff.setArg(1, phase_raw);
    queue.enqueueNDRangeKernel(kernel_feff, NULL, global_item_size, local_item_size, NULL, NULL);
    
    //red factor
    kernel_feff.setArg(0, redFactor);
    kernel_feff.setArg(1, redFactor_raw);
    queue.enqueueNDRangeKernel(kernel_feff, NULL, global_item_size, local_item_size, NULL, NULL);
    
    //lambda
    kernel_feff.setArg(0, lambda);
    kernel_feff.setArg(1, lambda_raw);
    queue.enqueueNDRangeKernel(kernel_feff, NULL, global_item_size, local_item_size, NULL, NULL);
    
    //real_p
    kernel_feff.setArg(0, real_p);
    kernel_feff.setArg(1, real_p_raw);
    queue.enqueueNDRangeKernel(kernel_feff, NULL, global_item_size, local_item_size, NULL, NULL);
    
    
    //fill initial parameter
    //CN
    queue.enqueueFillBuffer(CN, (cl_float)shell.getDegen(), 0, sizeof(cl_float)*imageSizeM);
    //dR
    queue.enqueueFillBuffer(dR, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM);
    //dE0
    queue.enqueueFillBuffer(dE0, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM);
    //ss
    queue.enqueueFillBuffer(ss, (cl_float)3e-3f, 0, sizeof(cl_float)*imageSizeM);
    //E0imag
    queue.enqueueFillBuffer(E0imag, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM);
    //C3
    queue.enqueueFillBuffer(C3, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM);
    //C4
    queue.enqueueFillBuffer(C4, (cl_float)0.0f, 0, sizeof(cl_float)*imageSizeM);
    
    //freeFix para
    freeFixPara.push_back(true); //1: CN
    freeFixPara.push_back(true); //2: dR
    freeFixPara.push_back(true); //3: dE0
    freeFixPara.push_back(true); //4: ss
    freeFixPara.push_back(false);//5: E0img
    freeFixPara.push_back(false);//6: C3
    freeFixPara.push_back(false);//7: C4
}

int shellObjects::outputChiFit(cl::Buffer chi, cl::Buffer S02, int kw,
                               float kstart, float kend){
    
    int ksize = (int)ceil((min(float(kend+WIN_DK),(float)MAX_KQ)-max(float(kstart-WIN_DK),0.0f))/KGRID)+1;
    int koffset = (int)floor(max((float)(kstart-WIN_DK),0.0f)/KGRID);
    
    size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    const cl::NDRange local_item_size(min((int)maxWorkGroupSize,imageSizeX),1,1);
    const cl::NDRange global_item_size(imageSizeX,imageSizeY,ksize);
    const cl::NDRange global_item_offset(0,0,koffset);
    
    
    kernel_chiout.setArg(0, chi);
    kernel_chiout.setArg(1, (cl_float)Reff);
    kernel_chiout.setArg(2, S02);
    kernel_chiout.setArg(3, CN);
    kernel_chiout.setArg(4, dR);
    kernel_chiout.setArg(5, dE0);
    kernel_chiout.setArg(6, ss);
    kernel_chiout.setArg(7, E0imag);
    kernel_chiout.setArg(8, C3);
    kernel_chiout.setArg(9, C4);
    kernel_chiout.setArg(10, real_2phc);
    kernel_chiout.setArg(11, mag);
    kernel_chiout.setArg(12, phase);
    kernel_chiout.setArg(13, redFactor);
    kernel_chiout.setArg(14, lambda);
    kernel_chiout.setArg(15, real_p);
    kernel_chiout.setArg(16, (cl_int)kw);
    queue.enqueueNDRangeKernel(kernel_chiout, global_item_offset, global_item_size, local_item_size, NULL, NULL);
    
    
    return 0;
}

int shellObjects::outputJacobiank(cl::Buffer Jacob, cl::Buffer S02, int kw, int paramode,
                                  float kstart, float kend, bool useRealPart){
    
    int ksize = (int)ceil((min(float(kend+WIN_DK),(float)MAX_KQ)-max(float(kstart-WIN_DK),0.0f))/KGRID)+1;
    int koffset = (int)floor(max((float)(kstart-WIN_DK),0.0f)/KGRID);
                        
    size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    const cl::NDRange local_item_size(min((int)maxWorkGroupSize,imageSizeX),1,1);
    const cl::NDRange global_item_size(imageSizeX,imageSizeY,ksize);
    const cl::NDRange global_offset(0,0,koffset);
    
    kernel_jacob_k.setArg(0, Jacob);
    kernel_jacob_k.setArg(1, (cl_float)Reff);
    kernel_jacob_k.setArg(2, S02);
    kernel_jacob_k.setArg(3, CN);
    kernel_jacob_k.setArg(4, dR);
    kernel_jacob_k.setArg(5, dE0);
    kernel_jacob_k.setArg(6, ss);
    kernel_jacob_k.setArg(7, E0imag);
    kernel_jacob_k.setArg(8, C3);
    kernel_jacob_k.setArg(9, C4);
    kernel_jacob_k.setArg(10, real_2phc);
    kernel_jacob_k.setArg(11, mag);
    kernel_jacob_k.setArg(12, phase);
    kernel_jacob_k.setArg(13, redFactor);
    kernel_jacob_k.setArg(14, lambda);
    kernel_jacob_k.setArg(15, real_p);
    kernel_jacob_k.setArg(16, (cl_int)kw);
    kernel_jacob_k.setArg(17, (cl_int)paramode);
    if(useRealPart){
        kernel_jacob_k.setArg(18, (cl_int)1);
    }else{
        kernel_jacob_k.setArg(18, (cl_int)0);
    }
    
    queue.enqueueNDRangeKernel(kernel_jacob_k, global_offset, global_item_size, local_item_size, NULL, NULL);
    queue.finish();
    
    return 0;
}

int shellObjects::inputIniCN(float iniCN, cl::Buffer edgeJ){
    
    size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    const cl::NDRange local_item_size(min((int)maxWorkGroupSize,imageSizeX),1,1);
    const cl::NDRange global_item_size(imageSizeX,imageSizeY,1);
    
    kernel_CNw.setArg(0, CN);
    kernel_CNw.setArg(1, edgeJ);
    kernel_CNw.setArg(2, (cl_float)iniCN);
    queue.enqueueNDRangeKernel(kernel_CNw, NULL, global_item_size, local_item_size, NULL, NULL);
    queue.finish();
    
    return 0;
}

int shellObjects::inputIniR(float iniR){
    
    queue.enqueueFillBuffer(dR, (cl_float)iniR-Reff, 0, sizeof(float)*imageSizeX*imageSizeY);
    return 0;
}

int shellObjects::inputInidE0(float inidE0){
    
    queue.enqueueFillBuffer(dE0, (cl_float)inidE0, 0, sizeof(float)*imageSizeX*imageSizeY);
    return 0;
}

int shellObjects::inputIniss(float iniss){
    
    queue.enqueueFillBuffer(ss, (cl_float)iniss, 0, sizeof(float)*imageSizeX*imageSizeY);
    return 0;
}

void shellObjects::setFreeFixPara(string paraname, bool val){
    
    if(paraname=="CN"){
        freeFixPara[0]=val;
    }else if(paraname=="distance"){
        freeFixPara[1]=val;
    }else if(paraname=="dE0"){
        freeFixPara[2]=val;
    }else if(paraname=="ss"){
        freeFixPara[3]=val;
    }else if(paraname=="E0imag"){
        freeFixPara[4]=val;
    }else if(paraname=="C3"){
        freeFixPara[5]=val;
    }else if(paraname=="C4"){
        freeFixPara[6]=val;
    }
}

bool shellObjects::getFreeFixPara(string paraname){
    
    bool val;
    if(paraname=="CN"){
        val=freeFixPara[0];
    }else if(paraname=="distance"){
        val=freeFixPara[1];
    }else if(paraname=="dE0"){
        val=freeFixPara[2];
    }else if(paraname=="ss"){
        val=freeFixPara[3];
    }else if(paraname=="E0imag"){
        val=freeFixPara[4];
    }else if(paraname=="C3"){
        val=freeFixPara[5];
    }else if(paraname=="C4"){
        val=freeFixPara[6];
    }else{
        val=false;
    }
    
    return val;
}

bool shellObjects::getFreeFixPara(int paramode){
    
    bool val=false;
    switch (paramode) {
        case 1:         //CN
            val=freeFixPara[0];
            break;
        case 2:         //dR
            val=freeFixPara[1];
            break;
        case 3:         //dE0
            val=freeFixPara[2];
            break;
        case 4:         //ss
            val=freeFixPara[3];
            break;
        case 5:         //E0img
            val=freeFixPara[4];
            break;
        case 6:         //C3
            val=freeFixPara[5];
            break;
        case 7:         //C4
            val=freeFixPara[6];
            break;
        default:
            break;
    }
    
    return val;
}

int shellObjects::getFreeParaSize(){
    
    int val=0;
    for (int i=0; i<7; i++) {
        val += (freeFixPara[i]) ? 1:0;
    }
    
    return val;
}

int shellObjects::copyPara(cl::Buffer dstPara, int offsetN, int paramode){
    
    switch (paramode) {
        case 1: //CN
            queue.enqueueCopyBuffer(CN, dstPara, 0, sizeof(float)*imageSizeM*offsetN, sizeof(float)*imageSizeM);
            break;
        case 2:
            queue.enqueueCopyBuffer(dR, dstPara, 0, sizeof(float)*imageSizeM*offsetN, sizeof(float)*imageSizeM);
            break;
        case 3:
            queue.enqueueCopyBuffer(dE0, dstPara, 0, sizeof(float)*imageSizeM*offsetN, sizeof(float)*imageSizeM);
            break;
        case 4:
            queue.enqueueCopyBuffer(ss, dstPara, 0, sizeof(float)*imageSizeM*offsetN, sizeof(float)*imageSizeM);
            break;
        case 5:
            queue.enqueueCopyBuffer(E0imag, dstPara, 0, sizeof(float)*imageSizeM*offsetN, sizeof(float)*imageSizeM);
            break;
        case 6:
            queue.enqueueCopyBuffer(C3, dstPara, 0, sizeof(float)*imageSizeM*offsetN, sizeof(float)*imageSizeM);
            break;
        case 7:
            queue.enqueueCopyBuffer(C4, dstPara, 0, sizeof(float)*imageSizeM*offsetN, sizeof(float)*imageSizeM);
            break;
        default:
            break;
    }
    
    return 0;
}

int shellObjects::updatePara(cl::Buffer dp_img, int paramode, int z_id){
    
    size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    const cl::NDRange local_item_size(min((int)maxWorkGroupSize,imageSizeX),1,1);
    const cl::NDRange global_item_size(imageSizeX,imageSizeY,1);
    
    kernel_update.setArg(0, dp_img);
    kernel_update.setArg(2, (cl_int)z_id);
    kernel_update.setArg(3, (cl_int)0);
    
    switch (paramode) {
        case 1: //CN
            kernel_update.setArg(1, CN);
            break;
        case 2:
            kernel_update.setArg(1, dR);
            break;
        case 3:
            kernel_update.setArg(1, dE0);
            break;
        case 4:
            kernel_update.setArg(1, ss);
            break;
        case 5:
            kernel_update.setArg(1, E0imag);
            break;
        case 6:
            kernel_update.setArg(1, C3);
            break;
        case 7:
            kernel_update.setArg(1, C4);
            break;
        default:
            break;
    }
    queue.enqueueNDRangeKernel(kernel_update,NULL,global_item_size,local_item_size,NULL,NULL);
    queue.finish();
    
    return 0;
}

int shellObjects::restorePara(cl::Buffer para_backup, int offsetN,cl::Buffer rho_img, int paramode){
    
    size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    const cl::NDRange local_item_size(min((int)maxWorkGroupSize,imageSizeX),1,1);
    const cl::NDRange global_item_size(imageSizeX,imageSizeY,1);
    
    kernel_UR.setArg(1, para_backup);
    kernel_UR.setArg(2, rho_img);
    kernel_UR.setArg(3, (cl_int)0);
    kernel_UR.setArg(4, (cl_int)offsetN);
    
    switch (paramode) {
        case 1: //CN
            kernel_UR.setArg(0, CN);
            break;
        case 2: //dR
            kernel_UR.setArg(0, dR);
            break;
        case 3: //dE0
            kernel_UR.setArg(0, dE0);
            break;
        case 4: //ss
            kernel_UR.setArg(0, ss);
            break;
        case 5: //E0img
            kernel_UR.setArg(0, E0imag);
            break;
        case 6: //C3
            kernel_UR.setArg(0, C3);
            break;
        case 7: //C4
            kernel_UR.setArg(0, C4);
            break;
        default:
            break;
    }
    queue.enqueueNDRangeKernel(kernel_UR,NULL,global_item_size,local_item_size,NULL,NULL);
    queue.finish();
    
    return 0;
}

int shellObjects::readParaImage(float* paraData, int paramode){
    
    cl::Buffer Rval;
    size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    const cl::NDRange local_item_size(min((int)maxWorkGroupSize,imageSizeX),1,1);
    const cl::NDRange global_item_size(imageSizeX,imageSizeY,1);
    
    switch (paramode) {
        case 1: //CN
            queue.enqueueReadBuffer(CN, CL_TRUE, 0, sizeof(float)*imageSizeM, paraData);
            queue.finish();
            break;
        case 2: //R
            Rval = cl::Buffer(context,CL_MEM_READ_WRITE,sizeof(cl_float)*imageSizeM,0,NULL);
            kernel_OBD.setArg(0, dR);
            kernel_OBD.setArg(1, Rval);
            kernel_OBD.setArg(2, (cl_float)Reff);
            queue.enqueueNDRangeKernel(kernel_OBD,NULL,global_item_size,local_item_size,NULL,NULL);
            queue.finish();
            queue.enqueueReadBuffer(Rval, CL_TRUE, 0, sizeof(float)*imageSizeM, paraData);
            queue.finish();
            break;
        case 3: //dE0
            queue.enqueueReadBuffer(dE0, CL_TRUE, 0, sizeof(float)*imageSizeM, paraData);
            queue.finish();
            break;
        case 4: //ss
            queue.enqueueReadBuffer(ss, CL_TRUE, 0, sizeof(float)*imageSizeM, paraData);
            queue.finish();
            break;
        case 5: //E0img
            queue.enqueueReadBuffer(E0imag, CL_TRUE, 0, sizeof(float)*imageSizeM, paraData);
            queue.finish();
            break;
        case 6: //C3
            queue.enqueueReadBuffer(C3, CL_TRUE, 0, sizeof(float)*imageSizeM, paraData);
            queue.finish();
            break;
        case 7: //C4
            queue.enqueueReadBuffer(C4, CL_TRUE, 0, sizeof(float)*imageSizeM, paraData);
            queue.finish();
            break;
        default:
            break;
    }
    
    return 0;
}


int shellObjects::constrain1(cl::Buffer eval_img, cl::Buffer C_mat,
                             int cnum, int pnum, int paramode){
    
    size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    const cl::NDRange local_item_size(min((int)maxWorkGroupSize,imageSizeX),1,1);
    const cl::NDRange global_item_size(imageSizeX,imageSizeY,1);
    
    
    switch (paramode) {
        case 1: //CN
            kernel_contrain1.setArg(0, CN);
            break;
        case 2: //dR
            kernel_contrain1.setArg(0, dR);
            break;
        case 3: //dE0
            kernel_contrain1.setArg(0, dE0);
            break;
        case 4: //ss
            kernel_contrain1.setArg(0, ss);
            break;
        case 5: //E0img
            kernel_contrain1.setArg(0, E0imag);
            break;
        case 6: //C3
            kernel_contrain1.setArg(0, C3);
            break;
        case 7: //C4
            kernel_contrain1.setArg(0, C4);
            break;
        default:
            break;
    }
    kernel_contrain1.setArg(1, eval_img);
    kernel_contrain1.setArg(2, C_mat);
    kernel_contrain1.setArg(3, (cl_int)cnum);
    kernel_contrain1.setArg(4, (cl_int)pnum);
    queue.enqueueNDRangeKernel(kernel_contrain1,NULL,global_item_size,local_item_size,NULL,NULL);
    queue.finish();
    
    
    return 0;
}


int shellObjects::constrain2(cl::Buffer eval_img, cl::Buffer edgeJ, cl::Buffer C_mat, cl::Buffer D_vec, cl::Buffer C2_vec, int cnum, int pnum, int paramode){
    
    size_t maxWorkGroupSize = queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    const cl::NDRange local_item_size(min((int)maxWorkGroupSize,imageSizeX),1,1);
    const cl::NDRange global_item_size(imageSizeX,imageSizeY,1);
    
    
    switch (paramode) {
        case 1: //CN
            kernel_contrain2.setArg(0, CN);
            kernel_contrain2.setArg(8, (cl_char)49);
            break;
        case 2: //dR
            kernel_contrain2.setArg(0, dR);
            kernel_contrain2.setArg(8, (cl_char)48);
            break;
        case 3: //dE0
            kernel_contrain2.setArg(0, dE0);
            kernel_contrain2.setArg(8, (cl_char)48);
            break;
        case 4: //ss
            kernel_contrain2.setArg(0, ss);
            kernel_contrain2.setArg(8, (cl_char)48);
            break;
        case 5: //E0img
            kernel_contrain2.setArg(0, E0imag);
            kernel_contrain2.setArg(8, (cl_char)48);
            break;
        case 6: //C3
            kernel_contrain2.setArg(0, C3);
            kernel_contrain2.setArg(8, (cl_char)48);
            break;
        case 7: //C4
            kernel_contrain2.setArg(0, C4);
            kernel_contrain2.setArg(8, (cl_char)48);
            break;
        default:
            break;
    }
    kernel_contrain2.setArg(1, edgeJ);
    kernel_contrain2.setArg(2, eval_img);
    kernel_contrain2.setArg(3, C_mat);
    kernel_contrain2.setArg(4, D_vec);
    kernel_contrain2.setArg(5, C2_vec);
    kernel_contrain2.setArg(6, (cl_int)cnum);
    kernel_contrain2.setArg(7, (cl_int)pnum);
    queue.enqueueNDRangeKernel(kernel_contrain2,NULL,global_item_size,local_item_size,NULL,NULL);
    queue.finish();
    
    return 0;
}
