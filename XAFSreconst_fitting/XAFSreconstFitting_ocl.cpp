//
//  XAFSreconstFitting_ocl.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2017/02/16.
//  Copyright © 2017年 Nozomu Ishiguro. All rights reserved.
//

#include "XANES_fitting.hpp"
#include "CT_reconstruction.hpp"
//#include "XAFSreconstFitting_cl.hpp"

int readRawFile_offset(string filepath_input,float *binImgf, int64_t offset, int64_t size);

string kernel_reconst_preprocessor_nums(){
    ostringstream OSS;
    
    OSS << " -D IMAGESIZE_X=" << g_nx;
    
    OSS << " -D IMAGESIZE_Y=" << g_nx;
    
    OSS << " -D IMAGESIZE_M=" << g_nx*g_nx;
    
    OSS << " -D PRJ_IMAGESIZE=" << g_px;
    
    OSS << " -D PRJ_ANGLESIZE=" << g_pa;
    
    OSS << " -D PRJ_IMAGESIZE_M=" << g_px*g_pa;
    
    OSS << " -D SS=" << g_ss;

	OSS << " -D SS_ANGLESIZE=" << g_pa/g_ss;
    
    OSS << " -D AMP_FACTOR=" << amp;
    
    return OSS.str();
}

int reconstFitResult_output_thread(fitting_eq fit_eq, int Z, input_parameter inp, vector<float*> result_outputs){
    
    string output_dir = inp.getFittingOutputDir();
    string output_base = inp.getOutputFileBase();
    ostringstream oss;
    for (int i=0; i<fit_eq.ParaSize(); i++) {
        char buffer;
        buffer=fit_eq.freefix_para()[i];
        if (atoi(&buffer)==1) {
            string fileName_output= output_dir+ "/"+fit_eq.param_name(i)+"/"+numTagString(Z, output_base, ".raw", 4);
            oss << "output file: " << fileName_output << endl;
            outputRawFile_stream(fileName_output,result_outputs[i],IMAGE_SIZE_M);
        }
    }
    oss << endl;
    cout << oss.str();
    for (int i=0; i<fit_eq.ParaSize(); i++) {
        delete [] result_outputs[i];
    }
    return 0;
}

int XAFSreconstFit_AART_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernels,
                                     fitting_eq fiteq, int Z, int thread_id, input_parameter inp,
                                     cl::Buffer energy_buff, cl::Buffer angle_buff,
                                     cl::Buffer C_matrix_buff, cl::Buffer D_vector_buff,
                                     cl::Buffer freeFix_buff, cl::Buffer attenuator_buff,
                                     vector<float*> prj_mt_vec, int *sub){
    try {
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        string devicename = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>();
        size_t maxWorkGroupSize = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        
        int fittingStartEnergyNo = inp.getFittingStartEnergyNo();
        int fittingEndEnergyNo = inp.getFittingEndEnergyNo();
        int startEnergyNo=inp.getStartEnergyNo();
        int num_energy = fittingEndEnergyNo - fittingStartEnergyNo + 1;
        int paramsize = (int)fiteq.ParaSize();
        
        
        
        cout <<"GPU("<<thread_id+1<<") : Processing Z No. " << Z << "..." << endl << endl;
        
        
        //OCLオブジェクト
        //M-Lパラメータ
        cl::Buffer lambda_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        cl::Buffer nyu_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        cl::Buffer dL_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        cl::Buffer chi2_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*g_px*g_pa/g_ss, 0, NULL);
        //cl::Buffer chi2_new_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*g_px*g_pa/g_ss, 0, NULL);
        
        //フィッティングパラメータ
        cl::Buffer results_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*paramsize, 0, NULL);
        cl::Buffer results_cnd_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*paramsize, 0, NULL);
        
        //mtデータ
        cl::ImageFormat format(CL_R, CL_FLOAT);
        cl::Image2DArray mt_fit_img(context, CL_MEM_READ_WRITE, format, num_energy, IMAGE_SIZE_X, IMAGE_SIZE_Y, 0, 0, NULL, NULL);
        cl::Image2D prj_dummy_img(context, CL_MEM_READ_WRITE, format, g_px, g_pa, 0, NULL, NULL);
        cl::Image2DArray prj_mt_img(context, CL_MEM_READ_WRITE, format, num_energy, g_px, g_pa, 0, 0, NULL, NULL);
        cl::Image2DArray prj_delta_mt_img(context, CL_MEM_READ_WRITE, format, num_energy, g_px, g_pa/g_ss, 0, 0, NULL, NULL);
        cl::Buffer bprj_delta_mt_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*num_energy, 0, NULL);
        
        //ヤコビアンetc.
        //2048*2048*paramsize(8)*num_energy(146)x4byteのオブジェクトは現時点でGPUハード的にあつかえない
        cl::Image2DArray jacob_img(context, CL_MEM_READ_WRITE, format, paramsize, IMAGE_SIZE_X, IMAGE_SIZE_Y, 0, 0, NULL, NULL);
        cl::Image2DArray prj_jacob_img(context, CL_MEM_READ_WRITE, format, paramsize, g_px, g_pa/g_ss, 0, 0, NULL, NULL);
        cl::Buffer bprj_jacob_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*paramsize, 0, NULL);
        cl::Buffer tJJ_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*paramsize*paramsize, 0, NULL);
        cl::Buffer tJdF_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*paramsize, 0, NULL);
        cl::Buffer dummy_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*paramsize, 0, NULL);
        
        
        //lambda, nyu, フィッティングパラメータの初期化
        float lambda = inp.getLambda_t_fit();
        float nyu = 2.0f;
        float chi2_old=0.0f;
        float chi2_new=0.0f;
        float dL=0.0f;
        for (int i=0; i<paramsize; i++) {
            command_queue.enqueueFillBuffer(results_img, (cl_float)fiteq.fit_para()[i], sizeof(cl_float)*IMAGE_SIZE_M*i, sizeof(cl_float)*IMAGE_SIZE_M,NULL,NULL);
        }
        
        
        //フィッティング結果格納先
        vector<float*> results_vec;
        for (int i=0; i<paramsize; i++) {
            results_vec.push_back(new float[IMAGE_SIZE_M]);
        }
        
#ifdef DEBUG
        vector<float> chi2_vec;
        vector<float> lambda_vec;
#endif
        
#ifdef DEBUG
        cout<<"GPU("<<thread_id+1<<"): upload data to GPU"<<endl;
#endif
        const cl::NDRange local_item_size1(maxWorkGroupSize,1,1);
        const cl::NDRange global_item_size0(g_px,g_pa,1);
        const cl::NDRange global_item_size1(IMAGE_SIZE_X,IMAGE_SIZE_Y,1);
        const cl::NDRange global_item_size2(IMAGE_SIZE_X,IMAGE_SIZE_Y,paramsize);
        const cl::NDRange global_item_size4(g_px,g_pa/g_ss,1);
        const cl::NDRange global_item_size5(g_px,g_pa/g_ss,paramsize);
        const cl::NDRange global_item_size11_1(maxWorkGroupSize,IMAGE_SIZE_Y,1);
        const cl::NDRange global_item_size11_2(maxWorkGroupSize,1,1);
        const cl::NDRange global_item_size16(IMAGE_SIZE_X,IMAGE_SIZE_Y,num_energy);
        const cl::NDRange global_item_size18(g_px,g_pa/g_ss,num_energy);
        const cl::NDRange global_item_size7_2(IMAGE_SIZE_X,IMAGE_SIZE_Y,paramsize);
        
        cl::size_t<3> origin;
        cl::size_t<3> region;
        cl::size_t<3> region2;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        region[0] = g_px;
        region[1] = g_pa;
        region[2] = 1;
        region2[0] = IMAGE_SIZE_X;
        region2[1] = IMAGE_SIZE_Y;
        region2[2] = 1;
        
        
        for (int i = fittingStartEnergyNo; i<=fittingEndEnergyNo; i++) {
            //投影mtデータをGPUにアップロード
            command_queue.enqueueWriteImage(prj_dummy_img, CL_TRUE, origin, region, g_px*sizeof(float), 0, prj_mt_vec[i-startEnergyNo], NULL, NULL);
            
            //sinogram correction
            kernels[0].setArg(0, prj_dummy_img);
            kernels[0].setArg(1, prj_mt_img);
            kernels[0].setArg(2, angle_buff);
            kernels[0].setArg(3, (cl_int)correctionMode);
            kernels[0].setArg(4, (cl_int)i-fittingStartEnergyNo);
            command_queue.enqueueNDRangeKernel(kernels[0], NULL, global_item_size0, local_item_size1, NULL, NULL);
            command_queue.finish();
            
        }
        //最初のE点で逆投影する
        kernels[1].setArg(0, dummy_img);
        kernels[1].setArg(1, prj_mt_img);
        kernels[1].setArg(2, angle_buff);
        command_queue.enqueueFillBuffer(dummy_img, (cl_float)0.0f, 0, sizeof(cl_float)*IMAGE_SIZE_M, NULL, NULL);
        command_queue.enqueueNDRangeKernel(kernels[1], NULL, global_item_size1, local_item_size1, NULL, NULL);
        command_queue.finish();
        
        
        //初期パラメータをmt逆投影像で重み付け
        kernels[2].setArg(0, results_img);
        kernels[2].setArg(1, dummy_img);
        kernels[2].setArg(2, attenuator_buff);
        command_queue.enqueueNDRangeKernel(kernels[2], NULL, global_item_size2, local_item_size1, NULL, NULL);
        command_queue.finish();
        
        
        //初期パラメータデータを円形にマスク
        /*kernels[21].setArg(0, results_img);
         kernels[21].setArg(1, attenuator_buff);
         command_queue.enqueueNDRangeKernel(kernels[21], NULL, global_item_size2, local_item_size1, NULL, NULL);
         command_queue.finish();*/
        
        
        //fitting試行開始
        //set kernel (代入)
        kernels[15].setArg(1, mt_fit_img);
        kernels[15].setArg(2, energy_buff);
        //set kernel (Jacobian代入)
        kernels[13].setArg(0, results_img);
        kernels[13].setArg(1, jacob_img);
        kernels[13].setArg(2, energy_buff);
        /*kernels[20].setArg(0, results_img);
        kernels[20].setArg(1, bprj_jacob_buff);
        kernels[20].setArg(2, energy_buff);*/
        //set kernel (delta mtへの投影計算)
        kernels[17].setArg(0, mt_fit_img);
        kernels[17].setArg(1, prj_mt_img);
        kernels[17].setArg(2, prj_delta_mt_img);
        kernels[17].setArg(3, angle_buff);
        //set kernel (Jacobianの投影)
        kernels[5].setArg(0, jacob_img);
        kernels[5].setArg(1, prj_jacob_img);
        kernels[5].setArg(2, angle_buff);
        //set kernel (prj delta_mt, Jacobの逆投影)
        kernels[7].setArg(2, angle_buff);
        //set kernel (chi2の計算その1)
        kernels[18].setArg(0, chi2_buff);
        kernels[18].setArg(1, prj_delta_mt_img);
        //set kernel (chi2の計算その1)
        kernels[14].setArg(0, chi2_buff);
        kernels[14].setArg(1, cl::Local(sizeof(cl_float)*maxWorkGroupSize));//locmem
        //set kernel (tJJ, tJdF計算)
        kernels[19].setArg(0, bprj_jacob_buff);
        kernels[19].setArg(1, bprj_delta_mt_buff);
        kernels[19].setArg(2, tJJ_img);
        kernels[19].setArg(3, tJdF_img);
        //set kernel (新パラメータ候補の計算)
        kernels[10].setArg(0, results_cnd_img);
        kernels[10].setArg(1, tJJ_img);
        kernels[10].setArg(2, tJdF_img);
        kernels[10].setArg(4, dL_buff);
        kernels[10].setArg(5, freeFix_buff);
        //set kernel (拘束条件適用)
        kernels[11].setArg(0, results_cnd_img);
        kernels[11].setArg(1, C_matrix_buff);
        kernels[11].setArg(2, D_vector_buff);
        //set kernel (dL計算)
        kernels[12].setArg(0, dL_buff);
        kernels[12].setArg(1, cl::Local(sizeof(cl_float)*maxWorkGroupSize));//locmem
        
        for (int trial=0; trial<g_it; trial++) {
#ifdef DEBUG
            cout<<"GPU("<<thread_id+1<<"): trial "<<trial+1<<"/"<<g_it<<endl;
#endif
            for(int k=0; k<g_ss; k++){
#ifdef DEBUG
                cout<<" GPU("<<thread_id+1<<"): subset "<<k+1<<"/"<<g_ss<<endl;
                cout<<"     GPU("<<thread_id+1<<"): calc mt_fit"<<endl;
#endif
                //パラメータをフィッティング式に代入
                kernels[15].setArg(0, results_img);
                command_queue.enqueueNDRangeKernel(kernels[15], NULL, global_item_size16, local_item_size1, NULL, NULL);
                command_queue.finish();
                
                //delta mtへの投影計算
#ifdef DEBUG
                cout<<"     GPU("<<thread_id+1<<"): calc prj. dF"<<endl;
#endif
                kernels[17].setArg(4, sub[k]);
                command_queue.enqueueNDRangeKernel(kernels[17], NULL, global_item_size18, local_item_size1, NULL, NULL);
                command_queue.finish();
                
                //prj delta mtの逆投影
#ifdef DEBUG
                cout<<"     GPU("<<thread_id+1<<"): calc back-prj. dF"<<endl;
#endif
                kernels[7].setArg(0, bprj_delta_mt_buff);
                kernels[7].setArg(1, prj_delta_mt_img);
                kernels[7].setArg(3, sub[k]);
                command_queue.enqueueNDRangeKernel(kernels[7], NULL, global_item_size16, local_item_size1, NULL, NULL);
                command_queue.finish();
                
#ifdef DEBUG
                cout<<"     GPU("<<thread_id+1<<"): Estimate chi2(old)"<<endl;
#endif
                //chi2(old)計算その1
                command_queue.enqueueFillBuffer(chi2_buff, (cl_float)0.0f, 0, sizeof(cl_float)*g_px*g_pa/g_ss,NULL,NULL);
                command_queue.enqueueNDRangeKernel(kernels[18], NULL, global_item_size4, local_item_size1, NULL, NULL);
                command_queue.finish();
                //chi2(old)計算その2
                command_queue.enqueueNDRangeKernel(kernels[14], NULL, global_item_size11_2, local_item_size1, NULL, NULL);
                command_queue.finish();
                command_queue.enqueueReadBuffer(chi2_buff, CL_TRUE, 0, sizeof(float), &chi2_old);
#ifdef DEBUG
                chi2_vec.push_back(chi2_old);
#endif
                
                
#ifdef DEBUG
                cout<<"     GPU("<<thread_id+1<<"): calc tJJ and tJdF"<<endl;
#endif
                kernels[5].setArg(3, sub[k]);
                command_queue.enqueueFillBuffer(tJJ_img, (cl_float)0.0f, 0, sizeof(cl_float)*IMAGE_SIZE_M*paramsize*paramsize,NULL,NULL);
                command_queue.enqueueFillBuffer(tJdF_img, (cl_float)0.0f, 0, sizeof(cl_float)*IMAGE_SIZE_M*paramsize,NULL,NULL);
                for (int i=fittingStartEnergyNo; i<=fittingEndEnergyNo; i++) {
                    //パラメータをJacobianに代入
                    kernels[13].setArg(3, i-fittingStartEnergyNo);
                    command_queue.enqueueNDRangeKernel(kernels[13], NULL, global_item_size1, local_item_size1, NULL, NULL);
                    command_queue.finish();
                    
                    //Jacobianの投影
                    command_queue.enqueueNDRangeKernel(kernels[5], NULL, global_item_size5, local_item_size1, NULL, NULL);
                    command_queue.finish();
                    
                    //prj Jacobの逆投影
                    kernels[7].setArg(0, bprj_jacob_buff);
                    kernels[7].setArg(1, prj_jacob_img);
                    command_queue.enqueueNDRangeKernel(kernels[7], NULL, global_item_size7_2, local_item_size1, NULL, NULL);
                    command_queue.finish();
                    
                    //パラメータをJacobianに代入
                    /*kernels[20].setArg(3, i-fittingStartEnergyNo);
                    command_queue.enqueueNDRangeKernel(kernels[20], NULL, global_item_size1, local_item_size1, NULL, NULL);
                    command_queue.finish();*/
                    
                    //tJJ, tJdF計算
                    kernels[19].setArg(4, i-fittingStartEnergyNo);
                    command_queue.enqueueNDRangeKernel(kernels[19], NULL, global_item_size1, local_item_size1, NULL, NULL);
                    command_queue.finish();
                }
                
                
                
                //新パラメータ候補の計算
#ifdef DEBUG
                cout<<"     GPU("<<thread_id+1<<"): calculate new parameter candidate"<<endl;
                lambda_vec.push_back(lambda);
#endif
                kernels[10].setArg(3, lambda);
                command_queue.enqueueCopyBuffer(results_img, results_cnd_img, 0, 0, sizeof(cl_float)*IMAGE_SIZE_M*paramsize);
                command_queue.enqueueNDRangeKernel(kernels[10], NULL, global_item_size1, local_item_size1, NULL, NULL);
                command_queue.finish();
                
                
                //拘束条件の適用
#ifdef DEBUG
                cout<<"     GPU("<<thread_id+1<<"): set constrain"<<endl;
#endif
                command_queue.enqueueNDRangeKernel(kernels[11], NULL, global_item_size1, local_item_size1, NULL, NULL);
                command_queue.finish();
                
                
                //dLの計算
#ifdef DEBUG
                cout<<"     GPU("<<thread_id+1<<"): calc dL"<<endl;
#endif
                command_queue.enqueueNDRangeKernel(kernels[11], NULL, global_item_size11_1, local_item_size1, NULL, NULL);
                command_queue.finish();
                command_queue.enqueueNDRangeKernel(kernels[11], NULL, global_item_size11_2, local_item_size1, NULL, NULL);
                command_queue.finish();
                
                
                //新パラメータ候補の評価
                //パラメータをフィッティング式に代入
#ifdef DEBUG
                cout<<"     GPU("<<thread_id+1<<"): calc new mt candidate"<<endl;
#endif
                kernels[15].setArg(0, results_cnd_img);
                command_queue.enqueueNDRangeKernel(kernels[15], NULL, global_item_size16, local_item_size1, NULL, NULL);
                command_queue.finish();
                
                //delta mtへの投影計算
#ifdef DEBUG
                cout<<"     GPU("<<thread_id+1<<"): calc new prj dF candidate"<<endl;
#endif
                command_queue.enqueueNDRangeKernel(kernels[17], NULL, global_item_size18, local_item_size1, NULL, NULL);
                command_queue.finish();
                
                //chi2(new)計算その1
#ifdef DEBUG
                cout<<"     GPU("<<thread_id+1<<"): evaluate chi2(new)"<<endl;
#endif
                command_queue.enqueueFillBuffer(chi2_buff, (cl_float)0.0f, 0, sizeof(cl_float)*g_px*g_pa/g_ss,NULL,NULL);
                command_queue.enqueueNDRangeKernel(kernels[18], NULL, global_item_size4, local_item_size1, NULL, NULL);
                command_queue.finish();
                //chi2(new)計算その2
                command_queue.enqueueNDRangeKernel(kernels[14], NULL, global_item_size11_2, local_item_size1, NULL, NULL);
                command_queue.finish();
                
                
                //パラメータの更新
#ifdef DEBUG
                cout<<"     GPU("<<thread_id+1<<"): update paramater"<<endl;
#endif
                
                command_queue.enqueueReadBuffer(chi2_buff, CL_TRUE, 0, sizeof(float), &chi2_new);
                command_queue.enqueueReadBuffer(dL_buff, CL_TRUE, 0, sizeof(float), &dL);
                float rho = (chi2_old-chi2_new)/dL*g_pa/g_ss/g_px;
                cout << "       rho=("<<chi2_old<<"-"<<chi2_new<<")/(" << dL << ")*"<<g_pa/g_ss<<"/"<<g_px<<"="<<rho<<endl;
                if(rho>=0.0f){
                    float l_A=(2.0f*rho-1.0f);
                    l_A = 1.0f-l_A*l_A*l_A;
                    l_A = max(0.333f,l_A)*lambda;
                    lambda = l_A;
                    lambda = (lambda<1e-9) ? 0:lambda;
                    nyu=2.0f;
                    command_queue.enqueueCopyBuffer(results_cnd_img, results_img, 0, 0, sizeof(cl_float)*IMAGE_SIZE_M*paramsize);
#ifdef DEBUG
                    cout<<"        updated"<<endl;
                    cout << "       next lambda="<<lambda<<endl;
#endif
                }else{
                    lambda=nyu*lambda;
                    nyu=2.0f*nyu;
#ifdef DEBUG
                    cout<<"        not updated"<<endl;
                    cout << "       next lambda="<<lambda<<endl;
#endif
                }
            }
        }
        
#ifdef DEBUG
        string fileName_log=inp.getFittingOutputDir()+ "/chi2.log";
        ofstream ofs(fileName_log,ios::out|ios::trunc);
        ofs<<"trial"<<"\t"<<"subset"<<"\t"<<"chi2(old)"<<"\t"<<"lambda"<<endl;
        for (int i=0; i<g_it; i++) {
            for (int j=0; j<g_ss; j++) {
                ofs<<i<<"\t"<<j<<"\t"<<chi2_vec[j+i*g_ss]<<"\t"<<lambda_vec[j+i*g_ss]<<endl;
            }
        }
        ofs.close();
#endif
        
        //read results image from buffer
        for (int i = 0; i<paramsize; i++) {
            command_queue.enqueueReadBuffer(results_img, CL_TRUE, sizeof(cl_float)*IMAGE_SIZE_M*i, sizeof(cl_float)*IMAGE_SIZE_M, results_vec[i]);
        }
        
        //output thread
        output_th_fit[thread_id].join();
        output_th_fit[thread_id]=thread(reconstFitResult_output_thread, fiteq,Z,inp,move(results_vec));
        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
    
    return 0;
}

int XAFSreconstFit_AART_light_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernels,
                               fitting_eq fiteq, int Z, int thread_id, input_parameter inp,
                               cl::Buffer energy_buff, cl::Buffer angle_buff,
                               cl::Buffer C_matrix_buff, cl::Buffer D_vector_buff,
                               cl::Buffer freeFix_buff, cl::Buffer attenuator_buff,
                               vector<float*> prj_mt_vec, int *sub){
    try {
        cl::Context context = command_queue.getInfo<CL_QUEUE_CONTEXT>();
        string devicename = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_NAME>();
        size_t maxWorkGroupSize = command_queue.getInfo<CL_QUEUE_DEVICE>().getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        
        int fittingStartEnergyNo = inp.getFittingStartEnergyNo();
        int fittingEndEnergyNo = inp.getFittingEndEnergyNo();
        int startEnergyNo=inp.getStartEnergyNo();
        int num_energy = fittingEndEnergyNo - fittingStartEnergyNo + 1;
        int paramsize = (int)fiteq.ParaSize();
        
        
        
        cout <<"GPU("<<thread_id+1<<") : Processing Z No. " << Z << "..." << endl << endl;
        
        
        //OCLオブジェクト
        //M-Lパラメータ
        cl::Buffer lambda_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        cl::Buffer nyu_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        cl::Buffer dL_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        cl::Buffer chi2_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*g_px*g_pa/g_ss, 0, NULL);
		//cl::Buffer chi2_new_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*g_px*g_pa/g_ss, 0, NULL);
        
        //フィッティングパラメータ
        cl::Buffer results_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*paramsize, 0, NULL);
        cl::Buffer results_cnd_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*paramsize, 0, NULL);
        
        //mtデータ
        cl::ImageFormat format(CL_R, CL_FLOAT);
        cl::Image2D mt_fit_img(context, CL_MEM_READ_WRITE, format, IMAGE_SIZE_X, IMAGE_SIZE_Y, 0, NULL, NULL);
        cl::Image2D prj_dummy_img(context, CL_MEM_READ_WRITE, format, g_px, g_pa, 0, NULL, NULL);
        cl::Image2DArray prj_mt_img(context, CL_MEM_READ_WRITE, format, num_energy, g_px, g_pa, 0, 0, NULL, NULL);
        cl::Image2D prj_delta_mt_img(context, CL_MEM_READ_WRITE, format, g_px, g_pa/g_ss, 0, NULL, NULL);
        cl::Buffer bprj_delta_mt_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M, 0, NULL);
        
        //ヤコビアンetc.
        cl::Image2DArray jacob_img(context, CL_MEM_READ_WRITE, format, paramsize, IMAGE_SIZE_X, IMAGE_SIZE_Y, 0, 0, NULL, NULL);
        cl::Image2DArray prj_jacob_img(context, CL_MEM_READ_WRITE, format, paramsize, g_px, g_pa/g_ss, 0, 0, NULL, NULL);
        cl::Buffer bprj_jacob_buff(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*paramsize, 0, NULL);
        cl::Buffer tJJ_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*paramsize*paramsize, 0, NULL);
        cl::Buffer tJdF_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*paramsize, 0, NULL);
        cl::Buffer dummy_img(context, CL_MEM_READ_WRITE, sizeof(cl_float)*IMAGE_SIZE_M*paramsize, 0, NULL);
        
        
        //lambda, nyu, フィッティングパラメータの初期化
        float lambda = inp.getLambda_t_fit();
        float nyu = 2.0f;
        float chi2_old=0.0f;
        float chi2_new=0.0f;
        float dL=0.0f;
        for (int i=0; i<paramsize; i++) {
            command_queue.enqueueFillBuffer(results_img, (cl_float)fiteq.fit_para()[i], sizeof(cl_float)*IMAGE_SIZE_M*i, sizeof(cl_float)*IMAGE_SIZE_M,NULL,NULL);
        }
        
        
        //フィッティング結果格納先
        vector<float*> results_vec;
        for (int i=0; i<paramsize; i++) {
            results_vec.push_back(new float[IMAGE_SIZE_M]);
        }
        
#ifdef DEBUG
        cout<<"GPU("<<thread_id+1<<"): upload data to GPU"<<endl;
#endif
        const cl::NDRange local_item_size1(maxWorkGroupSize,1,1);
        const cl::NDRange global_item_size0(g_px,g_pa,1);
        const cl::NDRange global_item_size1(IMAGE_SIZE_X,IMAGE_SIZE_Y,1);
        const cl::NDRange global_item_size2(IMAGE_SIZE_X,IMAGE_SIZE_Y,paramsize);
        const cl::NDRange global_item_size4(g_px,g_pa/g_ss,1);
        const cl::NDRange global_item_size5(g_px,g_pa/g_ss,paramsize);
        const cl::NDRange global_item_size11_1(maxWorkGroupSize,IMAGE_SIZE_Y,1);
        const cl::NDRange global_item_size11_2(maxWorkGroupSize,1,1);
        
        cl::size_t<3> origin;
        cl::size_t<3> region;
        cl::size_t<3> region2;
        origin[0] = 0;
        origin[1] = 0;
        origin[2] = 0;
        region[0] = g_px;
        region[1] = g_pa;
        region[2] = 1;
        region2[0] = IMAGE_SIZE_X;
        region2[1] = IMAGE_SIZE_Y;
        region2[2] = 1;
        
        
        for (int i = fittingStartEnergyNo; i<=fittingEndEnergyNo; i++) {
            //投影mtデータをGPUにアップロード
            command_queue.enqueueWriteImage(prj_dummy_img, CL_TRUE, origin, region, g_px*sizeof(float), 0, prj_mt_vec[i-startEnergyNo], NULL, NULL);
            
            //sinogram correction
            kernels[0].setArg(0, prj_dummy_img);
            kernels[0].setArg(1, prj_mt_img);
            kernels[0].setArg(2, angle_buff);
            kernels[0].setArg(3, (cl_int)correctionMode);
            kernels[0].setArg(4, (cl_int)i-fittingStartEnergyNo);
            command_queue.enqueueNDRangeKernel(kernels[0], NULL, global_item_size0, local_item_size1, NULL, NULL);
            command_queue.finish();
            
        }
        //最初のE点で逆投影する
        kernels[1].setArg(0, dummy_img);
        kernels[1].setArg(1, prj_mt_img);
        kernels[1].setArg(2, angle_buff);
        command_queue.enqueueFillBuffer(dummy_img, (cl_float)0.0f, 0, sizeof(cl_float)*IMAGE_SIZE_M, NULL, NULL);
        command_queue.enqueueNDRangeKernel(kernels[1], NULL, global_item_size1, local_item_size1, NULL, NULL);
        command_queue.finish();
        /*float *bprj_mt;
         bprj_mt = new float[IMAGE_SIZE_M];
         command_queue.enqueueReadBuffer(dummy_img, CL_TRUE, 0, sizeof(float)*IMAGE_SIZE_M, bprj_mt);
         string fileName_output= inp.getFittingOutputDir()+ "/"+numTagString(1, "bprjmt", ".raw", 3);
         outputRawFile_stream(fileName_output,bprj_mt,IMAGE_SIZE_M);
         cout << "finish" << endl;
         delete[] bprj_mt;*/
        
        
        //初期パラメータをmt逆投影像で重み付け
        kernels[2].setArg(0, results_img);
        kernels[2].setArg(1, dummy_img);
        kernels[2].setArg(2, attenuator_buff);
        command_queue.enqueueNDRangeKernel(kernels[2], NULL, global_item_size2, local_item_size1, NULL, NULL);
        command_queue.finish();
        
        
        //初期パラメータデータを円形にマスク
        /*kernels[*].setArg(0, results_img);
        kernels[*].setArg(1, attenuator_buff);
        command_queue.enqueueNDRangeKernel(kernels[*], NULL, global_item_size*, local_item_size1, NULL, NULL);
        command_queue.finish();*/
        
        
        //fitting試行開始
        //set kernel (代入)
        kernels[3].setArg(1, mt_fit_img);
        kernels[3].setArg(2, energy_buff);
        //set kernel (Jacobian代入)
        kernels[13].setArg(1, jacob_img);
        kernels[13].setArg(2, energy_buff);
        //set kernel (delta mtへの投影計算)
        kernels[4].setArg(0, mt_fit_img);
        kernels[4].setArg(1, prj_mt_img);
        kernels[4].setArg(2, prj_delta_mt_img);
        kernels[4].setArg(3, angle_buff);
        //set kernel (Jacobianの投影)
        kernels[5].setArg(0, jacob_img);
        kernels[5].setArg(1, prj_jacob_img);
        kernels[5].setArg(2, angle_buff);
        //set kernel (prj delta mtの逆投影)
        kernels[6].setArg(0, bprj_delta_mt_buff);
        kernels[6].setArg(1, prj_delta_mt_img);
        kernels[6].setArg(2, angle_buff);
        //set kernel (prj Jacobの逆投影)
        kernels[7].setArg(0, bprj_jacob_buff);
        kernels[7].setArg(1, prj_jacob_img);
        kernels[7].setArg(2, angle_buff);
        //set kernel (chi2の計算その1)
        kernels[8].setArg(1, prj_delta_mt_img);
        //set kernel (chi2の計算その1)
        kernels[14].setArg(1, cl::Local(sizeof(cl_float)*maxWorkGroupSize));//locmem
        kernels[15].setArg(1, prj_delta_mt_img);
        //set kernel (tJJ, tJdF計算)
        kernels[9].setArg(0, bprj_jacob_buff);
        kernels[9].setArg(1, bprj_delta_mt_buff);
        kernels[9].setArg(2, tJJ_img);
        kernels[9].setArg(3, tJdF_img);
        //set kernel (新パラメータ候補の計算)
        kernels[10].setArg(0, results_cnd_img);
        kernels[10].setArg(1, tJJ_img);
        kernels[10].setArg(2, tJdF_img);
        kernels[10].setArg(4, dL_buff);
        kernels[10].setArg(5, freeFix_buff);
        //set kernel (拘束条件適用)
        kernels[11].setArg(0, results_cnd_img);
        kernels[11].setArg(1, C_matrix_buff);
        kernels[11].setArg(2, D_vector_buff);
        //set kernel (dL計算)
        kernels[12].setArg(0, dL_buff);
        kernels[12].setArg(1, cl::Local(sizeof(cl_float)*maxWorkGroupSize));//locmem
        for (int trial=0; trial<g_it; trial++) {
#ifdef DEBUG
            cout<<"GPU("<<thread_id+1<<"): trial "<<trial+1<<"/"<<g_it<<endl;
#endif
            for(int k=0; k<g_ss; k++){
#ifdef DEBUG
                cout<<" GPU("<<thread_id+1<<"): subset "<<k+1<<"/"<<g_ss<<endl;
                cout<<"     GPU("<<thread_id+1<<"): calculate tJJ, tJdF, chi2(old)"<<endl;
#endif
                command_queue.enqueueFillBuffer(tJJ_img, (cl_float)0.0f, 0, sizeof(cl_float)*IMAGE_SIZE_M*paramsize*paramsize,NULL,NULL);
                command_queue.enqueueFillBuffer(tJdF_img, (cl_float)0.0f, 0, sizeof(cl_float)*IMAGE_SIZE_M*paramsize,NULL,NULL);
                command_queue.enqueueFillBuffer(chi2_buff, (cl_float)0.0f, 0, sizeof(cl_float)*g_px*g_pa/g_ss,NULL,NULL);
                command_queue.enqueueFillBuffer(bprj_delta_mt_buff, (cl_float)0.0f, 0, sizeof(cl_float)*IMAGE_SIZE_M,NULL,NULL);
                command_queue.enqueueFillBuffer(bprj_jacob_buff, (cl_float)0.0f, 0, sizeof(cl_float)*IMAGE_SIZE_M*paramsize,NULL,NULL);
                kernels[3].setArg(0, results_img);
                kernels[13].setArg(0, results_img);
                kernels[4].setArg(4, sub[k]);
                kernels[5].setArg(3, sub[k]);
                kernels[6].setArg(3, sub[k]);
                kernels[7].setArg(3, sub[k]);
                kernels[8].setArg(0, chi2_buff);
                kernels[14].setArg(0, chi2_buff);
                for (int i = fittingStartEnergyNo; i<=fittingEndEnergyNo; i++) {
                    //パラメータをフィッティング式に代入
                    kernels[3].setArg(3, i-fittingStartEnergyNo);
                    command_queue.enqueueNDRangeKernel(kernels[3], NULL, global_item_size1, local_item_size1, NULL, NULL);
                    command_queue.finish();
                    kernels[13].setArg(3, i-fittingStartEnergyNo);
                    command_queue.enqueueNDRangeKernel(kernels[13], NULL, global_item_size1, local_item_size1, NULL, NULL);
                    command_queue.finish();
                    
                    //delta mtへの投影計算
                    kernels[4].setArg(5, i-fittingStartEnergyNo);
                    command_queue.enqueueNDRangeKernel(kernels[4], NULL, global_item_size4, local_item_size1, NULL, NULL);
                    command_queue.finish();
                    
                    //Jacobianの投影
                    command_queue.enqueueNDRangeKernel(kernels[5], NULL, global_item_size5, local_item_size1, NULL, NULL);
                    command_queue.finish();
                    
                    //prj delta mtの逆投影
                    command_queue.enqueueNDRangeKernel(kernels[6], NULL, global_item_size1, local_item_size1, NULL, NULL);
                    command_queue.finish();
                        
                    //prj Jacobの逆投影
                    command_queue.enqueueNDRangeKernel(kernels[7], NULL, global_item_size2, local_item_size1, NULL, NULL);
                    command_queue.finish();
                    
                    //chi2(old)計算その1
                    command_queue.enqueueNDRangeKernel(kernels[8], NULL, global_item_size4, local_item_size1, NULL, NULL);
                    command_queue.finish();
                    
                    //tJJ, tJdF計算
                    command_queue.enqueueNDRangeKernel(kernels[9], NULL, global_item_size1, local_item_size1, NULL, NULL);
                    command_queue.finish();
                }
                //chi2(old)計算その2
                command_queue.enqueueNDRangeKernel(kernels[14], NULL, global_item_size11_2, local_item_size1, NULL, NULL);
                command_queue.finish();
				command_queue.enqueueReadBuffer(chi2_buff, CL_TRUE, 0, sizeof(float), &chi2_old);
                /*float * prj_mt_fit;
                 prj_mt_fit = new float [g_px*g_pa];
                 cl::size_t<3> origin1;
                 cl::size_t<3> region1;
                 origin1[0] = 0;
                 origin1[1] = 0;
                 region1[0] = g_px;
                 region1[1] = g_pa;
                 region1[2] = 1;
                 for(int E=0;E<num_energy;E++){
                 origin1[2] = E;
                 command_queue.enqueueReadImage(prj_mt_fit_img, CL_TRUE, origin1, region1, 0, 0, prj_mt_fit);
                 string fileName_output= inp.getFittingOutputDir()+ "/"+numTagString(E, "prjmtfit", ".raw", 3);
                 outputRawFile_stream(fileName_output,prj_mt_fit,g_px*g_pa);
                 }
                 cout << "finish" << endl;
                 delete[] prj_mt_fit;*/
                //delete[] mt_fit;
                /*float *bprj_delta;
                 bprj_delta = new float[IMAGE_SIZE_M];
                 for(int E=0;E<num_energy;E++){
                 //kernels[3].setArg(5, (cl_int)E);
                 command_queue.enqueueReadBuffer(delta_mt_buff, CL_TRUE, sizeof(float)*IMAGE_SIZE_M*E, sizeof(float)*IMAGE_SIZE_M, bprj_delta);
                 string fileName_output= inp.getFittingOutputDir()+ "/"+numTagString(E, "bprj", ".raw", 3);
                 outputRawFile_stream(fileName_output,bprj_delta,IMAGE_SIZE_M);
                 }
                 cout << "finish" << endl;
                 delete[] bprj_delta;*/
                /*float *bprj_chi2_old;
                 bprj_chi2_old = new float[IMAGE_SIZE_M];
                 command_queue.enqueueReadBuffer(bprj_chi2_old_buff, CL_TRUE, 0, sizeof(float)*IMAGE_SIZE_M, bprj_chi2_old);
                 string fileName_outputO= inp.getFittingOutputDir()+ "/chi2_old.raw";
                 outputRawFile_stream(fileName_outputO,bprj_chi2_old,IMAGE_SIZE_M);
                 cout << "finish" << endl;
                 delete [] bprj_chi2_old;*/
                
                
                /*float *tJdF;
                 tJdF = new float[IMAGE_SIZE_M];
                 for (int i=0; i<PARAM_SIZE; i++) {
                 command_queue.enqueueReadBuffer(tJdF_img, CL_TRUE, sizeof(float)*IMAGE_SIZE_M*i, sizeof(float)*IMAGE_SIZE_M, tJdF);
                 string fileName_output= inp.getFittingOutputDir()+ "/"+numTagString(i, "tJdF", ".raw", 2);;
                 outputRawFile_stream(fileName_output,tJdF,IMAGE_SIZE_M);
                 }
                 cout << "finish" << endl;
                 delete [] tJdF;*/
                /*float *tJJ;
                 tJJ = new float[IMAGE_SIZE_M];
                 for (int i=0; i<PARAM_SIZE*PARAM_SIZE; i++) {
                 command_queue.enqueueReadBuffer(tJJ_img, CL_TRUE, sizeof(float)*IMAGE_SIZE_M*i, sizeof(float)*IMAGE_SIZE_M, tJJ);
                 string fileName_output= inp.getFittingOutputDir()+ "/"+numTagString(i, "tJJ", ".raw", 2);;
                 outputRawFile_stream(fileName_output,tJJ,IMAGE_SIZE_M);
                 }
                 cout << "finish" << endl;
                 delete [] tJJ;*/
                
                
                //新パラメータ候補の計算
#ifdef DEBUG
                cout<<"     GPU("<<thread_id+1<<"): calculate new parameter candidate"<<endl;
#endif
                kernels[10].setArg(3, lambda);
                command_queue.enqueueCopyBuffer(results_img, results_cnd_img, 0, 0, sizeof(cl_float)*IMAGE_SIZE_M*paramsize);
                command_queue.enqueueNDRangeKernel(kernels[10], NULL, global_item_size1, local_item_size1, NULL, NULL);
                command_queue.finish();
                /*float *invtJJ;
                 invtJJ = new float[IMAGE_SIZE_M];
                 for (int i=0; i<PARAM_SIZE*PARAM_SIZE; i++) {
                 command_queue.enqueueReadBuffer(tJJ_img, CL_TRUE, sizeof(float)*IMAGE_SIZE_M*i, sizeof(float)*IMAGE_SIZE_M, invtJJ);
                 string fileName_outputInv= inp.getFittingOutputDir()+ "/"+numTagString(i, "invtJJ", ".raw", 2);;
                 outputRawFile_stream(fileName_outputInv,invtJJ,IMAGE_SIZE_M);
                 }
                 cout << "finish" << endl;
                 delete [] invtJJ;*/
			
                
                
                //拘束条件の適用
#ifdef DEBUG
                cout<<"     GPU("<<thread_id+1<<"): set constrain"<<endl;
#endif
                command_queue.enqueueNDRangeKernel(kernels[11], NULL, global_item_size1, local_item_size1, NULL, NULL);
                command_queue.finish();
                
                
                //dLの計算
#ifdef DEBUG
                cout<<"     GPU("<<thread_id+1<<"): calc dL"<<endl;
#endif
				/*float *dL_p;
				dL_p = new float[IMAGE_SIZE_M];
				command_queue.enqueueReadBuffer(dL_buff, CL_TRUE, 0, sizeof(float)*IMAGE_SIZE_M, dL_p);
				string fileName_outputL = inp.getFittingOutputDir() + "/dL.raw";
				outputRawFile_stream(fileName_outputL, dL_p, IMAGE_SIZE_M);
				cout << "finish" << endl;*/
				command_queue.enqueueNDRangeKernel(kernels[11], NULL, global_item_size11_1, local_item_size1, NULL, NULL);
                command_queue.finish();
				/*command_queue.enqueueReadBuffer(dL_buff, CL_TRUE, 0, sizeof(float)*IMAGE_SIZE_M, dL_p);
				fileName_outputL = inp.getFittingOutputDir() + "/dLX.raw";
				outputRawFile_stream(fileName_outputL, dL_p, IMAGE_SIZE_M);
				cout << "finish" << endl;*/
                command_queue.enqueueNDRangeKernel(kernels[11], NULL, global_item_size11_2, local_item_size1, NULL, NULL);
                command_queue.finish();
				/*command_queue.enqueueReadBuffer(dL_buff, CL_TRUE, 0, sizeof(float)*IMAGE_SIZE_M, dL_p);
				fileName_outputL = inp.getFittingOutputDir() + "/dLXY.raw";
				outputRawFile_stream(fileName_outputL, dL_p, IMAGE_SIZE_M);
				cout << "finish" << endl;
				delete[] dL_p;*/
                
                
                //新パラメータ候補の評価
#ifdef DEBUG
                cout<<"     GPU("<<thread_id+1<<"): evaluate chi2(new)"<<endl;
#endif
                command_queue.enqueueFillBuffer(chi2_buff, (cl_float)0.0f, 0, sizeof(cl_float)*g_px*g_pa/g_ss,NULL,NULL);
                kernels[3].setArg(0, results_cnd_img);
                //kernels[15].setArg(0, chi2_new_buff);
                //kernels[15].setArg(1, prj_delta_mt_img);
                //kernels[14].setArg(0, chi2_new_buff);
                for (int i = fittingStartEnergyNo; i<=fittingEndEnergyNo; i++) {
                    //パラメータをフィッティング式に代入
                    kernels[3].setArg(3, i-fittingStartEnergyNo);
                    command_queue.enqueueNDRangeKernel(kernels[3], NULL, global_item_size1, local_item_size1, NULL, NULL);
                    command_queue.finish();
                    
                    //delta mtへの投影計算
                    kernels[4].setArg(5, i-fittingStartEnergyNo);
                    command_queue.enqueueNDRangeKernel(kernels[4], NULL, global_item_size4, local_item_size1, NULL, NULL);
                    command_queue.finish();
                    
                    //chi2(new)計算その1
                    command_queue.enqueueNDRangeKernel(kernels[8], NULL, global_item_size4, local_item_size1, NULL, NULL);
                    command_queue.finish();
                }
                //chi2(new)計算その2
                command_queue.enqueueNDRangeKernel(kernels[14], NULL, global_item_size11_2, local_item_size1, NULL, NULL);
                command_queue.finish();
                /*float *bprj_chi2_new;
                 bprj_chi2_new = new float[IMAGE_SIZE_M];
                 command_queue.enqueueReadBuffer(bprj_chi2_new_buff, CL_TRUE, 0, sizeof(float)*IMAGE_SIZE_M, bprj_chi2_new);
                 string fileName_outputN= inp.getFittingOutputDir()+ "/chi2_new.raw";
                 outputRawFile_stream(fileName_outputN,bprj_chi2_new,IMAGE_SIZE_M);
                 cout << "finish" << endl;
                 delete[] bprj_chi2_new;*/
                
                
                //パラメータの更新
#ifdef DEBUG
                cout<<"     GPU("<<thread_id+1<<"): update paramater"<<endl;
#endif
                
                command_queue.enqueueReadBuffer(chi2_buff, CL_TRUE, 0, sizeof(float), &chi2_new);
                command_queue.enqueueReadBuffer(dL_buff, CL_TRUE, 0, sizeof(float), &dL);
                float rho = (chi2_old-chi2_new)/dL;
                cout << "       rho=("<<chi2_old<<"-"<<chi2_new<<")/(" << dL << ")="<<rho<<endl;
                if(rho>=0.0f){
                    float l_A=(2.0f*rho-1.0f);
                    l_A = 1.0f-l_A*l_A*l_A;
                    l_A = min(2.0f,max(0.333f,l_A))*lambda;
                    lambda = l_A;
                    nyu=2.0f;
                    command_queue.enqueueCopyBuffer(results_cnd_img, results_img, 0, 0, sizeof(cl_float)*IMAGE_SIZE_M*paramsize);
#ifdef DEBUG
                    cout<<"        updated"<<endl;
                    cout << "       next lambda="<<lambda<<endl;
#endif
                }else{
                    lambda=nyu*lambda;
                    nyu=2.0f*nyu;
#ifdef DEBUG
                    cout<<"        not updated"<<endl;
                    cout << "       next lambda="<<lambda<<endl;
#endif
                }
            }
        }
        
        
        //read results image from buffer
        for (int i = 0; i<paramsize; i++) {
            command_queue.enqueueReadBuffer(results_img, CL_TRUE, sizeof(cl_float)*IMAGE_SIZE_M*i, sizeof(cl_float)*IMAGE_SIZE_M, results_vec[i]);
        }
        
        //output thread
        output_th_fit[thread_id].join();
        output_th_fit[thread_id]=thread(reconstFitResult_output_thread, fiteq,Z,inp,move(results_vec));
        
    } catch (cl::Error ret) {
        cerr << "ERROR: " << ret.what() << "(" << ret.err() << ")" << endl;
    }
    
    return 0;
}



int data_input_thread(cl::CommandQueue command_queue, vector<cl::Kernel> kernels,
                      fitting_eq fiteq, int Z, int thread_id, input_parameter inp,
                      cl::Buffer energy_buff,cl::Buffer angle_buff,
                      cl::Buffer C_matrix_buff, cl::Buffer D_vector_buff,
                      cl::Buffer freeFix_buff,cl::Buffer attenuator_buff,int *sub){
    
    
    int startEnergyNo = inp.getFittingStartEnergyNo();
    int endEnergyNo = inp.getFittingEndEnergyNo();
    string fileName_base = inp.getFittingFileBase();
    string input_dir=inp.getInputDir();
    vector<float*> prj_mt_vec;
    //vector<vector<string>> filepath_input;
    vector<string> filepath_input;
    for (int i=startEnergyNo; i<=endEnergyNo; i++) {
        /*vector<string> filepath_input_atE;
        for (int j=1; j<=g_pa; j++) {
            filepath_input_atE.push_back(input_dir + EnumTagString(i,"/",fileName_base) + AnumTagString(j,"",".raw"));
        }
        filepath_input.push_back(filepath_input_atE);*/
        filepath_input.push_back(input_dir + EnumTagString(i,"/",fileName_base) + AnumTagString(Z,"",".raw"));
        prj_mt_vec.push_back(new float[g_px*g_pa]);
    }
    
    
    //input mt data
    m1.lock();
    int num_energy = endEnergyNo -startEnergyNo +1;
    for (int i=0; i<num_energy; i++) {
		//cout << "E: " << i << endl;
		/*for (int j=0; j<g_pa; j++) {
            int64_t offset = (int64_t)Z*IMAGE_SIZE_X*sizeof(float);
            int64_t size = (int64_t)IMAGE_SIZE_X*sizeof(float);
            readRawFile_offset(filepath_input[i][j],&prj_mt_vec[i][j*IMAGE_SIZE_X],offset,size);
        }*/
        readRawFile_offset(filepath_input[i],prj_mt_vec[i],0,g_px*g_pa*sizeof(float));
    }
    m1.unlock();
    
    fitting_th[thread_id].join();
    switch (g_mode){
        case 1: //add (加算型) algebraic reconstruction technique 法
            cout << "Processing by AART method" << endl << endl;
            fitting_th[thread_id] = thread(XAFSreconstFit_AART_thread,
                                           command_queue, kernels, fiteq, Z, thread_id, inp,
                                           energy_buff, angle_buff, C_matrix_buff, D_vector_buff,
                                           freeFix_buff, attenuator_buff, move(prj_mt_vec), sub);
            break;
        case 2: //multiply (乗算型) algebraic reconstruction technique 法
            //OS-EMのg_ss=g_paと同等
            //OS-EMのg_ss=1と同等
            cout << "Processing by MART method" << endl << endl;
            g_ss=g_pa;
            /*fitting_th[thread_id] = thread(XAFSreconstFit_OSEM_thread,
                                       command_queue, kernels, fiteq, Z, thread_id, inp,
                                       energy_buff, angle_buff, C_matrix_buff, D_vector_buff,
                                       freeFix_buff, attenuator_buff, move(prj_mt_vec), sub);*/
            break;
        case 3: //add (加算型) simultaneous reconstruction technique 法
            //AARTのg_ss=1と同等?
            cout << "Processing by ASIRT method" << endl << endl;
            g_ss=1;
            fitting_th[thread_id] = thread(XAFSreconstFit_AART_thread,
                                           command_queue, kernels, fiteq, Z, thread_id, inp,
                                           energy_buff, angle_buff, C_matrix_buff, D_vector_buff,
                                           freeFix_buff, attenuator_buff, move(prj_mt_vec), sub);
            break;
        case 4: //multiply(乗算型) simultaneous reconstruction technique 法
            //ML-EMと同等?
            cout << "Processing by MSIRT method" << endl << endl;
            g_ss=1;
            /*fitting_th[thread_id] = thread(XAFSreconstFit_OSEM_thread,
                                       command_queue, kernels, fiteq, Z, thread_id, inp,
                                       energy_buff, angle_buff, C_matrix_buff, D_vector_buff,
                                       freeFix_buff, attenuator_buff, move(prj_mt_vec), sub);*/
            break;
        case 5: //maximum likelihood-expection maximumization (ML-EM)法
            cout << "Processing by ML-EM method" << endl << endl;
            //OS-EMのg_ss=1と同等
            g_ss=1;
            /*fitting_th[thread_id] = thread(XAFSreconstFit_OSEM_thread,
                                           command_queue, kernels, fiteq, Z, thread_id, inp,
                                           energy_buff, angle_buff, C_matrix_buff, D_vector_buff,
                                           freeFix_buff, attenuator_buff, move(prj_mt_vec), sub);*/
            break;
        case 6:  //ordered subset EM (OS-EM)法
            cout << "Processing by OS-EM method" << endl << endl;
            /*fitting_th[thread_id] = thread(XAFSreconstFit_OSEM_thread,
                                       command_queue, kernels, fiteq, Z, thread_id, inp,
                                       energy_buff, angle_buff, C_matrix_buff, D_vector_buff,
                                       freeFix_buff, attenuator_buff, move(prj_mt_vec), sub);*/
            break;
        case 7: //filter back-projection法
            //cout << "Processing by FBP method" << endl << endl;
            break;
        default:
        
            break;
    }
    
    
    return 0;
}




//dummy thread
static int dummy(){
    return 0;
}


int XAFSreconstRit_ocl(fitting_eq fiteq, input_parameter inp,
                  OCL_platform_device plat_dev_list)
{
    cl_int ret;
    float startEnergy=inp.getStartEnergy();
    float endEnergy=inp.getEndEnergy();
    float E0=inp.getE0();
    
    
    //create output dir
    for (int i=0; i<fiteq.ParaSize(); i++) {
        char buffer=fiteq.freefix_para()[i];
        if (atoi(&buffer)==1) {
            string fileName_output = inp.getFittingOutputDir() + "/"+fiteq.param_name(i);
            MKDIR(fileName_output.c_str());
        }
    }
    
    
    //energy file input & processing
    ifstream energy_ifs(inp.getEnergyFilePath(),ios::in);
    vector<float> energy;
    int i=0, startEnergyNo=0, endEnergyNo=0;
    do {
        float a;
        energy_ifs>>a;
        if (energy_ifs.eof()) break;
        cout<<i<<": "<<a;
        if ((a>=startEnergy)&(a<=endEnergy)) {
            energy.push_back(a-E0);
            cout<<" <- fitting range";
            endEnergyNo = i;
        } else if(a<startEnergy) {
            startEnergyNo = i+1;
        }
        cout<<endl;
        i++;
    } while (!energy_ifs.eof());
    int num_energy=endEnergyNo-startEnergyNo+1;
    inp.setFittingStartEnergyNo(startEnergyNo);
    inp.setFittingEndEnergyNo(endEnergyNo);
    ostringstream oss;
    oss << startEnergyNo<<"-"<<endEnergyNo;
    inp.setEnergyNoRange(oss.str());
    oss.flush();
    cout << "energy num for fitting: "<<num_energy<<endl<<endl;
    
    
    //angle list file読み込み
    g_ang = new float[(unsigned long)g_pa];
    read_log(g_f3, g_pa);
    
    
    // サブセットの順番を決定する
    int *sub;
    sub =  new int[g_ss];
    int k = 0;
    for (int i = 0; i < 32; i++) k += (g_ss >> i) & 1;
	if (k == 1){    //ssが2^nの場合
		int m1 = 0;
		sub[m1++] = 0;
		//cout << m1-1 << ":" << sub[m1-1] << endl;
		int i, m2;
		for (i = g_ss, m2 = 1; i > 1; i /= 2, m2 *= 2){
			for (int j = 0; j < m2; j++) {
				sub[m1++] = sub[j] + i / 2;
				//cout << m1-1 << ":" << sub[m1-1] << endl;
			}
		}
	}
	else {
	    for (int i = 0; i < g_ss; i++) {
		    sub[i] = i;
			//cout<<i<<":"<< sub[i] <<endl;
		}
	}
    
    
    //kernel program source
    ifstream ifs("E:/Dropbox/CTprogram/XAFSreconst_fitting/XAFSreconstFitting.cl", ios::in);
     if(!ifs) {
     cerr << "   Failed to load kernel \n" << endl;
     return -1;
     }
     istreambuf_iterator<char> it(ifs);
     istreambuf_iterator<char> last;
     string kernel_fit_src(it,last);
     ifs.close();
    
    
    //OpenCL Program
    vector<vector<cl::Kernel>> kernels;
    string kernel_code="";
    kernel_code += fiteq.preprocessor_str();
    //cout << fiteq.preprocessor_str();
    kernel_code += kernel_fit_src;
    //cout << kernel_code<<endl;
    size_t kernel_code_size = kernel_code.length();
    for (int i=0; i<plat_dev_list.contextsize(); i++) {
        
        cl::Program::Sources source(1,std::make_pair(kernel_code.c_str(),kernel_code_size));
        cl::Program program(plat_dev_list.context(i), source,&ret);
        //kernel build
        string option = "";
#ifdef DEBUG
        option += "-D DEBUG";
#endif
        option += kernel_preprocessor_nums(E0,num_energy,fiteq.ParaSize(),fiteq.constrain_size);
        option += kernel_reconst_preprocessor_nums();
        string GPUvendor =  plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0].getInfo<CL_DEVICE_VENDOR>();
        if(GPUvendor == "nvidia"){
            option += " -cl-nv-maxrregcount=64";
#ifdef DEBUG
            option += " -cl-nv-verbose -Werror";
#endif
        }
        else if (GPUvendor.find("NVIDIA Corporation")==0) {
            option += " -cl-nv-maxrregcount=64";
#ifdef DEBUG
            option += " -cl-nv-verbose -Werror";
#endif
        }
        else {
#ifdef DEBUG
            option += " -Werror";
#endif
        }
        ret = program.build(option.c_str());
#ifdef DEBUG
        string logstr=program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(plat_dev_list.context(i).getInfo<CL_CONTEXT_DEVICES>()[0]);
        cout << logstr << endl;
#endif
        //cout<<ret<<endl;
        vector<cl::Kernel> kernels_plat;
        kernels_plat.push_back(cl::Kernel::Kernel(program,"sinogramCorrection", &ret));//0
        kernels_plat.push_back(cl::Kernel::Kernel(program,"backProjectionArrayFull", &ret));//1
        kernels_plat.push_back(cl::Kernel::Kernel(program,"parameterMask", &ret));//2
        kernels_plat.push_back(cl::Kernel::Kernel(program,"assign2FittingEq", &ret));//3
        kernels_plat.push_back(cl::Kernel::Kernel(program,"projectionToDeltaMt", &ret));//4
        kernels_plat.push_back(cl::Kernel::Kernel(program,"projectionArray", &ret));//5
        kernels_plat.push_back(cl::Kernel::Kernel(program,"backProjectionSingle", &ret));//6
        kernels_plat.push_back(cl::Kernel::Kernel(program,"backProjectionArray", &ret));//7
        kernels_plat.push_back(cl::Kernel::Kernel(program,"calcChi2_1", &ret));//8
        kernels_plat.push_back(cl::Kernel::Kernel(program,"calc_tJJ_tJdF", &ret));//9
        kernels_plat.push_back(cl::Kernel::Kernel(program,"calc_pCandidate", &ret));//10
        kernels_plat.push_back(cl::Kernel::Kernel(program,"setConstrain", &ret));//11
        kernels_plat.push_back(cl::Kernel::Kernel(program,"calc_dL", &ret));//12
        kernels_plat.push_back(cl::Kernel::Kernel(program,"assign2Jacobian", &ret));//13
        kernels_plat.push_back(cl::Kernel::Kernel(program,"calcChi2_2", &ret));//14
        
        kernels_plat.push_back(cl::Kernel::Kernel(program,"assign2FittingEq_EArray", &ret));//15
        kernels_plat.push_back(cl::Kernel::Kernel(program,"assign2Jacobian_EArray", &ret));//16
        kernels_plat.push_back(cl::Kernel::Kernel(program,"projectionToDeltaMt_Earray", &ret));//17
        kernels_plat.push_back(cl::Kernel::Kernel(program,"calcChi2_1_EArray", &ret));//18
        kernels_plat.push_back(cl::Kernel::Kernel(program,"calc_tJJ_tJdF_EArray", &ret));//19
        kernels_plat.push_back(cl::Kernel::Kernel(program,"assign2Jacobian_2", &ret));//20

		kernels_plat.push_back(cl::Kernel::Kernel(program, "circleAttenuator", &ret));//21
        
        
        kernels.push_back(kernels_plat);
    }
    //cout << kernels[0][0].getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(plat_dev_list.dev(0, 0)) << endl;
    
    
    //display OCL device
    int t=0;
    for (int i=0; i<plat_dev_list.platsize(); i++) {
        for (int j=0; j<plat_dev_list.devsize(i); j++) {
            cout<<"Device No. "<<t+1<<endl;
            string platform_param;
            plat_dev_list.plat(i).getInfo(CL_PLATFORM_NAME, &platform_param);
            cout << "CL PLATFORM NAME: "<< platform_param<<endl;
            plat_dev_list.plat(i).getInfo(CL_PLATFORM_VERSION, &platform_param);
            cout << "   "<<platform_param<<endl;
            string device_pram;
            plat_dev_list.dev(i,j).getInfo(CL_DEVICE_NAME, &device_pram);
            cout << "CL DEVICE NAME: " << device_pram << endl << endl;
            t++;
        }
    }
    
    
    //energy_buffers create (per context)
    vector<cl::Buffer> energy_buffers;
    vector<cl::Buffer> angle_buffers;
    vector<cl::Buffer> C_matrix_buffers;
    vector<cl::Buffer> D_vector_buffers;
    vector<cl::Buffer> freeFix_buffers;
    vector<cl::Buffer> attenuator_buffers;
    int paramsize = (int)fiteq.ParaSize();
    int contrainsize = (int)fiteq.constrain_size;
    for (int i=0; i<plat_dev_list.contextsize(); i++){
        energy_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*num_energy, 0, NULL));
        angle_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*g_pa, 0, NULL));
        C_matrix_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*paramsize*contrainsize, 0, NULL));
        D_vector_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*contrainsize, 0, NULL));
        freeFix_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_char)*paramsize, 0, NULL));
		attenuator_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*paramsize, 0, NULL));


        //GPUへアップロード
        plat_dev_list.queue(i,0).enqueueWriteBuffer(energy_buffers[i], CL_TRUE, 0, sizeof(cl_float)*num_energy, &energy[0], NULL, NULL);
        plat_dev_list.queue(i,0).enqueueWriteBuffer(angle_buffers[i], CL_TRUE, 0, sizeof(cl_float)*g_pa, &g_ang[0], NULL, NULL);
        for (int j = 0; j < contrainsize; j++) {
            plat_dev_list.queue(i, 0).enqueueWriteBuffer(C_matrix_buffers[i], CL_TRUE, sizeof(cl_float)*j*paramsize, sizeof(cl_float)*paramsize, &(fiteq.C_matrix[j][0]), NULL, NULL);
        }
        plat_dev_list.queue(i, 0).enqueueWriteBuffer(D_vector_buffers[i], CL_TRUE, 0, sizeof(cl_float)*contrainsize, &fiteq.D_vector[0], NULL, NULL);
        plat_dev_list.queue(i,0).enqueueWriteBuffer(freeFix_buffers[i], CL_TRUE, 0, sizeof(cl_char)*paramsize, fiteq.freefix_para(), NULL, NULL);
        plat_dev_list.queue(i,0).enqueueWriteBuffer(attenuator_buffers[i], CL_TRUE, 0, sizeof(cl_float)*paramsize, fiteq.paraAttenuator(), NULL, NULL);
    }

    
    
    //start thread
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        input_th.push_back(thread(dummy));
        fitting_th.push_back(thread(dummy));
        output_th_fit.push_back(thread(dummy));
    }
    for (int i=g_st; i<min(g_st+g_num,IMAGE_SIZE_Y);) {
        for (int j=0; j<plat_dev_list.contextsize(); j++) {
            if (input_th[j].joinable()) {
                input_th[j].join();
                input_th[j] = thread(data_input_thread,
                                     plat_dev_list.queue(j, 0),kernels[j],
                                     fiteq,i,j,inp,
                                     energy_buffers[j], angle_buffers[j],
                                     C_matrix_buffers[j], D_vector_buffers[j],
                                     freeFix_buffers[j], attenuator_buffers[j],sub);
                i++;
                if (i >= IMAGE_SIZE_Y) break;
            } else{
                this_thread::sleep_for(chrono::milliseconds(100));
            }
        }
        if (i >= IMAGE_SIZE_Y) break;
    }
    
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        input_th[j].join();
    }
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        fitting_th[j].join();
    }
    for (int j = 0; j<plat_dev_list.contextsize(); j++) {
        output_th_fit[j].join();
    }
    
    
    return 0;
}
