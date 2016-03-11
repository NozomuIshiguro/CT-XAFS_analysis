//
//  Reslice_CTreconst_ocl.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2016/03/10.
//  Copyright © 2016年 Nozomu Ishiguro. All rights reserved.
//

#include "reslice.hpp"
#include "CT_reconstruction.hpp"

extern string base_f2;
extern string tale_f2;

extern vector<thread> input_th, output_th, imageReg_th, reconst_th, reslice_th, output_th1, output_th2;

extern int g_ss;

//dummy thread
static int dummy() {
	return 0;
}

int reslice_CTreconst_execute(OCL_platform_device plat_dev_list,
                              vector<cl::Kernel>kernels_reslice,
                              vector<vector<cl::Kernel>>kernels_reconst,
                              vector<cl::Buffer> angle_buffers,int *sub,
                              input_parameter inp,string subDir_str,string fileName_base_i,
                              vector<int>dN){
    
    int num_angle=inp.getEndAngleNo()-inp.getStartAngleNo()+1;
    
    int startAngleNo = inp.getStartAngleNo();
    int endAngleNo   = inp.getEndAngleNo();
    
    //input raw mt/fit files
    cout << "Input data of " << subDir_str << "...";
    vector<float*> mt_img_vec;
    for (int i=0; i<num_angle; i++) {
        mt_img_vec.push_back(new float[IMAGE_SIZE_M]);
    }
    //input mt data
    string input_dir = inp.getInputDir();
    for (int i=startAngleNo; i<=endAngleNo; i++) {
        string filepath_input = input_dir+"/"+subDir_str+fileName_base_i+AnumTagString(i,"", ".raw");
        //cout<<filepath_input;
        readRawFile(filepath_input,mt_img_vec[i-startAngleNo]);
    }
    /*for (int i = 0; i < num_angle; i++) {
     string fileName_output = inp.getOutputDir() + "/" + AnumTagString(i + 1, inp.getOutputFileBase(), ".raw");
     outputRawFile_stream(fileName_output, mt_img_vec[i], IMAGE_SIZE_M);
     }*/
    
    
    
    
    //reslice
	//plat_dev_list.queue(0, 0).finish();
    cout << "Reslicing data of " << subDir_str << "...";
    vector<float*> prj_img_vec;
    for (int i=0; i<IMAGE_SIZE_Y; i++) {
        prj_img_vec.push_back(new float[IMAGE_SIZE_X*num_angle]);
    }
    reslice_mtImg(plat_dev_list,kernels_reslice[0],mt_img_vec,prj_img_vec,inp);
    //delete input mt_vec
    for (int i=0; i<num_angle; i++) {
        delete [] mt_img_vec[i];
    }
    /*for (int i = 0; i < IMAGE_SIZE_Y; i++) {
     string fileName_output = inp.getOutputDir() + "/" + AnumTagString(i + 1, inp.getOutputFileBase(), ".raw");
     outputRawFile_stream(fileName_output, prj_img_vec[i], (size_t)(IMAGE_SIZE_X*num_angle));
     }*/
    
    


	//reconstruction
	g_d2 = inp.getOutputDir() + "/" + subDir_str;
	MKDIR(g_d2.c_str());
    //reconstruction
    cout << "Reconstructing CT images of "<<subDir_str<<"..."<<endl << endl;
    for (int N = g_st-1; N < g_st-1 + g_num;) {
        for (int j=0; j<plat_dev_list.contextsize(); j++) {
            if (reconst_th[j].joinable()) {
				reconst_th[j].join();
                int startN = N;
                int endN = min(N+dN[j],g_st-2 + g_num);
                vector<float*> reconst_img_vec;
                for (int i=0; i<endN-startN+1; i++) {
                    reconst_img_vec.push_back(new float[IMAGE_SIZE_M]);
					first_image(g_f4, reconst_img_vec[i], g_nx*g_nx);
                }
                switch (g_mode){
                    case 1: //add (加算型) algebraic reconstruction technique 法
                        //AART(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_it, N, g_wt1);
                        break;
                    case 2: //multiply (乗算型) algebraic reconstruction technique 法
                        //OS-EMのg_ss=g_paと同等
						reconst_th[j] = thread(OSEM_thread,plat_dev_list.queue(j,0),kernels_reconst[j],angle_buffers[j],sub,move(reconst_img_vec), prj_img_vec,startN, endN, g_it, j);
                        break;
                    case 3: //add (加算型) simultaneous reconstruction technique 法
                        //ASIRT(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_it, N, g_wt2);
                        break;
                    case 4: //multiply(乗算型) simultaneous reconstruction technique 法
                        //MSIRT(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_it, N);
                        break;
                    case 5: //maximum likelihood-expection maximumization (ML-EM)法
                        //OS-EMのg_ss=1と同等
						reconst_th[j] = thread(OSEM_thread,plat_dev_list.queue(j,0),kernels_reconst[j],angle_buffers[j],sub,move(reconst_img_vec), prj_img_vec,startN, endN, g_it, j);
                        break;
                    case 6:  //ordered subset EM (OS-EM)法
						reconst_th[j] = thread(OSEM_thread,plat_dev_list.queue(j,0),kernels_reconst[j],angle_buffers[j],sub,move(reconst_img_vec), prj_img_vec,startN, endN, g_it, j);
                        break;
                    case 7: //filter back-projection法
						reconst_th[j] = thread(FBP_thread,plat_dev_list.queue(j,0),kernels_reconst[j],angle_buffers[j],move(reconst_img_vec),prj_img_vec,startN, endN, j);
                        break;
                    default:
                        break;
                }
                N+=dN[j];
                if (N >= g_st-1 + g_num) break;
                
            } else this_thread::sleep_for(chrono::seconds(1));
        }
    }
    /*for (int i=0; i<IMAGE_SIZE_Y; i++) {
        delete [] prj_img_vec[i];
    }*/
    
    return 0;
}


int reslice_CTreconst_ocl(input_parameter inp,string fileName_base, float *ang){
    
    OCL_platform_device plat_dev_list(g_devList,true);
    

    base_f2=inp.getOutputFileBase();
    tale_f2=".raw";
    
    //program build (mt conversion, reslice)
    vector<cl::Kernel> kernels_reslice;
    reslice_programBuild(plat_dev_list.context(0),&kernels_reslice,inp.getStartAngleNo(),inp.getEndAngleNo());//reslice:2,xprojection:3,zcorrection:4
    //0:reslice, 1:xProjection, 2:zCorrection
    
    //program build (reconst)
    vector<vector<cl::Kernel>> kernels_reconst;
    vector<cl::Buffer> angle_buffers;
    for (int i = 0; i < plat_dev_list.contextsize(); i++) {
        vector<cl::Kernel> kernels_plat;
        switch (g_mode){
            case 1: //add (加算型) algebraic reconstruction technique 法
                //AART(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_it, N, g_wt1);
                break;
            case 2: //multiply (乗算型) algebraic reconstruction technique 法
                OSEM_programBuild(plat_dev_list.context(i), &kernels_plat);//OSEM1:0,OSEM2:1,:2
                break;
            case 3: //add (加算型) simultaneous reconstruction technique 法
                //ASIRT(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_it, N, g_wt2);
                break;
            case 4: //multiply(乗算型) simultaneous reconstruction technique 法
                //MSIRT(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_it, N);
                break;
            case 5: //maximum likelihood-expection maximumization (ML-EM)法
                OSEM_programBuild(plat_dev_list.context(i), &kernels_plat);//OSEM1:0,OSEM2:1,:2
                break;
            case 6:  //ordered subset EM (OS-EM)法
                OSEM_programBuild(plat_dev_list.context(i), &kernels_plat);//OSEM1:0,OSEM2:1,:2
                break;
            case 7: //filter back-projection法
                FBP_programBuild(plat_dev_list.context(i), &kernels_plat);
                break;
            default:
                break;
        }
        kernels_reconst.push_back(kernels_plat);
        
        angle_buffers.push_back(cl::Buffer(plat_dev_list.context(i), CL_MEM_READ_ONLY, sizeof(cl_float)*g_pa, 0, NULL));
        plat_dev_list.queue(i, 0).enqueueWriteBuffer(angle_buffers[i], CL_TRUE, 0, sizeof(cl_float)*g_pa, ang, NULL, NULL);

    }
    
    
    
    //display OCL device
    int t=0;
    vector<int> dN;
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
            cout << "CL DEVICE NAME: "<< device_pram<<endl;
            
            //working compute unit
            size_t device_pram_size[3]={0,0,0};
            plat_dev_list.dev(i,j).getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &device_pram_size);
            dN.push_back(min(numParallel, (int)device_pram_size[0]));
            cout<<"Number of working compute unit: "<<dN[t]<<"/"<<device_pram_size[0]<<endl<<endl;
            t++;
        }
    }

	int ss = 0;
	switch (g_mode) {
	case 1: //add (加算型) algebraic reconstruction technique 法
		break;
	case 2: //multiply (乗算型) algebraic reconstruction technique 法												  
		cout << "Processing by M-ART method" << endl << endl;
		ss = g_pa; //OS-EMのg_ss=g_paと同等
		break;
	case 3: //add (加算型) simultaneous reconstruction technique 法
			//ASIRT(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_it, N, g_wt2);
		break;
	case 4: //multiply(乗算型) simultaneous reconstruction technique 法
			//MSIRT(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_it, N);
		break;
	case 5: //maximum likelihood-expection maximumization (ML-EM)法
		cout << "Processing by ML-EM method" << endl << endl;													  
		ss = 1; //OS-EMのg_ss=1と同等
		break;
	case 6:  //ordered subset EM (OS-EM)法
		cout << "Processing by OS-EM method" << endl << endl;
		ss = g_ss;
		break;
	case 7: //filter back-projection法
		cout << "Processing by FBP method" << endl << endl;
		break;
	default:
		break;
	}
    
    
    // サブセットの順番を決定する
    int *sub=nullptr;
    if(ss>0){
        sub =  new int[ss];
        int k = 0;
        for (int i = 0; i < 32; i++) k += (ss >> i) & 1;
        if (k == 1){    //ssが2^nの場合
            int m1 = 0;
            sub[m1++] = 0;
            int i, m2;
            for (i = ss, m2 = 1; i > 1; i /= 2, m2 *= 2){
                for (int j = 0; j < m2; j++) sub[m1++] = sub[j] + i / 2;
            }
        }
        else {
            for (int i = 0; i < ss; i++) {
                sub[i] = i;
                //cout<<i<<":"<< sub[i] <<endl;
            }
        }
    }
    
	//start thread
	for (int j = 0; j<plat_dev_list.contextsize(); j++) {
		reconst_th.push_back(thread(dummy));
		output_th.push_back(thread(dummy));
	}
    //mt E番号ごとに実行
    if ((inp.getStartEnergyNo()>0)&(inp.getEndEnergyNo()>0)){
        int startEnergyNo=inp.getStartEnergyNo();
        int endEnergyNo=inp.getEndEnergyNo();
        for (int i=startEnergyNo; i<=endEnergyNo;i++) {
            reslice_CTreconst_execute(plat_dev_list,kernels_reslice,kernels_reconst,angle_buffers,sub,inp,EnumTagString(i,"",""),fileName_base,dN);
		}
    }
    // fitting 名前ごとに実行
    else if (inp.getFittingParaName().size()>0) {
        for (int i=0; i<inp.getFittingParaName().size();i++) {
            string paraname = inp.getFittingParaName()[i];
            reslice_CTreconst_execute(plat_dev_list,kernels_reslice,kernels_reconst,angle_buffers,sub,inp,paraname,fileName_base,dN);
		}
    }

	for (int j = 0; j<plat_dev_list.contextsize(); j++) {
		reconst_th[j].join();
	}
	for (int j = 0; j<plat_dev_list.contextsize(); j++) {
		output_th[j].join();
	}
    
    return 0;
}