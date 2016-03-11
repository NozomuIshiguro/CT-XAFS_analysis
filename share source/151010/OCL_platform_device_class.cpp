//
//  OCL_platform_device.cpp
//  CT-XANES_analysis
//
//  Created by Nozomu Ishiguro on 2015/03/14.
//  Copyright (c) 2015年 Nozomu Ishiguro. All rights reserved.
//

#include "OpenCL_analysis.hpp"

int OCL_platform_device::plat_num(int list_num){
    return platform_num[list_num];
}

int OCL_platform_device::dev_num(int plat_list_num, int dev_list_num){
    return device_num[plat_list_num][dev_list_num];
}

size_t OCL_platform_device::platsize(){
    return platform_num.size();
}

size_t OCL_platform_device::devsize(int plat_list_num){
    return device_num[plat_list_num].size();
}

size_t OCL_platform_device::contextsize(){
    return contexts.size();
}

size_t OCL_platform_device::queuesize(int context_list_num){
    return queues[context_list_num].size();
}

cl::Platform OCL_platform_device::plat(int list_num){
    return p_ids[list_num];
}

cl::Device OCL_platform_device::dev(int plat_list_num, int dev_list_num){
    return d_ids[plat_list_num][dev_list_num];
}

cl::Context OCL_platform_device::context(int list_num){
    return contexts[list_num];
}

cl::CommandQueue OCL_platform_device::queue(int context_list_num, int queue_list_num){
    if (nonshared) {
        return queues[context_list_num][0];
    }else{
        return queues[context_list_num][queue_list_num];
    }
}

OCL_platform_device::OCL_platform_device(string plat_dev_str,bool nonshared){
    
    vector<cl::Platform> platform_ids;
    cl::Platform::get(&platform_ids);
    vector<vector<cl::Device>> device_ids;
    vector<vector<int>> ocl_plat_dev;
    vector<vector<bool>> plat_dev_selected;
    
    OCL_platform_device::nonshared = nonshared;
    
    size_t dialog = plat_dev_str.length();
    if (dialog==0) {
        cout << "Select OpenCL device (ex. 1,2,3-5)...\n";
    }
    
    
    //display platform/device list
    int platdevice_num = 1;
    for (unsigned int i=0; i<platform_ids.size(); i++) {
        string platform_pram;
        platform_ids[i].getInfo(CL_PLATFORM_NAME, &platform_pram);
        if (dialog==0) {
            cout << "    OpenCL platform:" << platform_pram << "\n";
        }
        
        vector<cl::Device> device_ids_atPlat;
        platform_ids[i].getDevices(CL_DEVICE_TYPE_ALL, &device_ids_atPlat);
        device_ids.push_back(device_ids_atPlat);
        string device_pram;
        vector<bool> device_selected;
        for (unsigned int j=0; j<device_ids_atPlat.size(); j++) {
            device_ids_atPlat[j].getInfo(CL_DEVICE_NAME, &device_pram);
            if (dialog==0) {
                cout << "        " << platdevice_num << ": " << device_pram << "\n";
            }
            
            
            device_selected.push_back(false);
            
            vector<int> plat_dev_ij;
            plat_dev_ij.push_back(i); //[0]はプラットフォーム番号
            plat_dev_ij.push_back(j); //[1]はデバイス番号
            
            ocl_plat_dev.push_back(plat_dev_ij);
            platdevice_num++;
        }
        plat_dev_selected.push_back(device_selected);
    }
    //platform_ids[i] i:platform番号
    //device_id[i][j] i:platform番号, j:device番号
    //ocl_plat_dev[i][j] i:platform-device通し番号, j=0:platform番号, j=1:device番号
    //plat_dev_selected[i][j] i:platform番号, j:device番号
    
    //select devices
    int select_devNum=1;
    vector<OCL_platform_device> plat_dev_list_dum;
    if (dialog==0) {
        cin>>plat_dev_str;
    }
    do {
        istringstream iss0(plat_dev_str);
        //入力文字が残っていない→ループ脱出
        if (plat_dev_str.length()==0) {
            break;
        
        //入力の頭文字が','→次の文字に移る
        }else if (plat_dev_str.at(0)==',') {
            plat_dev_str.erase(0, 1);
            
        //入力の頭文字が'-'→start+1〜endのデバイスを加える
        } else if (plat_dev_str.at(0)=='-'){
            plat_dev_str.erase(0, 1);
            istringstream iss(plat_dev_str);
            plat_dev_str.erase();
            int end_devNum;
            iss >> end_devNum >> plat_dev_str;
            for (int i=select_devNum+1; i<=end_devNum; i++) {
                if(i<=ocl_plat_dev.size()) {
                    plat_dev_selected[ocl_plat_dev[i-1][0]][ocl_plat_dev[i-1][1]]=true;
                }
            }
            
        //入力の頭文字が数字→startとしてのデバイスを加える
        } else if( iss0 >> select_devNum){
            istringstream iss(plat_dev_str);
            plat_dev_str.erase();
            iss >> select_devNum >> plat_dev_str;
            if(select_devNum<=ocl_plat_dev.size() && select_devNum>0) {
                plat_dev_selected[ocl_plat_dev[select_devNum-1][0]][ocl_plat_dev[select_devNum-1][1]]=true;
            }
        }
    } while (plat_dev_str.length()>0);
    
    
    //context(platform毎)とcommand que(device毎)を作成 ←　nonshared == false
    //context(platform毎)とcommand que(device毎)を作成 ←　nonshared == true
    int t=0;
    for (int i=0; i<platform_ids.size(); i++) {
        vector<cl::Device> selected_device_ids;
        for (int j=0; j<device_ids[i].size(); j++) {
            if (plat_dev_selected[i][j]) {
                selected_device_ids.push_back(device_ids[i][j]);
            }
        }
        if (selected_device_ids.size()>0) {
            platform_num.push_back(i);
            p_ids.push_back(platform_ids[i]);
            cl_context_properties properties[]={CL_CONTEXT_PLATFORM, (cl_context_properties)(platform_ids[i])(), 0};
            
            vector<cl::Device> d_ids_AtPlat;
            vector<int> device_num_AtPlat;
            
            if (!nonshared) { //共有context
                contexts.push_back(cl::Context(selected_device_ids, properties,NULL,NULL,NULL));
                vector<cl::CommandQueue> queues_InCotext;
                for (int j=0; j<selected_device_ids.size(); j++) {
                    device_num_AtPlat.push_back(j);
                    d_ids_AtPlat.push_back(selected_device_ids[j]);
                    queues_InCotext.push_back(cl::CommandQueue(contexts[t],selected_device_ids[j], 0, NULL));
                }
                queues.push_back(queues_InCotext);
                t++;
            }else{ //非共有context
                for (int j=0; j<selected_device_ids.size(); j++) {
                    contexts.push_back(cl::Context(selected_device_ids, properties,NULL,NULL,NULL));
                    vector<cl::CommandQueue> queues_InCotext;
                    device_num_AtPlat.push_back(j);
                    d_ids_AtPlat.push_back(selected_device_ids[j]);
                    queues_InCotext.push_back(cl::CommandQueue(contexts[t],selected_device_ids[j], 0, NULL));
                    queues.push_back(queues_InCotext);
                    t++;
                }
            }
            
            device_num.push_back(device_num_AtPlat);
            d_ids.push_back(d_ids_AtPlat);
        }
    }
}
