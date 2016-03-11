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

int OCL_platform_device::dev_num(int list_num){
    return device_num[list_num];
}

size_t OCL_platform_device::size(){
    return platform_num.size();
}

cl::Platform OCL_platform_device::plat(int list_num){
    return p_ids[list_num];
}

cl::Device OCL_platform_device::dev(int list_num){
    return d_ids[list_num];
}

cl::Context OCL_platform_device::context(int list_num){
    return contexts[list_num];
}

cl::CommandQueue OCL_platform_device::queue(int list_num){
    return queues[list_num];
}

OCL_platform_device::OCL_platform_device(string plat_dev_str){
    
    vector<cl::Platform> platform_ids;
    cl::Platform::get(&platform_ids);
    vector<vector<cl::Device>> device_ids;
    vector<vector<int>> ocl_plat_dev;
    
    size_t dialog = plat_dev_str.length();
    if (dialog==0) {
        cout << "Select OpenCL device (ex. 1,2,3-5)...\n";
    }
    
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
        for (unsigned int j=0; j<device_ids_atPlat.size(); j++) {
            device_ids_atPlat[j].getInfo(CL_DEVICE_NAME, &device_pram);
            if (dialog==0) {
                cout << "        " << platdevice_num << ": " << device_pram << "\n";
            }
            
            vector<int> plat_dev_ij;
            plat_dev_ij.push_back(i);
            plat_dev_ij.push_back(j);
            
            ocl_plat_dev.push_back(plat_dev_ij);
            platdevice_num++;
        }
    }
    
    int select_devNum=1;
    vector<OCL_platform_device> plat_dev_list_dum;
    if (dialog==0) {
        cin>>plat_dev_str;
    }
    int t=0;
    do {
        istringstream iss0(plat_dev_str);
        if (plat_dev_str.length()==0) {
            break;
        }else if (plat_dev_str.at(0)==',') {
            plat_dev_str.erase(0, 1);
        } else if (plat_dev_str.at(0)=='-'){
            plat_dev_str.erase(0, 1);
            istringstream iss(plat_dev_str);
            plat_dev_str.erase();
            int end_devNum;
            iss >> end_devNum >> plat_dev_str;
            for (int i=select_devNum+1; i<=end_devNum; i++) {
                if(i<=ocl_plat_dev.size()) {
                    platform_num.push_back(ocl_plat_dev[i-1][0]);
                    device_num.push_back(ocl_plat_dev[i-1][1]);
                    p_ids.push_back(platform_ids[platform_num[t]]);
                    d_ids.push_back(device_ids[platform_num[t]][device_num[t]]);
                    cl_context_properties properties[]={CL_CONTEXT_PLATFORM, (cl_context_properties)(p_ids[t])(), 0};
                    contexts.push_back(cl::Context(d_ids[t], properties,NULL,NULL,NULL));
                    queues.push_back(cl::CommandQueue(contexts[t], d_ids[t], 0, NULL));
                    t++;
                }
            }
        } else if( iss0 >> select_devNum){
            istringstream iss(plat_dev_str);
            plat_dev_str.erase();
            iss >> select_devNum >> plat_dev_str;
            if(select_devNum<=ocl_plat_dev.size() && select_devNum>0) {
                platform_num.push_back(ocl_plat_dev[select_devNum-1][0]);
                device_num.push_back(ocl_plat_dev[select_devNum-1][1]);
                p_ids.push_back(platform_ids[platform_num[t]]);
                d_ids.push_back(device_ids[platform_num[t]][device_num[t]]);
                cl_context_properties properties[]={CL_CONTEXT_PLATFORM, (cl_context_properties)(p_ids[t])(), 0};
                contexts.push_back(cl::Context(d_ids[t], properties,NULL,NULL,NULL));
                queues.push_back(cl::CommandQueue(contexts[t], d_ids[t], 0, NULL));
                t++;
            }
        }
    } while (plat_dev_str.length()>0);
    
}
