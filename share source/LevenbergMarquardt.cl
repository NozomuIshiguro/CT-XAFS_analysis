#ifndef PARA_NUM
#define PARA_NUM 1
#endif

#ifndef PARA_NUM_SQ
#define PARA_NUM_SQ 1
#endif

#ifndef EPSILON
#define EPSILON 1.0f
#endif


//simultaneous linear equation
inline void sim_linear_eq(float *A, float *x, float *inv_A,
                                 size_t dim,__constant char *p_fix){
    
    float a = 0.0f;
    
    //inversed matrix
    for(int i=0; i<dim; i++){
        if(p_fix[i]== 48) continue;
        
        //devide (i,i) to 1
        a = 1/A[i+i*dim];
        
        for(int j=0; j<dim; j++){
            if(p_fix[j]== 48) continue;
            A[i+j*dim] *= a;
            inv_A[i+j*dim] *= a;
        }
        x[i] *= a;
        
        //erase (j,i) (i!=j) to 0
        for(int j=i+1; j<dim; j++){
            if(p_fix[j]== 48) continue;
            
            a = A[j+i*dim];
            for(int k=0; k<dim; k++){
                if(p_fix[k]== 48) continue;
                A[j+k*dim] -= a*A[i+k*dim];
                inv_A[j+k*dim] -= a*inv_A[i+k*dim];
            }
            x[j] -= a*x[i];
        }
    }
    for(int i=0; i<dim; i++){
        if(p_fix[i]== 48) continue;
        
        for(int j=i+1; j<dim; j++){
            if(p_fix[j]== 48) continue;
            
            a = A[i+j*dim];
            for(int k=0; k<dim; k++){
                if(p_fix[k]== 48) continue;
                A[k+j*dim] -= a*A[k+i*dim];
                inv_A[k+j*dim] -= a*inv_A[k+i*dim];
            }
            x[i] -= a*x[j];
        }
    }
}



__kernel void LevenbergMarquardt(__global float* tJdF_img, __global float* tJJ_img,
                                 __global float* dp_img, __global float* lambda_img,
                                 __constant char *p_fix,__global float* inv_tJJ_img){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t imageSizeX = get_global_size(0);
    const size_t imageSizeY = get_global_size(1);
    const size_t global_ID = global_x+global_y*imageSizeX;
    const size_t imageSizeM = imageSizeX*imageSizeY;
    
    float dp[PARA_NUM],lambda_diag_tJJ[PARA_NUM],tJdF[PARA_NUM];
    float tJJ[PARA_NUM_SQ],inv_tJJ[PARA_NUM_SQ];
    float lambda = lambda_img[global_ID];
    
    int num = 0;
    for(int i=0;i<PARA_NUM;i++){
        tJdF[i] = tJdF_img[global_ID + i*imageSizeM];
        dp[i] = tJdF[i];
        tJJ[i+i*PARA_NUM] = tJJ_img[global_ID+(PARA_NUM*i-(i-1)*i/2)*imageSizeM];
        lambda_diag_tJJ[i] = tJJ[i+i*PARA_NUM]*lambda;
        tJJ[i+i*PARA_NUM] *= (1.0f+lambda);
        inv_tJJ[i+i*PARA_NUM] = 1.0f;
        for(int j=i+1;j<PARA_NUM;j++){
            tJJ[i+j*PARA_NUM] = tJJ_img[global_ID+(PARA_NUM*i-(i+1)*i/2+j)*imageSizeM];
            tJJ[j+i*PARA_NUM] = tJJ[i+j*PARA_NUM];
            inv_tJJ[i+j*PARA_NUM] = 0.0f;
            inv_tJJ[j+i*PARA_NUM] = 0.0f;
        }
    }
    
    sim_linear_eq(tJJ,dp,inv_tJJ,PARA_NUM,p_fix);
    
    for(int i=0;i<PARA_NUM;i++){
        dp_img[global_ID+i*imageSizeM] = dp[i];
        inv_tJJ_img[global_ID+(PARA_NUM*i-(i-1)*i/2)*imageSizeM] = inv_tJJ[i+i*PARA_NUM];
        for(int j=i+1;j<PARA_NUM;j++){
            inv_tJJ_img[global_ID+(PARA_NUM*i-(i+1)*i/2+j)*imageSizeM] = inv_tJJ[i+j*PARA_NUM];
        }
    }
}



__kernel void estimate_dL(__global float* dp_img,
                          __global float* tJJ_img, __global float* tJdF_img,
                          __global float* lambda_img,__global float* dL_img){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t imageSizeX = get_global_size(0);
    const size_t imageSizeY = get_global_size(1);
    const size_t imageSizeM = imageSizeX*imageSizeY;
    const size_t global_ID = global_x+global_y*imageSizeX;
    
    float dp[PARA_NUM],lambda_diag_tJJ[PARA_NUM],tJdF[PARA_NUM];
    float lambda = lambda_img[global_ID];
    
    
    for(int i=0;i<PARA_NUM;i++){
        tJdF[i] = tJdF_img[global_ID + i*imageSizeM];
        dp[i] = dp_img[global_ID + i*imageSizeM];
        lambda_diag_tJJ[i] = tJJ_img[global_ID+(PARA_NUM*i-(i-1)*i/2)*imageSizeM]*lambda;
    }
    
    float d_L = 0.0f;
    for(int i=0;i<PARA_NUM;i++){
        d_L += dp[i]*(dp[i]*lambda_diag_tJJ[i] + tJdF[i]);
    }
    dL_img[global_ID] = d_L;
    
}


__kernel void updatePara(__global float* dp, __global float* fp, int z_id1, int z_id2){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t IDxy = global_x + global_y*size_x;
    const size_t IDxyz1 = IDxy + (global_z+z_id1)*size_x*size_y;
    const size_t IDxyz2 = IDxy + (global_z+z_id2)*size_x*size_y;
    
    float p = fp[IDxyz2]+dp[IDxyz1];
    fp[IDxyz2] = p;
}


__kernel void evaluateUpdateCandidate(__global float* tJdF_img, __global float* tJJ_img,
                                      __global float* lambda_img,__global float* nyu_img,
                                      __global float* dF2_old_img,__global float *dF2_new_img,
                                      __global float* dL_img, __global float* rho_img){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t imageSizeX = get_global_size(0);
    const size_t global_ID = global_x+global_y*imageSizeX;
    
    float d_L = dL_img[global_ID];
    float lambda = lambda_img[global_ID];
    
    float rho = (dF2_old_img[global_ID] - dF2_new_img[global_ID] + EPSILON)/d_L;
    float nyu = nyu_img[global_ID];
    
    float l_A = (2.0f*rho-1.0f);
    l_A = 1.0f-l_A*l_A*l_A;
    l_A = max(0.333f,l_A)*lambda;
    float l_B = nyu*lambda;
    
    lambda_img[global_ID] = (rho>=0.0f) ? l_A:l_B;
    nyu_img[global_ID] = (rho>=0.0f) ? 2.0f:2.0f*nyu;
    
    rho_img[global_ID] = rho;
}



__kernel void updateOrRestore(__global float* para_img, __global float* para_img_backup,
                              __global float* rho_img){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t imageSizeX = get_global_size(0);
    const size_t imageSizeY = get_global_size(1);
    const size_t IDxy = global_x+global_y*imageSizeX;
    const size_t IDxyz = IDxy + global_z*imageSizeX*imageSizeY;
    
    float rho = rho_img[IDxy];
    
    float para_A = para_img[IDxyz];
    float para_B = para_img_backup[IDxyz];
    
    
    para_img[IDxyz] = (rho>=0) ? para_A:para_B;
}


__kernel void updateOrHold(__global float* para_img, __global float* para_img_cnd,
                              __global float* rho_img){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t imageSizeX = get_global_size(0);
    const size_t imageSizeY = get_global_size(1);
    const size_t ID = global_x+global_y*imageSizeX;
    const size_t IDxyz = ID + global_z*imageSizeX*imageSizeY;
    
    float rho = rho_img[ID];
    
    float para_A = para_img_cnd[IDxyz];
    float para_B = para_img[IDxyz];
    
    
    para_img[IDxyz] = (rho>=0) ? para_A:para_B;
}



