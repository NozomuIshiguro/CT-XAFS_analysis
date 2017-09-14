#ifndef PARA_NUM
#define PARA_NUM 1
#endif

#ifndef PARA_NUM_SQ
#define PARA_NUM_SQ 1
#endif

#ifndef EPSILON
#define EPSILON 1.0f
#endif

#ifndef IMAGESIZE_X
#define IMAGESIZE_X 2048
#endif

#ifndef IMAGESIZE_Y
#define IMAGESIZE_Y 2048
#endif

#ifndef IMAGESIZE_M
#define IMAGESIZE_M 4194304 //2048*2048
#endif

#ifndef CONTRAIN_NUM
#define CONTRAIN_NUM 0
#endif

inline void linear_transform(float *A, float *x_src, float *x_dest, int dim, __constant char *p_fix){
    
    for(int i=0; i<dim; i++){
        x_dest[i]=0.0f;
        if(p_fix[i]== 48) continue;
        for(int j=0; j<dim; j++){
            if(p_fix[j]== 48) continue;
            x_dest[i] += A[i+j*PARA_NUM]*x_src[j];
        }
    }
}


//simultaneous linear equation
inline void sim_linear_eq(float *A, float *x, int dim, __constant char *p_fix){
    
    float a = 0.0f;
    
    //inversed matrix
    for(int i=0; i<dim; i++){
        if(p_fix[i]== 48) continue;
        
        //devide (i,i) to 1
        a = 1/A[i+i*dim];
        for(int j=0; j<dim; j++){
            if(p_fix[j]== 48) continue;
            A[i+j*dim] *= a;
        }
        x[i] *= a;
        
        //erase (j,i) (i!=j) to 0
        for(int j=i+1; j<dim; j++){
            if(p_fix[j]== 48) continue;
            
            a = A[j+i*dim];
            for(int k=0; k<dim; k++){
                if(p_fix[k]== 48) continue;
                A[j+k*dim] -= a*A[i+k*dim];
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
            }
            x[i] -= a*x[j];
        }
    }
}



__kernel void LevenbergMarquardt(__global float* tJdF_img, __global float* tJJ_img,
                                 __global float* dp_img, __global float* lambda_img,
                                 __constant char *p_fix){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t imageSizeX = get_global_size(0);
    const size_t imageSizeY = get_global_size(1);
    const size_t global_ID = global_x+global_y*imageSizeX;
    const size_t imageSizeM = imageSizeX*imageSizeY;
    
    float dp[PARA_NUM],lambda_diag_tJJ[PARA_NUM];
    float tJJ[PARA_NUM_SQ];
    float lambda = lambda_img[global_ID];
    float wgt[PARA_NUM];
    
    for(int i=0;i<PARA_NUM;i++){
        if(p_fix[i]==48) continue;
        
        tJJ[i+i*PARA_NUM] = tJJ_img[global_ID+(PARA_NUM*i-(i-1)*i/2)*imageSizeM];
        tJJ[i+i*PARA_NUM] *= (1.0f+lambda);
        wgt[i] = 1.0f/sqrt(tJJ[i+i*PARA_NUM]);
        lambda_diag_tJJ[i] = tJJ[i+i*PARA_NUM]*lambda;
        dp[i] = tJdF_img[global_ID + i*imageSizeM];
        
        for(int j=i+1;j<PARA_NUM;j++){
            if(p_fix[j]==48) continue;
            
            tJJ[i+j*PARA_NUM] = tJJ_img[global_ID+(PARA_NUM*i-(i+1)*i/2+j)*imageSizeM];
            tJJ[j+i*PARA_NUM] = tJJ[i+j*PARA_NUM];
        }
    }
    
    
    for(int i=0;i<PARA_NUM;i++){
        if(p_fix[i]==48) continue;
        
        dp[i] *= wgt[i];
        for(int j=0;j<PARA_NUM;j++){
            if(p_fix[j]==48) continue;
            
            tJJ[i+j*PARA_NUM] *= wgt[i]*wgt[j];
        }
    }
    
    
    /*for(int i=0;i<PARA_NUM;i++){
        for(int j=0;j<PARA_NUM;j++){
            printf("%f\t",tJJ[i+j*PARA_NUM]);
        }
        printf("\n");
    }*/
    
    sim_linear_eq(tJJ,dp,PARA_NUM,p_fix);
    
    for(int i=0;i<PARA_NUM;i++){
        dp[i] = (p_fix[i]==48) ? 0.0f:dp[i]*wgt[i];
        dp[i] = isnan(dp[i]) ? 0.0f:dp[i];
        dp[i] = isinf(dp[i]) ? 0.0f:dp[i];
        
        dp_img[global_ID+i*imageSizeM] = dp[i];
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
    const size_t imageSizeY = get_global_size(1);
    const size_t imageSizeM = imageSizeX*imageSizeY;
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
    
    
    //update tJJ, tJdF
    for(int i=0;i<PARA_NUM;i++){
        float a = tJJ_img[global_ID+(PARA_NUM*i-(i-1)*i/2)*imageSizeM];
        tJJ_img[global_ID+(PARA_NUM*i-(i-1)*i/2)*imageSizeM] = (rho>=0.0f) ? a:1.0f;
        
        a=tJdF_img[global_ID+i*imageSizeM];
        tJdF_img[global_ID+i*imageSizeM] = (rho>=0.0f) ? a:0.0f;
        for(int j=i+1;j<PARA_NUM;j++){
            a = tJJ_img[global_ID+(PARA_NUM*i-(i+1)*i/2+j)*imageSizeM];
            tJJ_img[global_ID+(PARA_NUM*i-(i+1)*i/2+j)*imageSizeM] = (rho>=0.0f) ? a:0.0f;
        }
    }
    
    
    rho_img[global_ID] = rho;
}



__kernel void updateOrRestore(__global float* para_img, __global float* para_img_backup,
                              __global float* rho_img, int z_id1, int z_id2){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t imageSizeX = get_global_size(0);
    const size_t imageSizeY = get_global_size(1);
    const size_t IDxy = global_x+global_y*imageSizeX;
    
    const size_t IDxyz1 = IDxy + (global_z+z_id1)*imageSizeX*imageSizeY;
    const size_t IDxyz2 = IDxy + (global_z+z_id2)*imageSizeX*imageSizeY;
    
    float rho = rho_img[IDxy];
    
    float para_A = para_img[IDxyz1];
    float para_B = para_img_backup[IDxyz2];
    
    
    para_img[IDxyz1] = (rho>=0) ? para_A:para_B;
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



__kernel void ISTA(__global float* fp_img,__global float* tJJ_img,__constant char *p_fix,
                   __global float* lambda_LM, __constant float* lambda_fista){
    
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const size_t IDxy = x+y*IMAGESIZE_X;

    float lambda1 = lambda_LM[IDxy];
    
    float fp_neighbor[4*PARA_NUM];
    const int px = x+1;
    const int py = y+1;
    const int mx = x-1;
    const int my = y-1;
    const size_t IDpxy = px+y*IMAGESIZE_X;
    const size_t IDmxy = mx+y*IMAGESIZE_X;
    const size_t IDxpy = x+py*IMAGESIZE_X;
    const size_t IDxmy = x+my*IMAGESIZE_X;
    float fp[PARA_NUM], fp_cnd[PARA_NUM];
    float sigma[PARA_NUM], tJJ[PARA_NUM_SQ];
    
    
    //load tJJ fp, dp
    for(int i=0;i<PARA_NUM;i++){
        if(p_fix[i]==48) continue;
        
        fp[i] = fp_img[IDxy + i*IMAGESIZE_M];
        tJJ[i+i*PARA_NUM] = tJJ_img[IDxy+(PARA_NUM*i-(i-1)*i/2)*IMAGESIZE_M];
        tJJ[i+i*PARA_NUM] *= (1.0f+lambda1);
        for(int j=i+1;j<PARA_NUM;j++){
            
            tJJ[i+j*PARA_NUM] = tJJ_img[IDxy+(PARA_NUM*i-(i+1)*i/2+j)*IMAGESIZE_M];
            tJJ[j+i*PARA_NUM] = tJJ[i+j*PARA_NUM];
        }
    }
    
    
    for(int i=0;i<PARA_NUM;i++){
        if(p_fix[i]==48) continue;
        
        //estimate neighbor fp
        fp_neighbor[0+i*4]=(px>=IMAGESIZE_X)? fp[i]:fp_img[IDpxy+i*IMAGESIZE_M];
        fp_neighbor[1+i*4]=(py>=IMAGESIZE_Y)? fp[i]:fp_img[IDxpy+i*IMAGESIZE_M];
        fp_neighbor[2+i*4]=(mx < 0)			? fp[i]:fp_img[IDmxy+i*IMAGESIZE_M];
        fp_neighbor[3+i*4]=(my < 0)			? fp[i]:fp_img[IDxmy+i*IMAGESIZE_M];
        
        
        //change order of fp_neighbor[4]
        float fp1, fp2;
        int biggerN;
        for(int k=0;k<4;k++){
            fp1 = fp_neighbor[k+i*4];
            fp2 = fp1;
            biggerN = k;
            for(int j=k+1;j<4;j++){
                biggerN = (fp_neighbor[j+i*4]>fp2) ? j:biggerN;
                fp2 = (fp_neighbor[j+i*4]>fp2) ? fp_neighbor[j+i*4]:fp2;
            }
            fp_neighbor[k+i*4] = fp2;
            fp_neighbor[biggerN+i*4] = fp1;
        }
        
        float lambda2 = lambda_fista[i];
        sigma[i] = 4.0f*lambda2;
        sigma[i] = (fp[i]>=fp_neighbor[0+i*4]) ? sigma[i]:2.0f*lambda2;
        sigma[i] = (fp[i]>=fp_neighbor[1+i*4]) ? sigma[i]:0.0f*lambda2;
        sigma[i] = (fp[i]>=fp_neighbor[2+i*4]) ? sigma[i]:-2.0f*lambda2;
        sigma[i] = (fp[i]>=fp_neighbor[3+i*4]) ? sigma[i]:-4.0f*lambda2;
    }

    
    sim_linear_eq(tJJ,sigma,PARA_NUM,p_fix);
    
    
    //apply soft threshold
    for(int i=0;i<PARA_NUM;i++){
        if(p_fix[i]==48) continue;
        
       // float lambda2 = lambda_fista[i];
        fp_cnd[i] = fp[i] - sigma[i];//fmax(-10.0f*lambda2,fmin(10.0f*lambda2, sigma[i]));
        if(fp[i]>=fp_neighbor[0+i*4]){
            fp_cnd[i] = (fp_cnd[i]< fp_neighbor[0+i*4]) ? fp_neighbor[0+i*4]:fp_cnd[i];
        }else if(fp[i]>=fp_neighbor[1+i*4]){
            fp_cnd[i] = (fp_cnd[i]>=fp_neighbor[0+i*4]) ? fp_neighbor[0+i*4]:fp_cnd[i];
            fp_cnd[i] = (fp_cnd[i]< fp_neighbor[1+i*4]) ? fp_neighbor[1+i*4]:fp_cnd[i];
        }else if(fp[i]>=fp_neighbor[2+i*4]){
            fp_cnd[i] = (fp_cnd[i]>=fp_neighbor[1+i*4]) ? fp_neighbor[1+i*4]:fp_cnd[i];
            fp_cnd[i] = (fp_cnd[i]< fp_neighbor[2+i*4]) ? fp_neighbor[2+i*4]:fp_cnd[i];
        }else if(fp[i]>=fp_neighbor[3+i*4]){
            fp_cnd[i] = (fp_cnd[i]>=fp_neighbor[2+i*4]) ? fp_neighbor[2+i*4]:fp_cnd[i];
            fp_cnd[i] = (fp_cnd[i]< fp_neighbor[3+i*4]) ? fp_neighbor[3+i*4]:fp_cnd[i];
        }
    }
    
    
    //save fp
    for(int i=0;i<PARA_NUM;i++){
        if(p_fix[i]==48) continue;
        fp_img[IDxy + i*IMAGESIZE_M] = fp_cnd[i];
    }
}


__kernel void FISTA(__global float* fp_x_img, __global float* fp_w_img, __global float* beta_img,
                     __global float* tJJ_img, __constant char *p_fix,
                     __global float* lambda_LM, __constant float* lambda_fista){
    
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const size_t IDxy = x+y*IMAGESIZE_X;
    
    float lambda1 = lambda_LM[IDxy];
    
    float fp_neighbor[4*PARA_NUM];
    const int px = x+1;
    const int py = y+1;
    const int mx = x-1;
    const int my = y-1;
    const size_t IDpxy = px+y*IMAGESIZE_X;
    const size_t IDmxy = mx+y*IMAGESIZE_X;
    const size_t IDxpy = x+py*IMAGESIZE_X;
    const size_t IDxmy = x+my*IMAGESIZE_X;
    float fp_x[PARA_NUM], fp_w[PARA_NUM], fp_x_new[PARA_NUM];
    float sigma[PARA_NUM], tJJ[PARA_NUM_SQ];
    float beta = beta_img[IDxy];
    float beta_new = beta*beta*4.0f + 1.0f;
    beta_new = (sqrt(beta_new) + 1.0f)*0.5f;
    float gamma = (beta-1.0f)/beta_new;
    
    
    //load tJJ fp, dp
    for(int i=0;i<PARA_NUM;i++){
        if(p_fix[i]==48) continue;
        
        fp_w[i] = fp_w_img[IDxy + i*IMAGESIZE_M];
        fp_x[i] = fp_x_img[IDxy + i*IMAGESIZE_M];
        tJJ[i+i*PARA_NUM] = tJJ_img[IDxy+(PARA_NUM*i-(i-1)*i/2)*IMAGESIZE_M];
        tJJ[i+i*PARA_NUM] *= (1.0f+lambda1);
        for(int j=i+1;j<PARA_NUM;j++){
            tJJ[i+j*PARA_NUM] = tJJ_img[IDxy+(PARA_NUM*i-(i+1)*i/2+j)*IMAGESIZE_M];
            tJJ[j+i*PARA_NUM] = tJJ[i+j*PARA_NUM];
        }
    }
    
    
    for(int i=0;i<PARA_NUM;i++){
        if(p_fix[i]==48) continue;
        
        //estimate neighbor fp_w
        fp_neighbor[0+i*4]=(px>=IMAGESIZE_X)? fp_w[i]:fp_w_img[IDpxy+i*IMAGESIZE_M];
        fp_neighbor[1+i*4]=(py>=IMAGESIZE_Y)? fp_w[i]:fp_w_img[IDxpy+i*IMAGESIZE_M];
        fp_neighbor[2+i*4]=(mx < 0)			? fp_w[i]:fp_w_img[IDmxy+i*IMAGESIZE_M];
        fp_neighbor[3+i*4]=(my < 0)			? fp_w[i]:fp_w_img[IDxmy+i*IMAGESIZE_M];
        
        
        //change order of fpdp_neibor[4]
        float fp1, fp2;
        int biggerN;
        for(int k=0;k<4;k++){
            fp1 = fp_neighbor[k+i*4];
            fp2 = fp1;
            biggerN = k;
            for(int j=k+1;j<4;j++){
                biggerN = (fp_neighbor[j+i*4]>fp2) ? j:biggerN;
                fp2 = (fp_neighbor[j+i*4]>fp2) ? fp_neighbor[j+i*4]:fp2;
            }
            fp_neighbor[k+i*4] = fp2;
            fp_neighbor[biggerN+i*4] = fp1;
        }
        
        
        float lambda2 = lambda_fista[i];
        sigma[i] = 4.0f*lambda2;
        sigma[i] = (fp_w[i]>=fp_neighbor[0+i*4]) ? sigma[i]:2.0f*lambda2;
        sigma[i] = (fp_w[i]>=fp_neighbor[1+i*4]) ? sigma[i]:0.0f*lambda2;
        sigma[i] = (fp_w[i]>=fp_neighbor[2+i*4]) ? sigma[i]:-2.0f*lambda2;
        sigma[i] = (fp_w[i]>=fp_neighbor[3+i*4]) ? sigma[i]:-4.0f*lambda2;
    }
    
    sim_linear_eq(tJJ,sigma,PARA_NUM,p_fix);
    
    //apply soft threshold
    for(int i=0;i<PARA_NUM;i++){
        if(p_fix[i]==48) continue;
        
        
        //float lambda2 = lambda_fista[i];
        fp_x_new[i] = fp_w[i] - sigma[i];//fmax(-10.0f*lambda2,fmin(10.0f*lambda2, sigma[i]));
        if(fp_w[i]>=fp_neighbor[0+i*4]){
            fp_x_new[i] = (fp_x_new[i]< fp_neighbor[0+i*4]) ? fp_neighbor[0+i*4]:fp_x_new[i];
        }else if(fp_w[i]>=fp_neighbor[1+i*4]){
            fp_x_new[i] = (fp_x_new[i]>=fp_neighbor[0+i*4]) ? fp_neighbor[0+i*4]:fp_x_new[i];
            fp_x_new[i] = (fp_x_new[i]< fp_neighbor[1+i*4]) ? fp_neighbor[1+i*4]:fp_x_new[i];
        }else if(fp_w[i]>=fp_neighbor[2+i*4]){
            fp_x_new[i] = (fp_x_new[i]>=fp_neighbor[1+i*4]) ? fp_neighbor[1+i*4]:fp_x_new[i];
            fp_x_new[i] = (fp_x_new[i]< fp_neighbor[2+i*4]) ? fp_neighbor[2+i*4]:fp_x_new[i];
        }else if(fp_w[i]>=fp_neighbor[3+i*4]){
            fp_x_new[i] = (fp_x_new[i]>=fp_neighbor[2+i*4]) ? fp_neighbor[2+i*4]:fp_x_new[i];
            fp_x_new[i] = (fp_x_new[i]< fp_neighbor[3+i*4]) ? fp_neighbor[3+i*4]:fp_x_new[i];
        }
        
        
        //update dp_w
        fp_w[i] = (1.0f-gamma)*fp_x_new[i] + gamma*fp_x[i];
    }
    
    //save dp
    for(int i=0;i<PARA_NUM;i++){
        if(p_fix[i]==48) continue;
        
        fp_x_img[IDxy + i*IMAGESIZE_M] = fp_x_new[i];
        fp_w_img[IDxy + i*IMAGESIZE_M] = fp_w[i];
    }
    //save beta
    beta_img[IDxy] = beta_new;
}


__kernel void contrain_0(__constant float *C_mat,__global float* C2_vec){
    
    const size_t c_num = get_global_id(0);
    float C2 =0.0f;
    for(int i=0; i<PARA_NUM; i++){
        C2 += C_mat[c_num*PARA_NUM+i]*C_mat[c_num*PARA_NUM+i];
    }
    
    C2_vec[c_num] = C2;
}


__kernel void contrain_1(__global float* fp_cnd_img, __global float* eval_img,
                          __constant float *C_mat, int c_num, int p_num){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t IDxy  = global_x+ global_y*size_x;
    const size_t IDxyz = IDxy + global_z*size_x*size_y;
    
    float fp = fp_cnd_img[IDxyz];
    float Cij = C_mat[c_num*PARA_NUM+p_num];
    eval_img[IDxy] += Cij*fp;
}


__kernel void contrain_2(__global float* fp_cnd_img, __global float* weight_img,
                          __global float* eval_img, __constant float *C_mat,
                          __constant float *D_vec, __constant float *C2_vec,
                          int c_num, int p_num, char weight_b){
    
    const size_t global_x = get_global_id(0);
    const size_t global_y = get_global_id(1);
    const size_t global_z = get_global_id(2);
    const size_t size_x = get_global_size(0);
    const size_t size_y = get_global_size(1);
    const size_t IDxy  = global_x+ global_y*size_x;
    const size_t IDxyz = IDxy + global_z*size_x*size_y;
    
    
    float fp   = fp_cnd_img[IDxyz];
    float eval = eval_img[IDxy];
    float C2 = C2_vec[c_num];
    float weight = weight_img[IDxy];
    float D = (weight_b==48) ? D_vec[c_num]:D_vec[c_num]*weight;
    
    bool eval_b = (eval>D);
    float h = (eval-D)/sqrt(C2);
    fp = (eval_b) ? fp-h*C_mat[c_num*PARA_NUM+p_num]:fp;
    
    fp_cnd_img[IDxyz] = fp;
}
