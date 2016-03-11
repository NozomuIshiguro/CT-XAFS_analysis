
#include "CT_reconstruction.hpp"

static string base_f1;
static string tale_f1;
static string base_f2;
static string tale_f2;

static float	*g_prj;					/* 投影データ */
static float	*g_img;					/* 再構成データ */
static int		*g_cx;					/* 検出位置 */
static float	*g_c0;					/* 検出確率(-1) */
static float	*g_c1;					/* 検出確率(0) */
static float	*g_c2;					/* 検出確率(+1) */

static float	*g_prjf;				/* 仮想投影データ */

void detection_probability(int, int, int, float *);
void projection(float *, float *, int, int, int, int);
void backprojection(float *, float *, int, int, int, float *, int);
void backprojection_em(float *, float *, int, int, int, int);
void make_pj(float *, float *, int, int, int, float *, float *, int, int);
void AART(float *, float *, int, int, int, float *, int, int, double);
void MART(float *, float *, int, int, int, float *, int, int);
void ASIRT(float *, float *, int, int, int, float *, int, int, double);
void MSIRT(float *, float *, int, int, int, float *, int, int);
void MLEM(float *, float *, int, int, int, float *, int, int);
void OSEM(float *, float *, int, int, int, float *, int, int, int);
void FBP(float *, float *, int, int, int, float *, int);
void zero_padding(float *, float *, int, int, int);
void FFT_filter(float *, int, int);
void FFT(int, int, float *, float *, float *, float *, unsigned short *);
void bitrev(int, float *, float *, unsigned short *);
void FFT_init(int, float *, float *, unsigned short *);
int br(int, unsigned);
void Filter(float *, int);

int CPU(){
	MKDIR(g_d5.c_str());
    
    base_f1.erase(g_f1.find_first_of("*"),g_f1.length()-1);
    tale_f1.erase(0,g_f1.find_last_of("*")+1);
    base_f2.erase(g_f2.find_first_of("*"),g_f2.length()-1);
    tale_f2.erase(0,g_f2.find_last_of("*")+1);
    
	for (int N = g_st-1; N < g_st-1 + g_num; N++){
        ostringstream oss1, oss2;
        oss1<<base_f1<<setfill('0')<<setw(4)<<N+1<<tale_f1;
        g_f1=oss1.str();
        oss2<<base_f2<<setfill('0')<<setw(4)<<N+1<<tale_f2;
        g_f2=oss2.str();
		read_data_input(g_d1,g_f1, g_prj, g_px*g_pa);
        
        
        /*//ここから回転中心補正
		float *cent;
		int c,d;
		cent = (float *)malloc((unsigned long)g_px*g_pa*sizeof(float));
		for(c=0;c<g_pa;c++){
			for(d=0;c<g_px;d++){
				if(g_x<0){
					if(d<abs(g_x)){
						cent[c*g_px+d]=0;
					}else{
						cent[c*g_px+d]=g_prj[c*g_px+d+g_x];
					}
				}else{
					if(g_px-d<abs(g_x)){
						cent[c*g_px+d]=0;
					}else{
						cent[c*g_px+d]=g_prj[c*g_px+d+g_x];
					}
				}
			}
		}
		for(c=0;c<g_px*g_pa;c++)
			g_prj[c]=cent[c];
		free(cent);
        //ここまで*/
        
		first_image(g_f4, g_img, g_nx*g_nx);
		switch (g_mode){
		case 1: //add (加算型) algebraic reconstruction technique 法
			AART(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_it, N, g_wt1);
			break;
		case 2: //multiply (乗算型) algebraic reconstruction technique 法
			MART(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_it, N);
			break;
		case 3: //add (加算型) simultaneous reconstruction technique 法
			ASIRT(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_it, N, g_wt2);
			break;
		case 4: //multiply(乗算型) simultaneous reconstruction technique 法
			MSIRT(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_it, N);
			break;
		case 5: //maximum likelihood-expection maximumization (ML-EM)法
			MLEM(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_it, N);
			break;
		case 6:  //ordered subset EM (OS-EM)法
			OSEM(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_it, N, g_ss);
			break;
		case 7: //filter back-projection法
			FBP(g_img, g_prj, g_px, g_pa, g_nx, g_ang, N);
			break;
		default:
			break;
		}
		write_data_output(g_d2, g_f2, g_img, g_nx*g_nx);
	}

// ここから投影像推定(pa=900 @180degとする)
	switch (g_cover){
        case 1:
            g_prjf = new float[(unsigned long)g_px * 900];

            read_log("#log_900.dat", 900);
            read_data_input(g_d1, g_f1, g_prj, g_px*g_pa);
            read_data_input(g_d2, g_f2, g_img, g_nx*g_nx);
            make_pj(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_prjf, 900, 100);
            write_data_output(g_d2, "Rec2.img", g_img, g_nx*g_nx);
            delete [] g_prjf;
            break;
    }
	delete [] g_ang;
	delete [] g_prj;
	delete [] g_img;
	return 0;
}




void detection_probability(int px, int pa, int nx, float *ang){
	int		i, j, k, ix;
	float	x, y, xx, th, a, b, x05, d, si, co, cc[3];
	for (i = 0; i < pa*nx*nx; i++){
		g_cx[i] = 0;
		g_c0[i] = g_c1[i] = g_c2[i] = 0.0;
	}
	for (k = 0; k < pa; k++){
		th = g_ang[k] / 180 * float(M_PI);
		si = sinf(th);
		co = cosf(th);
		if (fabs(si) > fabs(co)){
			a = fabs(si);
			b = fabs(co);
		}
		else{
			a = fabs(co);
			b = fabs(si);
		}
		for (i = 0; i < nx; i++){
			y = float(nx / 2 - i);
			for (j = 0; j < nx; j++){
				x = float(j - nx / 2);
				xx = x * co + y * si;
				cc[0] = cc[1] = cc[2] = 0.0;
				ix = (int)(floor(xx + 0.5));
				if (ix + px / 2 < 1 || ix + px / 2 > px - 2)
					continue;
				x05 = float(ix - 0.5);
				if ((d = x05 - (xx - (a - b) / 2)) > 0.0)
					cc[0] = b / (2 * a) + d / a;
				else if ((d = x05 - (xx - (a + b) / 2)) > 0.0)
					cc[0] = d * d / (2 * a * b);
				x05 = float(ix + 0.5);
				if ((d = xx + (a - b) / 2 - x05) > 0.0)
					cc[2] = b / (2 * a) + d / a;
				else if ((d = xx + (a + b) / 2 - x05) > 0.0)
					cc[2] = d * d / (2 * a * b);
				cc[1] = float(1.0 - cc[0] - cc[2]);
				g_cx[k*nx*nx + i*nx + j] = ix + px / 2 - 1;
				g_c0[k*nx*nx + i*nx + j] = cc[0];
				g_c1[k*nx*nx + i*nx + j] = cc[1];
				g_c2[k*nx*nx + i*nx + j] = cc[2];
			}
		}
	}
}
void projection(float *img, float *prj, int px, int pa, int nx, int k){
	int    i, j;
	for (i = 0; i < px; i++)
		prj[k*px + i] = 0;
	for (j = 0; j < nx*nx; j++){
		prj[k*px + g_cx[k*nx*nx + j] + 0] += g_c0[k*nx*nx + j] * img[j];
		prj[k*px + g_cx[k*nx*nx + j] + 1] += g_c1[k*nx*nx + j] * img[j];
		prj[k*px + g_cx[k*nx*nx + j] + 2] += g_c2[k*nx*nx + j] * img[j];
	}
}
void backprojection(float *img, float *prj, int px, int pa, int nx, float *ang, int k){
	int     i, j, ix;
	float  x0, cx, cy, th, tx, ty, t1, t2;
	th = g_ang[k] / 180 * float(M_PI);
	cx = cosf(th);
	cy = -sinf(th);
	x0 = -cx * nx / 2 - cy*nx / 2 + px / 2;
	ty = x0;
	for (i = 0; i < nx; i++, ty += cy){
		tx = ty;
		for (j = 0; j < nx; j++, tx += cx){
			ix = (int)tx;
			if (ix < 0 || ix > px - 2)
				continue;
			t1 = tx - ix;
			t2 = 1 - t1;
			img[i * nx + j] += t1 * prj[k * px + ix + 1] + t2 * prj[k * px + ix];
		}
	}
}
void backprojection_em(float *img, float *prj, int px, int pa, int nx, int k){
	int    i;
	for (i = 0; i < nx*nx; i++){
		img[i] += g_c0[k*nx*nx + i] * prj[k*px + g_cx[k*nx*nx + i] + 0];
		img[i] += g_c1[k*nx*nx + i] * prj[k*px + g_cx[k*nx*nx + i] + 1];
		img[i] += g_c2[k*nx*nx + i] * prj[k*px + g_cx[k*nx*nx + i] + 2];
	}
}
void make_pj(float *img, float *prj, int px, int pa, int nx, float *ang, float *prjf, int paf, int mae){
	int     i, k;
	float   *aprj;
    aprj = new float[px*paf];
	detection_probability(px, paf, nx, ang);
	cout<<endl<<" *** 投影像作成 *** "<<endl;
	for (k = 0; k < pa; k++)
		projection(img, aprj, px, pa, nx, k);
	for (i = 0; i < pa*px; i++){
		aprj[i + mae*px] = prj[i];
	}
	write_data_output(g_d2, "RAW_to_900.img", aprj, px*paf);
	delete [] aprj;
}

void AART(float *img, float *prj, int px, int pa, int nx, float *ang, int it, int N, double wt){
	int     i, j, k, m1, m2, *sub;
	string    fi;
	float   *aprj, *rprj, *aimg;
    time_t t1;
    aprj = new float[px*pa];
    rprj = new float[px*pa];
    aimg = new float[nx*nx];
	sub = new int[pa];
	k = 0;
	for (i = 0; i < 32; i++)
		k += (pa >> i) & 1;
	if (k == 1){
		m1 = 0;
		sub[m1++] = 0;
		for (i = pa, m2 = 1; i > 1; i /= 2, m2 *= 2){
			for (j = 0; j < m2; j++)
				sub[m1++] = sub[j] + i / 2;
		}
	}
	else{
		for (i = 0; i < pa; i++)
			sub[i] = i;
	}
	for (i = 0; i < it; i++){
        time(&t1);
        cout<<endl<<" [Layer "<<setfill('0')<<setw(4)<<N + 1<<"/"<<g_num;
        cout<<"- Iteration "<<setfill('0')<<setw(3)<<i + 1<<"/"<<it<<"] ";
        cout<<(float)(t1 - g_t0)<<"秒経過"<<endl;
		for (k = 0; k < pa; k++){
			projection(img, aprj, px, pa, nx, sub[k]);
			for (j = 0; j < px; j++){
				rprj[sub[k] * px + j] = prj[sub[k] * px + j] - aprj[sub[k] * px + j];
			}
			for (j = 0; j < nx*nx; j++)
				aimg[j] = (float)0; 
			backprojection(aimg, rprj, px, pa, nx, ang, sub[k]);
			for (j = 0; j < nx*nx; j++)
				img[j] += (float)wt*aimg[j]/pa;
		}
        ostringstream oss;
        oss<<"L"<<setfill('0')<<setw(4)<<N+1<<"_I"<<setw(3)<<i+1<<"_AART.img";
        fi=oss.str();
		write_data_temp(g_d2, fi, img, nx*nx);
	}
	delete [] aprj;
	delete [] rprj;
	delete [] aimg;
}
void MART(float *img, float *prj, int px, int pa, int nx, float *ang, int it, int N){
	int     i, j, k, m1, m2, *sub;
	string    fi;
	float   *aprj, *rprj, *aimg;
    time_t t1;
    aprj = new float[px*pa];
    rprj = new float[px*pa];
    aimg = new float[nx*nx];
    sub = new int[pa];
	k = 0;
	for (i = 0; i < 32; i++)
		k += (pa >> i) & 1;
	if (k == 1){
		m1 = 0;
		sub[m1++] = 0;
		for (i = pa, m2 = 1; i > 1; i /= 2, m2 *= 2){
			for (j = 0; j < m2; j++)
				sub[m1++] = sub[j] + i / 2;
		}
	}
	else{
		for (i = 0; i < pa; i++)
			sub[i] = i;
	}
	for (i = 0; i < it; i++){
        time(&t1);
        cout<<endl<<" [Layer "<<setfill('0')<<setw(4)<<N + 1<<"/"<<g_num;
        cout<<"- Iteration "<<setfill('0')<<setw(3)<<i + 1<<"/"<<it<<"] ";
        cout<<(float)(t1 - g_t0)<<"秒経過"<<endl;
		for (k = 0; k < pa; k++){
			projection(img, aprj, px, pa, nx, sub[k]);
			for (j = 0; j < px; j++){
				if ((double)aprj[sub[k] * px + j] < 0.0001)
					rprj[sub[k] * px + j] = prj[sub[k] * px + j];
				else
					rprj[sub[k] * px + j] = prj[sub[k] * px + j] / aprj[sub[k] * px + j];
			}
			for (j = 0; j < nx*nx; j++)
				aimg[j] = (float)0; 
			backprojection(aimg, rprj, px, pa, nx, ang, sub[k]);
			for (j = 0; j < nx*nx; j++)
				img[j] *= (float)aimg[j];
		}
        ostringstream oss;
        oss<<"L"<<setfill('0')<<setw(4)<<N+1<<"_I"<<setw(3)<<i+1<<"_MART.img";
        fi=oss.str();
		write_data_temp(g_d5, fi, img, nx*nx);
	}
	delete [] aprj;
	delete [] rprj;
    delete [] aimg;
    delete [] sub;
}
void ASIRT(float *img, float *prj, int px, int pa, int nx, float *ang, int it, int N, double wt){
	int     i, j, k;
	string    fi;
	float   *aprj, *rprj, *aimg;
    time_t t1;
    aprj = new float[px*pa];
    rprj = new float[px*pa];
    aimg = new float[nx*nx];
	for (i = 0; i < it; i++){
        time(&t1);
        cout<<endl<<" [Layer "<<setfill('0')<<setw(4)<<N + 1<<"/"<<g_num;
        cout<<"- Iteration "<<setfill('0')<<setw(3)<<i + 1<<"/"<<it<<"] ";
        cout<<(float)(t1 - g_t0)<<"秒経過"<<endl;
		for (k = 0; k < pa; k++)
			projection(img, aprj, px, pa, nx, k);
		for (j = 0; j < px*pa; j++){
			rprj[j] = prj[j] - aprj[j];
		}
		for (j = 0; j < nx*nx; j++)
			aimg[j] = (float)0; 
		for (k = 0; k < pa; k++)
			backprojection(aimg, rprj, px, pa, nx, ang, k);
		for (j = 0; j < nx*nx; j++)
			img[j] += (float)wt*aimg[j]/pa;
		{
            ostringstream oss;
            oss<<"L"<<setfill('0')<<setw(4)<<N+1<<"_I"<<setw(3)<<i+1<<"_ASIRT.img";
            fi=oss.str();
			write_data_temp(g_d5, fi, img, nx*nx);
		}
	}
    delete [] aprj;
    delete [] rprj;
    delete [] aimg;
}
void MSIRT(float *img, float *prj, int px, int pa, int nx, float *ang, int it, int N){
	int     i, j, k;
	string    fi;
	float   *aprj, *rprj, *aimg;
    time_t t1;
    aprj = new float[px*pa];
    rprj = new float[px*pa];
    aimg = new float[nx*nx];
	for (i = 0; i < it; i++){
        time(&t1);
        cout<<endl<<" [Layer "<<setfill('0')<<setw(4)<<N + 1<<"/"<<g_num;
        cout<<"- Iteration "<<setfill('0')<<setw(3)<<i + 1<<"/"<<it<<"] ";
        cout<<(float)(t1 - g_t0)<<"秒経過"<<endl;
		for (k = 0; k < pa; k++)
			projection(img, aprj, px, pa, nx, k);
		for (j = 0; j < px*pa; j++){
			if ((double)aprj[j] < 0.0001)
				rprj[j] = prj[j];
			else
				rprj[j] = prj[j] / aprj[j];
		}
		for (j = 0; j < nx*nx; j++)
			aimg[j] = (float)0; 
		for (k = 0; k < pa; k++)
			backprojection(aimg, rprj, px, pa, nx, ang, k);
		for (j = 0; j < nx*nx; j++)
			img[j] *= (float)aimg[j]/pa;
		{
            ostringstream oss;
            oss<<"L"<<setfill('0')<<setw(4)<<N+1<<"_I"<<setw(3)<<i+1<<"_MSIRT.img";
            fi=oss.str();
			write_data_temp(g_d5, fi, img, nx*nx);
		}
	}
    delete [] aprj;
    delete [] rprj;
    delete [] aimg;
}
void MLEM(float *img, float *prj, int px, int pa, int nx, float *ang, int it, int N){
	int     i, j, k;
	string    fi;
	float   *aprj, *rprj, *aimg;
    time_t t1;
    aprj = new float[px*pa];
    rprj = new float[px*pa];
    aimg = new float[nx*nx];
	for (i = 0; i < it; i++){
        time(&t1);
        cout<<endl<<" [Layer "<<setfill('0')<<setw(4)<<N + 1<<"/"<<g_num;
        cout<<"- Iteration "<<setfill('0')<<setw(3)<<i + 1<<"/"<<it<<"] ";
        cout<<(float)(t1 - g_t0)<<"秒経過"<<endl;
		for (k = 0; k < pa; k++)
			projection(img, aprj, px, pa, nx, k);
		for (j = 0; j < px*pa; j++){
			if ((double)aprj[j] < 0.0001)
				rprj[j] = prj[j];
			else
				rprj[j] = prj[j] / aprj[j];
		}
		for (j = 0; j < nx*nx; j++)
			aimg[j] = (float)0;
		for (j = 0; j < pa; j++)
			backprojection_em(aimg, rprj, px, pa, nx, j);
		for (j = 0; j < nx*nx; j++)
			img[j] *= (float)aimg[j] / pa;
		{
            ostringstream oss;
            oss<<"L"<<setfill('0')<<setw(4)<<N+1<<"_I"<<setw(3)<<i+1<<"_MLEM.img";
            fi=oss.str();
			write_data_temp(g_d5, fi, img, nx*nx);
		}
	}
    delete [] aprj;
    delete [] rprj;
    delete [] aimg;
}

void OSEM(float *img, float *prj, int px, int pa, int nx, float *ang, int it, int N, int ss){
	int     i, j, k, m1, m2, *sub, pi, pj, bi, bj, ix, *cx;
	string    fi;
	float	x, y, xx, th, a, b, x05, d, si, co, cc[3], *aprj, *rprj, *aimg, *c0, *c2;
    time_t t1;
    aprj = new float[px*pa];
    rprj = new float[px*pa];
    aimg = new float[nx*nx];
    sub =  new int[ss];
	cx =   new int[int(pa/ss)*g_nx*g_nx];
	c0 =   new float[int(pa/ss)*g_nx*g_nx];
	c2 =   new float[int(pa/ss)*g_nx*g_nx];
	
    // サブセットの順番を決定する
    k = 0;
	for (i = 0; i < 32; i++) k += (ss >> i) & 1;
	if (k == 1){    //ssが2^nの場合
		m1 = 0;
		sub[m1++] = 0;
		for (i = ss, m2 = 1; i > 1; i /= 2, m2 *= 2){
			for (j = 0; j < m2; j++) sub[m1++] = sub[j] + i / 2;
		}
	}
	else {
		for (i = 0; i < ss; i++) sub[i] = i;
	}
    
	for (i = 0; i < it; i++){   //反復
		// 検出確率(係数行列)の計算
        for (k = 0; k < ss; k++){
			for (pi = 0; pi < pa/ss*nx*nx; pi++){
				cx[pi] = 0;
				c0[pi] = c2[pi] = 0.0;
			}
			for (j = sub[k]; j < pa; j += ss){ //投影(角度)数
				th = g_ang[j] / 180 * float(M_PI);
				si = sinf(th);
				co = cosf(th);
				if (fabs(si) > fabs(co)){
					a = fabs(si);
					b = fabs(co);
				}
				else{
					a = fabs(co);
					b = fabs(si);
				}
				for (pi = 0; pi < nx; pi++){  //再構成xピクセルサイズ
					y = float(nx / 2 - pi);
					for (pj = 0; pj < nx; pj++){
						x = float(pj - nx / 2);
						xx = x * co + y * si;
						cc[0] = cc[2] = 0.0;
						ix = (int)(floor(xx + 0.5));
						if (ix + px / 2 < 1 || ix + px / 2 > px - 2)
							continue;
						x05 = float(ix - 0.5);
						if ((d = x05 - (xx - (a - b) / 2)) > 0.0)
							cc[0] = b / (2 * a) + d / a;
						else if ((d = x05 - (xx - (a + b) / 2)) > 0.0)
							cc[0] = d * d / (2 * a * b);
						x05 = float(ix + 0.5);
						if ((d = xx + (a - b) / 2 - x05) > 0.0)
							cc[2] = b / (2 * a) + d / a;
						else if ((d = xx + (a + b) / 2 - x05) > 0.0)
							cc[2] = d * d / (2 * a * b);
						cx[j/ss*nx*nx + pi*nx + pj] = ix + px / 2 - 1;
						c0[j/ss*nx*nx + pi*nx + pj] = cc[0];
						c2[j/ss*nx*nx + pi*nx + pj] = cc[2];
					}
				}
			}
            
            // ㈪ 仮定画像から投影を計算する
			for (j = sub[k]; j < pa; j += ss){  //投影(角度)数
				for (pi = 0; pi < px; pi++)
					aprj[j*px + pi] = 0;						
				for (pj = 0; pj < nx*nx; pj++){
					aprj[j*px + cx[j/ss*nx*nx + pj] + 0] += img[pj] * c0[j/ss*nx*nx + pj];
					aprj[j*px + cx[j/ss*nx*nx + pj] + 1] += img[pj] * (1 - c0[j/ss*nx*nx + pj] - c2[j/ss*nx*nx + pj]);
					aprj[j*px + cx[j/ss*nx*nx + pj] + 2] += img[pj] * c2[j/ss*nx*nx + pj];
				}
			}
            
            // ㈫ 投影データと，㈪で計算した投影との比を計算する
			for (j = sub[k]; j < pa; j += ss){  //投影(角度)数
				for (bi = 0; bi < px; bi++){
					// 仮定画像からの投影の値が小さいときは割り算を行わない
                    if ((double)aprj[j*px + bi] < 0.0001) rprj[j*px + bi] = prj[j*px + bi];
					else rprj[j*px + bi] = prj[j*px + bi] / aprj[j*px + bi];
				}
			}
            
            // ㈬ ㈫で計算された比を逆投影する
			for (j = 0; j < nx*nx; j++) aimg[j] = 0.0;
			for (j = sub[k]; j < pa; j += ss){
				for (bj = 0; bj < nx*nx; bj++){
					aimg[bj] += rprj[j*px + cx[j/ss*nx*nx + bj] + 0] * c0[j/ss*nx*nx + bj];
					aimg[bj] += rprj[j*px + cx[j/ss*nx*nx + bj] + 1] * (1 - c0[j/ss*nx*nx + bj] - c2[j/ss*nx*nx + bj]);
					aimg[bj] += rprj[j*px + cx[j/ss*nx*nx + bj] + 2] * c2[j/ss*nx*nx + bj];
				}
			}
            
            // ㈭ 逆投影画像を仮定画像に掛ける（画像の更新）
			for (j = 0; j < nx*nx; j++) img[j] *= (float)aimg[j] * ss / pa;
		}
        time(&t1);
        
        cout<<endl<<"[Layer " << setfill('0')<<setw(4)<< N+1 <<"/"<< g_st -1 + g_num;
        cout<<" - Iteration "<< setfill('0')<<setw(3)<<i + 1<<"/"<< it<<"] ";
        cout<<(float)(t1-g_t0)<<"秒経過"<<endl;
        
        ostringstream oss1,oss2,oss3,oss4;
        oss1<<"L"<<setfill('0')<<setw(4)<<N+1<<"_I"<<setw(3)<<i+1<<"_aprj.img";
        fi=oss1.str();
		write_data_temp(g_d5, fi, aprj, px*pa);
        oss2<<"L"<<setfill('0')<<setw(4)<<N+1<<"_I"<<setw(3)<<i+1<<"_rprj.img";
        fi=oss2.str();
		write_data_temp(g_d5, fi, rprj, px*pa);
        oss3<<"L"<<setfill('0')<<setw(4)<<N+1<<"_I"<<setw(3)<<i+1<<"_aim.img";
        fi=oss3.str();
		write_data_temp(g_d5, fi, aimg, nx*nx);
        oss4<<"L"<<setfill('0')<<setw(4)<<N+1<<"_I"<<setw(3)<<i+1<<"_rec.img";
        fi=oss4.str();
		write_data_temp(g_d5, fi, img, nx*nx);
	}
    delete [] aprj;
    delete [] rprj;
    delete [] aimg;
    delete [] sub;
	delete [] cx;
	delete [] c0;
	delete [] c2;
}

void FBP(float *img, float *prj, int px, int pa, int nx, float *ang, int N){
	int		zx = px*2;
	float	*prz;
    prz = new float[(unsigned long)zx*pa];
	zero_padding(prj, prz, px, pa, zx);
	FFT_filter(prz, zx, pa);
	int     i, j, k, ix;
	double  x0, cx, cy, th, tx, ty, t1, t2;
	float   *bp2;
    cout<<endl<<" *** FBP *** [Layer "<<setfill('0')<<setw(4)<<N + 1<<"/"<<g_num<<"]"<<endl;
	for (k = 0; k < pa; k++) {
		th = g_ang[k] / 180 * M_PI;
		cx = cos(th);
		cy = -sin(th);
		x0 = -cx * nx / 2 - cy * nx / 2 + zx / 2;
		bp2 = prz + k * zx;
		for (i = 0, ty = x0; i < nx; i++, ty += cy) {
			for (j = 0, tx = ty; j < nx; j++, tx += cx) {
				ix = (int)tx;
				if (ix < 0 || ix > zx - 2)     continue;
				t1 = tx - ix;
				t2 = 1 - t1;
				img[i*nx + j] += (float)(t1 * bp2[ix + 1] + t2 * bp2[ix]);
			}
		}
	}
	for (i = 0; i < nx * nx; i++)
		img[i] /= pa;
	delete [] prz;
}
void zero_padding(float *prj, float *prz, int px, int pa, int zx){
	int		i, j;
	for (i = 0; i < zx * pa; i++)
		prz[i] = 0;
	for (i = 0; i < pa; i++) {
		for (j = 0; j < px; j++) {
			prz[i * zx + zx / 2 + j - px / 2] = prj[i * px + j];
		}
	}
}
void FFT_filter(float *prj, int px, int pa){
	float	*xr, *xi, *si, *co;
	unsigned short	*br;
	int		i, j;
    xr = new float[(unsigned long)px];
    xi = new float[(unsigned long)px];
    si = new float[(unsigned long)px/2];
    co = new float[(unsigned long)px/2];
    br = new unsigned short[(unsigned long)px];
	FFT_init(px, si, co, br);
	for (i = 0; i < pa; i++) {
		for (j = 0; j < px / 2; j++) { 
			xr[j] = prj[i * px + j + px / 2];
			xr[j + px / 2] = prj[i * px + j];
			xi[j] = xi[j + px / 2] = 0;
		}
		FFT(1, px, xr, xi, si, co, br);
		Filter(xr, px);
		Filter(xi, px);
		FFT(-1, px, xr, xi, si, co, br);
		for (j = 0; j < px / 2; j++) {
			prj[i * px + j] = xr[j + px / 2];
			prj[i * px + j + px / 2] = xr[j];
		}
	}
    
    delete [] xr;
    delete [] xi;
    delete [] si;
    delete [] co;
    delete [] br;
}
void FFT_init(int nx, float *si, float *co, unsigned short *brv){
	double	d = 2*M_PI / nx;
	int		i;
	for (i = 0; i < nx / 4; i++) {
		si[i] = (float)sin(d * i);
		co[i + nx / 4] = -si[i];
	}
	for (i = nx / 4; i < nx / 2; i++) {
		si[i] = (float)sin(d * i);
		co[i - nx / 4] = si[i];
	}
	for (i = 0; i < nx; i++)
		brv[i] = br(nx, (unsigned)i);
}
int br(int nx, unsigned nn){
	unsigned	c, r = 0;
	for (c = 1; c <= (unsigned)nx / 2; c <<= 1) {
		r <<= 1;
		if ((nn&c) != 0)
			r++;
	}
	return(r);
}
void FFT(int ir, int nx, float *xr, float *xi, float *si, float *co, unsigned short *brv){
// int		ir;  順変換(1)と逆変換(-1)
	int		d = 1, g, i, j, j3, j4, k, l, ll, n1, n2 = nx;
	float	a, b, c, s;
	for (l = 1; l <= nx / 2; l *= 2, d += d) {
		g = 0;
		ll = n2;
		n2 /= 2;
		for (k = 1; k <= n2; k++) {
			n1 = k - ll;
			c = co[g];
			s = -ir * si[g];
			g += d;
			for (j = ll; j <= nx; j += ll) {
				j3 = j + n1 - 1;
				j4 = j3 + n2;
				a = xr[j3] - xr[j4];
				b = xi[j3] - xi[j4];
				xr[j3] += xr[j4];
				xi[j3] += xi[j4];
				xr[j4] = c * a + s * b;
				xi[j4] = c * b - s * a;
			}
		}
	}
	bitrev(nx, xr, xi, brv);
    if (ir == -1){
        for (i = 0; i < nx; i++) {
            xr[i] /= nx;
            xi[i] /= nx;
        }
    }
}
void bitrev(int nx, float *xr, float *xi, unsigned short *brv){
	int		i, j;
	float	a, b;
	for (i = 0; i < nx; i++){
		j = brv[i];
		if (i < j){
			a = xr[i];
			b = xi[i];
			xr[i] = xr[j];
			xi[i] = xi[j];
			xr[j] = a;
			xi[j] = b;
		}
	}
}
void Filter(float *xr, int px){
	int		i;
	double	h = M_PI / px;
	for (i = 0; i < px / 2; i++)
		xr[i] *= (float)(i * h);
	for (i = px / 2; i < px; i++)
		xr[i] *= (float)((px - i) * h);
}