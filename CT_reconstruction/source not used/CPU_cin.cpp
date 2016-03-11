#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <direct.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <windows.h>
#include <mmsystem.h>
#pragma comment (lib, "winmm.lib")
#include <algorithm>

char	g_f1[50] = "i****.raw";	/* 入力ファイル */
char	g_f2[50] = "r****.img";	/* 出力ファイル */
char	g_f3[50] = "log.dat";	/* logファイル */
char	g_f4[50] = "1st.img";	/* 初期画像ファイル */
int		g_fst = 2;				/* 初期画像の使用 */
double	g_ini = 1;				/* 初期値 */
int		g_px = 2048;				/* カメラピクセルサイズ */
int		g_pa = 1600;				/* 投影数 */
int		g_nx = 2048;				/* 再構成ピクセルサイズ */ //nx=pxでいい//
int		g_mode = 6;				/* 再構成法 */
int		g_it = 1;				/* 反復回数 */
int		g_num = 1;			/* レイヤー数 */
int		g_st = 1;			/* 開始ナンバー */
int		g_x = 0;			/* 回転中心のずれ */
double	g_wt1 = 1.0;			/* AARTファクター */
double	g_wt2 = 0.01;			/* ASIRTファクター */
int		g_ss = 20;				/* サブセット */ //投影数に応じて割り切れる数に//
float	*g_prj;					/* 投影データ */
float	*g_ang;					/* 投影角度 */
float	*g_img;					/* 再構成データ */
int		*g_cx;					/* 検出位置 */
float	*g_c0;					/* 検出確率(-1) */
float	*g_c1;					/* 検出確率(0) */
float	*g_c2;					/* 検出確率(+1) */
int		g_cover = 2;			/* 非投影領域の推定 */
float	*g_prjf;				/* 仮想投影データ */
DWORD	g_t0;					/* 時間 */

char *menu[] = {
//	" 測定ファイル(現在無効:inputフォルダ)      ",
//	" 出力ファイル(現在無効:outputフォルダ)     ",
//	" ログファイル名                              ",
	" レイヤー数                                        ",
	" 開始ナンバー                                      ",
	" 初期画像の使用 1:Yes 2: No                        ",
	" 初期画像ファイル名                          ",
//	" ベタ塗の初期値                             ",
	" カメラピクセル数                               ",
	" 投影数                                         ",
	" 再構成画像サイズ                               ",
//	" 回転中心のズレ                                    ",
//	" 1:AART 2:MART 3:ASIRT 4:MSIRT 5:MLEM 6:OSEM 7:FBP ",
	" 反復回数                                          ",
//	" ファクター(AART)                           ",
//	" ファクター(ASIRT)                          ",
	" サブセット(OSEM)                                 ",
//	" 非投影領域の推定 1:Yse 2:No                       ",
};

void getparameter();
void read_log(char *, int);
void read_data_input(char *, float *, int);
void write_data_output(char *, float *, int);
void write_data_temp(char *, float *, int);
//void first_image(char *, float *, int);//
//void detection_probability(int, int, int, float *);
//void projection(float *, float *, int, int, int, int);
////void backprojection(float *, float *, int, int, int, float *, int);//
//void backprojection_em(float *, float *, int, int, int, int);
//void make_pj(float *, float *, int, int, int, float *, float *, int, int);
//void AART(float *, float *, int, int, int, float *, int, int, double);
//void MART(float *, float *, int, int, int, float *, int, int);
//void ASIRT(float *, float *, int, int, int, float *, int, int, double);
//void MSIRT(float *, float *, int, int, int, float *, int, int);
//void MLEM(float *, float *, int, int, int, float *, int, int);
void OSEM(float *, float *, int, int, int, float *, int, int, int);

/*void FBP(float *, float *, int, int, int, float *, int);
void FFT_filter(float *, int, int);void FFT(int, int, float *, float *, float *, float *, unsigned short *);
void bitrev(int, float *, float *, unsigned short *);
void FFT_init(int, float *, float *, unsigned short *);
int br(int, unsigned);
void Filter(float *, int);*/

int main(void){
	printf("\n");
	getparameter();
	g_t0 = timeGetTime();
	int N;
	g_ang = (float *)malloc((unsigned long)g_pa*sizeof(float));
	g_prj = (float *)malloc((unsigned long)g_px*g_pa*sizeof(float));
	g_img = (float *)malloc((unsigned long)g_nx*g_nx*sizeof(float));
	read_log(g_f3, g_pa);
	_mkdir("output");
	_mkdir("temp");
	for (N = g_st-1; N < g_st-1 + g_num; N++){
		sprintf_s(g_f1, 32, "i%04d.raw", N + 1);
		sprintf_s(g_f2, 32, "r%04d.img", N + 1);
		read_data_input(g_f1, g_prj, g_px*g_pa);
//ここから回転中心補正
		//float *cent;
		//int c,d;
		//cent = (float *)malloc((unsigned long)g_px*g_pa*sizeof(float));
		//for(c=0;c<g_pa;c++){
		//	for(d=0;c<g_px;d++){
		//		if(g_x<0){
		//			if(d<abs(g_x)){
		//				cent[c*g_px+d]=0;
		//			}else{
		//				cent[c*g_px+d]=g_prj[c*g_px+d+g_x];
		//			}
		//		}else{
		//			if(g_px-d<abs(g_x)){
		//				cent[c*g_px+d]=0;
		//			}else{
		//				cent[c*g_px+d]=g_prj[c*g_px+d+g_x];
		//			}
		//		}
		//	}
		//}
		//for(c=0;c<g_px*g_pa;c++)
		//	g_prj[c]=cent[c];
		//free(cent);
//ここまで
		first_image(g_f4, g_img, g_nx*g_nx);
		OSEM(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_it, N, g_ss);

		write_data_output(g_f2, g_img, g_nx*g_nx);
	}

// ここから投影像推定(pa=900 @180degとする)
	//switch (g_cover){
	//case 1:
	//	g_prjf = (float *)malloc((unsigned long)g_px * 900 * sizeof(float));
	//	read_log("#log_900.dat", 900);
	//	read_data_input(g_f1, g_prj, g_px*g_pa);
	//	read_data_input(g_f2, g_img, g_nx*g_nx);
	//	make_pj(g_img, g_prj, g_px, g_pa, g_nx, g_ang, g_prjf, 900, 100);
	//	write_data_output("Rec2.img", g_img, g_nx*g_nx);
	//	free(g_prjf);
	//	break;
	//}
	free(g_ang);
	free(g_prj);
	free(g_img);
	return 0;
}

//ファイル読み込み関連//
void getparameter(){
	//パラメーター調整、初期値から変更を容易にしてある//

	int   i = 0;
	char  dat[256];
//	fprintf(stdout, " %s [%s] :", menu[i++], g_f1);
//	if (*fgets(dat, 256, stdin) != '\n')  strncpy_s(g_f1, 32, dat, strlen(dat) - 1);
//	fprintf(stdout, " %s [%s] :", menu[i++], g_f2);
//	if (*fgets(dat, 256, stdin) != '\n')  strncpy_s(g_f2, 32, dat, strlen(dat) - 1);
//	fprintf(stdout, " %s [%s] :", menu[i++], g_f3);
//	if (*fgets(dat, 256, stdin) != '\n')  strncpy_s(g_f3, 32, dat, strlen(dat) - 1);
	fprintf(stdout, " %s [%d] :", menu[i++], g_num);
	if (*fgets(dat, 256, stdin) != '\n')  g_num = atoi(dat);
	fprintf(stdout, " %s [%d] :", menu[i++], g_st);
	if (*fgets(dat, 256, stdin) != '\n')  g_st = atoi(dat);
	fprintf(stdout, " %s [%d] :", menu[i++], g_fst);
	if (*fgets(dat, 256, stdin) != '\n')  g_fst = atoi(dat);
	fprintf(stdout, " %s [%s] :", menu[i++], g_f4);
	if (*fgets(dat, 256, stdin) != '\n')  strncpy_s(g_f4, 32, dat, strlen(dat) - 1);
//	fprintf(stdout, " %s [%f] :", menu[i++], g_ini);
//	if (*fgets(dat, 256, stdin) != '\n')  g_ini = atof(dat);
	fprintf(stdout, " %s [%d] :", menu[i++], g_px);
	if (*fgets(dat, 256, stdin) != '\n')  g_px = atoi(dat);
	fprintf(stdout, " %s [%d] :", menu[i++], g_pa);
	if (*fgets(dat, 256, stdin) != '\n')  g_pa = atoi(dat);
	fprintf(stdout, " %s [%d] :", menu[i++], g_nx);
	if (*fgets(dat, 256, stdin) != '\n')  g_nx = atoi(dat);
//	fprintf(stdout, " %s [%d] :", menu[i++], g_x);
//	if (*fgets(dat, 256, stdin) != '\n')  g_x = atoi(dat);
//	fprintf(stdout, " %s [%d] :", menu[i++], g_mode);
//	if (*fgets(dat, 256, stdin) != '\n')  g_mode = atoi(dat);
	fprintf(stdout, " %s [%d] :", menu[i++], g_it);
	if (*fgets(dat, 256, stdin) != '\n')  g_it = atoi(dat);
//	fprintf(stdout, " %s [%f] :", menu[i++], g_wt1);
//	if (*fgets(dat, 256, stdin) != '\n')  g_wt1 = atoi(dat);
//	fprintf(stdout, " %s [%f] :", menu[i++], g_wt2);
//	if (*fgets(dat, 256, stdin) != '\n')  g_wt2 = atoi(dat);
	fprintf(stdout, " %s [%d] :", menu[i++], g_ss);
	if (*fgets(dat, 256, stdin) != '\n')  g_ss = atoi(dat);
//	fprintf(stdout, " %s [%d] :", menu[i++], g_cover);
//	if (*fgets(dat, 256, stdin) != '\n')  g_cover = atoi(dat);
}
void read_log(char *fi, int num){
	//ファイルfiを読み込みnumの数だけ１行ずつg_angに格納する//
	//メモリーの展開先をCUDA仕様にする際にグラボに写したほうがいいかもしれない//
	FILE   *fp;
	int k;
	errno_t error;
	if ((error = fopen_s(&fp, fi, "r")) != 0) {
		fprintf(stderr, "file open error [%s].\n", fi);
		exit(1);
	}
	for (k = 0; k < num; k++)fscanf_s(fp, "%f\n", &g_ang[k]);
	fclose(fp);
}
void read_data_input(char *fi, float *img, int size){
	//データ読み込み用//
	FILE   *fp;
	errno_t error;
	_chdir("input");//inputディレクトリに移動//
	if ((error = fopen_s(&fp, fi, "rb")) != 0) {
		fprintf(stderr, "file open error [%s].\n", fi);
		exit(1);
	}//filename fi をfpに開く読み取り専用バイナリーモード//
	fread(img, sizeof(float), size, fp);//ファイルfpよりimgにfloat長さのsize個読み込む//
	fclose(fp);
	_chdir("..");//元の作業フォルダに戻る//
}
void write_data_output(char *fi, float *img, int size){
	//データ書き込み用　ファイルfiにimgからsize個書き込む//
	FILE   *fp;
	errno_t error;
	_chdir("output");
	//outputフォルダに移動//
	if ((error = fopen_s(&fp, fi, "wb")) != 0) {
		fprintf(stderr, " file open error [%s].\n", fi);
		exit(1);
	}
	//ファイル名fiを書き込み用に新しく作りバイナリで開く//
	fwrite(img, sizeof(float), size, fp);
	//fiにimgからflortのサイズでsize個書き込む//
	fclose(fp);
	_chdir("..");
	//作業フォルダを元のフォルダにもどす//
}
void write_data_temp(char *fi, float *img, int size){
	//write_data_outputのtemp版//
	FILE   *fp;
	errno_t error;
	_chdir("temp");
	if ((error = fopen_s(&fp, fi, "wb")) != 0) {
		fprintf(stderr, " file open error [%s].\n", fi);
		exit(1);
	}
	fwrite(img, sizeof(float), size, fp);
	fclose(fp);
	_chdir("..");
}
void first_image(char *fi, float *img, int size){
	//g_fstの数値から最初の画像を使用するかを選択する//
	//g_fst初期値は2（使用しない）//
	int i;
	for (i = 0; i < size; i++)
	img[i] = (float)g_ini;
}

//計算関連//
//void detection_probability(int px, int pa, int nx, float *ang){
//	int		i, j, k, ix;
//	float	x, y, xx, th, a, b, x05, d, si, co, cc[3];
//	for (i = 0; i < pa*nx*nx; i++){
//		g_cx[i] = 0;
//		g_c0[i] = g_c1[i] = g_c2[i] = 0.0;
//	}
//	//int	*g_cx;					/* 検出位置 *//
//	//float	*g_c0;					/* 検出確率(-1) *//
//	//float	*g_c1;					/* 検出確率(0) *//
//	//float	*g_c2;					/* 検出確率(+1) *///
//	//上記関数をリセット//
//	for (k = 0; k < pa; k++){
//		th = g_ang[k] / 180 * float(M_PI);
//		//M_PI=円周率//
//		//float	*g_ang;					/* 投影角度 *///
//		si = sinf(th);		//sin(th)のfloat型//
//		co = cosf(th);		//cos(th)のfloat型//
//		if (fabs(si) > fabs(co)){
//			a = fabs(si);
//			b = fabs(co);
//		}
//		else{
//			a = fabs(co);
//			b = fabs(si);
//		}
//		//sin(th)の絶対値がcos(th)の絶対値より大きければ真、小さければ偽//
//		//つまるところaが大きい方、bが小さい方//
//		#pragma omp parallel for private(x, y, xx, ix, i, j, cc, d, x05)
//		//下記のfor文を並列化する。x,y,xx,ix,i,j,cc,d,x05はそれぞれのスレッドごとに独立した数字を持つ//
//		for (i = 0; i < nx; i++){
//		//nx回行う//
//			y = float(nx / 2 - i);
//			for (j = 0; j < nx; j++){
//				x = float(j - nx / 2);
//				xx = x * co + y * si;
//				cc[0] = cc[1] = cc[2] = 0.0;
//				ix = (int)(floor(xx + 0.5));
//				if (ix + px / 2 < 1 || ix + px / 2 > px - 2)
//					continue;
//				x05 = float(ix - 0.5);
//				if ((d = x05 - (xx - (a - b) / 2)) > 0.0)
//					cc[0] = b / (2 * a) + d / a;
//				else if ((d = x05 - (xx - (a + b) / 2)) > 0.0)
//					cc[0] = d * d / (2 * a * b);
//				x05 = float(ix + 0.5);
//				if ((d = xx + (a - b) / 2 - x05) > 0.0)
//					cc[2] = b / (2 * a) + d / a;
//				else if ((d = xx + (a + b) / 2 - x05) > 0.0)
//					cc[2] = d * d / (2 * a * b);
//				cc[1] = float(1.0 - cc[0] - cc[2]);
//				g_cx[k*nx*nx + i*nx + j] = ix + px / 2 - 1;
//				g_c0[k*nx*nx + i*nx + j] = cc[0];
//				g_c1[k*nx*nx + i*nx + j] = cc[1];
//				g_c2[k*nx*nx + i*nx + j] = cc[2];
//			}
//		}
//	}
//}
//void projection(float *img, float *prj, int px, int pa, int nx, int k){
//	int    i, j;
//	for (i = 0; i < px; i++)
//		prj[k*px + i] = 0;
//	for (j = 0; j < nx*nx; j++){
//		prj[k*px + g_cx[k*nx*nx + j] + 0] += g_c0[k*nx*nx + j] * img[j];
//		prj[k*px + g_cx[k*nx*nx + j] + 1] += g_c1[k*nx*nx + j] * img[j];
//		prj[k*px + g_cx[k*nx*nx + j] + 2] += g_c2[k*nx*nx + j] * img[j];
//	}
//}
//void backprojection(float *img, float *prj, int px, int pa, int nx, float *ang, int k){
//	int     i, j, ix;
//	float  x0, cx, cy, th, tx, ty, t1, t2;
//	th = g_ang[k] / 180 * float(M_PI);
//	cx = cosf(th);
//	cy = -sinf(th);
//	x0 = -cx * nx / 2 - cy*nx / 2 + px / 2;
//	ty = x0;
//	for (i = 0; i < nx; i++, ty += cy){
//		tx = ty;
//		for (j = 0; j < nx; j++, tx += cx){
//			ix = (int)tx;
//			if (ix < 0 || ix > px - 2)
//				continue;
//			t1 = tx - ix;
//			t2 = 1 - t1;
//			img[i * nx + j] += t1 * prj[k * px + ix + 1] + t2 * prj[k * px + ix];
//		}
//	}
//}
//void backprojection_em(float *img, float *prj, int px, int pa, int nx, int k){
//	int    i;
//	for (i = 0; i < nx*nx; i++){
//		img[i] += g_c0[k*nx*nx + i] * prj[k*px + g_cx[k*nx*nx + i] + 0];
//		img[i] += g_c1[k*nx*nx + i] * prj[k*px + g_cx[k*nx*nx + i] + 1];
//		img[i] += g_c2[k*nx*nx + i] * prj[k*px + g_cx[k*nx*nx + i] + 2];
//	}
//}
//void make_pj(float *img, float *prj, int px, int pa, int nx, float *ang, float *prjf, int paf, int mae){
//	int     i, k;
//	float   *aprj;
//	aprj = (float *)malloc(px*paf*sizeof(float));
//	//メモリー確保//
//	detection_probability(px, paf, nx, ang);
//	fprintf(stderr, "\r *** 投影像作成 *** ");
//	for (k = 0; k < pa; k++)
//		projection(img, aprj, px, pa, nx, k);
//	for (i = 0; i < pa*px; i++){
//		aprj[i + mae*px] = prj[i];
//	}
//	write_data_output("RAW_to_900.img", aprj, px*paf);
//	free(aprj);
//	//メモリー開放//
//}

/*void AART(float *img, float *prj, int px, int pa, int nx, float *ang, int it, int N, double wt){
	int     i, j, k, m1, m2, *sub;
	char    fi[50];
	float   *aprj, *rprj, *aimg;
	DWORD	t1;
	aprj = (float *)malloc(px*pa*sizeof(float));
	rprj = (float *)malloc(px*pa*sizeof(float));
	aimg = (float *)malloc(nx*nx*sizeof(float));
	sub = (int *)malloc(pa*sizeof(int));
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
		t1 = timeGetTime();
		fprintf(stderr, "\r [Layer %04d/%04d - Iteration %03d/%03d] %.3f秒経過", N + 1, g_num, i + 1, it, (float)(t1 - g_t0) / 1000);
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
		sprintf_s(fi, 32, "L%04d_I%03d_AART.img", N + 1, i + 1);
		write_data_temp(fi, img, nx*nx);
	}
	free(aprj);
	free(rprj);
	free(aimg);
}
void MART(float *img, float *prj, int px, int pa, int nx, float *ang, int it, int N){
	int     i, j, k, m1, m2, *sub;
	char    fi[50];
	float   *aprj, *rprj, *aimg;
	DWORD	t1;
	aprj = (float *)malloc(px*pa*sizeof(float));
	rprj = (float *)malloc(px*pa*sizeof(float));
	aimg = (float *)malloc(nx*nx*sizeof(float));
	sub = (int *)malloc(pa*sizeof(int));
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
		t1 = timeGetTime();
		fprintf(stderr, "\r [Layer %04d/%04d - Iteration %03d/%03d] %.3f秒経過", N + 1, g_num, i + 1, it, (float)(t1 - g_t0) / 1000);
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
		sprintf_s(fi, 32, "L%04d_I%03d_MART.img", N + 1, i + 1);
		write_data_temp(fi, img, nx*nx);
	}
	free(aprj);
	free(rprj);
	free(aimg);
}
void ASIRT(float *img, float *prj, int px, int pa, int nx, float *ang, int it, int N, double wt){
	int     i, j, k;
	char    fi[50];
	float   *aprj, *rprj, *aimg;
	DWORD	t1;
	aprj = (float *)malloc(px*pa*sizeof(float));
	rprj = (float *)malloc(px*pa*sizeof(float));
	aimg = (float *)malloc(nx*nx*sizeof(float));
	for (i = 0; i < it; i++){
		t1 = timeGetTime();
		fprintf(stderr, "\r [Layer %04d/%04d - Iteration %03d/%03d] %.3f秒経過", N + 1, g_num, i + 1, it, (float)(t1 - g_t0) / 1000);
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
			sprintf_s(fi, 32, "L%04d_I%03d_ASIRT.img", N + 1, i + 1);
			write_data_temp(fi, img, nx*nx);
		}
	}
	free(aprj);
	free(rprj);
	free(aimg);
}
void MSIRT(float *img, float *prj, int px, int pa, int nx, float *ang, int it, int N){
	int     i, j, k;
	char    fi[50];
	float   *aprj, *rprj, *aimg;
	DWORD	t1;
	aprj = (float *)malloc(px*pa*sizeof(float));
	rprj = (float *)malloc(px*pa*sizeof(float));
	aimg = (float *)malloc(nx*nx*sizeof(float));
	for (i = 0; i < it; i++){
		t1 = timeGetTime();
		fprintf(stderr, "\r [Layer %04d/%04d - Iteration %03d/%03d] %.3f秒経過", N + 1, g_num, i + 1, it, (float)(t1 - g_t0) / 1000);
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
			sprintf_s(fi, 32, "L%04d_I%03d_MSIRT.img", N + 1, i + 1);
			write_data_temp(fi, img, nx*nx);
		}
	}
	free(aprj);
	free(rprj);
	free(aimg);
}
void MLEM(float *img, float *prj, int px, int pa, int nx, float *ang, int it, int N){
	int     i, j, k;
	char    fi[50];
	float   *aprj, *rprj, *aimg;
	DWORD	t1;
	aprj = (float *)malloc(px*pa*sizeof(float));
	rprj = (float *)malloc(px*pa*sizeof(float));
	aimg = (float *)malloc(nx*nx*sizeof(float));
	for (i = 0; i < it; i++){
		t1 = timeGetTime();
		fprintf(stderr, "\r [Layer %04d/%04d - Iteration %03d/%03d] %.3f秒経過", N + 1, g_num, i + 1, it, (float)(t1 - g_t0) / 1000);
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
			sprintf_s(fi, 32, "L%04d_I%03d_MLEM.img", N + 1, i + 1);
			write_data_temp(fi, img, nx*nx);
		}
	}
	free(aprj);
	free(rprj);
	free(aimg);
}
*/
void OSEM(float *img, float *prj, int px, int pa, int nx, float *ang, int it, int N, int ss){
	int     i, j, k, m1, m2, *sub, pi, pj, bi, bj, ix, *cx;
	char    fi[50];
	float	x, y, xx, th, a, b, x05, d, si, co, cc[3], *aprj, *rprj, *aimg, *c0, *c2;
	DWORD	t1;
	aprj = (float *)malloc(px*pa*sizeof(float));
	rprj = (float *)malloc(px*pa*sizeof(float));
	aimg = (float *)malloc(nx*nx*sizeof(float));
	sub = (int *)malloc(ss*sizeof(int));
	cx = (int *)malloc(int(pa/ss)*g_nx*g_nx*sizeof(int));
	c0 = (float *)malloc(int(pa/ss)*g_nx*g_nx*sizeof(float));
	c2 = (float *)malloc(int(pa/ss)*g_nx*g_nx*sizeof(float));
	k = 0;
	for (i = 0; i < 32; i++)
		k += (ss >> i) & 1;
	if (k == 1){
		m1 = 0;
		sub[m1++] = 0;
		for (i = ss, m2 = 1; i > 1; i /= 2, m2 *= 2){
			for (j = 0; j < m2; j++)
				sub[m1++] = sub[j] + i / 2;
		}
	}
	else {
		for (i = 0; i < ss; i++)
			sub[i] = i;
	}
	for (i = 0; i < it; i++){
		for (k = 0; k < ss; k++){
			#pragma omp parallel for
			for (pi = 0; pi < pa/ss*nx*nx; pi++){
				cx[pi] = 0;
				c0[pi] = c2[pi] = 0.0;
			}
			for (j = sub[k]; j < pa; j += ss){
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
				#pragma omp parallel for private(x, y, xx, ix, pi, pj, cc, d, x05)
				for (pi = 0; pi < nx; pi++){
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
			#pragma omp parallel for private(pi, pj)
			for (j = sub[k]; j < pa; j += ss){
				for (pi = 0; pi < px; pi++)
					aprj[j*px + pi] = 0;						
				for (pj = 0; pj < nx*nx; pj++){
					aprj[j*px + cx[j/ss*nx*nx + pj] + 0] += img[pj] * c0[j/ss*nx*nx + pj];
					aprj[j*px + cx[j/ss*nx*nx + pj] + 1] += img[pj] * (1 - c0[j/ss*nx*nx + pj] - c2[j/ss*nx*nx + pj]);
					aprj[j*px + cx[j/ss*nx*nx + pj] + 2] += img[pj] * c2[j/ss*nx*nx + pj];
				}
			}
			for (j = sub[k]; j < pa; j += ss){
				#pragma omp parallel for
				for (bi = 0; bi < px; bi++){
					if ((double)aprj[j*px + bi] < 0.0001)
						rprj[j*px + bi] = prj[j*px + bi];
					else
						rprj[j*px + bi] = prj[j*px + bi] / aprj[j*px + bi];
				}
			}
			#pragma omp parallel for
			for (j = 0; j < nx*nx; j++)
				aimg[j] = 0.0;
			for (j = sub[k]; j < pa; j += ss){
				#pragma omp parallel for
				for (bj = 0; bj < nx*nx; bj++){
					aimg[bj] += rprj[j*px + cx[j/ss*nx*nx + bj] + 0] * c0[j/ss*nx*nx + bj];
					aimg[bj] += rprj[j*px + cx[j/ss*nx*nx + bj] + 1] * (1 - c0[j/ss*nx*nx + bj] - c2[j/ss*nx*nx + bj]);
					aimg[bj] += rprj[j*px + cx[j/ss*nx*nx + bj] + 2] * c2[j/ss*nx*nx + bj];
				}
			}
			#pragma omp parallel for
			for (j = 0; j < nx*nx; j++)
				img[j] *= (float)aimg[j] * ss / pa;
		}
		t1 = timeGetTime();
		fprintf(stderr, "\r [Layer %04d/%04d - Iteration %03d/%03d] %.3f秒経過", N + 1, g_st -1 + g_num, i + 1, it, (float)(t1-g_t0)/1000);
		sprintf_s(fi, 32, "L%04d_I%03d_aprj.img", N + 1, i + 1);
		write_data_temp(fi, aprj, px*pa);
		sprintf_s(fi, 32, "L%04d_I%03d_rprj.img", N + 1, i + 1);
		write_data_temp(fi, rprj, px*pa);
		sprintf_s(fi, 32, "L%04d_I%03d_aim.img", N + 1, i + 1);
		write_data_temp(fi, aimg, nx*nx);
		sprintf_s(fi, 32, "L%04d_I%03d_rec.img", N + 1, i + 1);
		write_data_temp(fi, img, nx*nx);
	}
	free(aprj);
	free(rprj);
	free(aimg);
	free(cx);
	free(c0);
	free(c2);
}
//
//void FBP(float *img, float *prj, int px, int pa, int nx, float *ang, int N){
//	int		zx = px*2;
//	float	*prz;
//	prz = (float *)malloc((unsigned long)zx*pa*sizeof(float));
//	zero_padding(prj, prz, px, pa, zx);
//	FFT_filter(prz, zx, pa);
//	int     i, j, k, ix;
//	double  x0, cx, cy, th, tx, ty, t1, t2;
//	float   *bp2;
//	fprintf(stderr, "\r *** FBP *** [Layer %04d/%04d]", N + 1, g_num);
//	for (k = 0; k < pa; k++) {
//		th = g_ang[k] / 180 * M_PI;
//		cx = cos(th);
//		cy = -sin(th);
//		x0 = -cx * nx / 2 - cy * nx / 2 + zx / 2;
//		bp2 = prz + k * zx;
//		for (i = 0, ty = x0; i < nx; i++, ty += cy) {
//			for (j = 0, tx = ty; j < nx; j++, tx += cx) {
//				ix = (int)tx;
//				if (ix < 0 || ix > zx - 2)     continue;
//				t1 = tx - ix;
//				t2 = 1 - t1;
//				img[i*nx + j] += (float)(t1 * bp2[ix + 1] + t2 * bp2[ix]);
//			}
//		}
//	}
//	for (i = 0; i < nx * nx; i++)
//		img[i] /= pa;
//	free(prz);
//}
//void zero_padding(float *prj, float *prz, int px, int pa, int zx){
//	int		i, j;
//	for (i = 0; i < zx * pa; i++)
//		prz[i] = 0;
//	for (i = 0; i < pa; i++) {
//		for (j = 0; j < px; j++) {
//			prz[i * zx + zx / 2 + j - px / 2] = prj[i * px + j];
//		}
//	}
//}
//void FFT_filter(float *prj, int px, int pa){
//	float	*xr, *xi, *si, *co;
//	unsigned short	*br;
//	int		i, j;
//	xr = (float *)malloc((unsigned long)px*sizeof(float));
//	xi = (float *)malloc((unsigned long)px*sizeof(float));
//	si = (float *)malloc((unsigned long)px*sizeof(float) / 2);
//	co = (float *)malloc((unsigned long)px*sizeof(float) / 2);
//	br = (unsigned short *)malloc((unsigned long)px*sizeof(unsigned short));
//	FFT_init(px, si, co, br);
//	for (i = 0; i < pa; i++) {
//		for (j = 0; j < px / 2; j++) { 
//			xr[j] = prj[i * px + j + px / 2];
//			xr[j + px / 2] = prj[i * px + j];
//			xi[j] = xi[j + px / 2] = 0;
//		}
//		FFT(1, px, xr, xi, si, co, br);
//		Filter(xr, px);
//		Filter(xi, px);
//		FFT(-1, px, xr, xi, si, co, br);
//		for (j = 0; j < px / 2; j++) {
//			prj[i * px + j] = xr[j + px / 2];
//			prj[i * px + j + px / 2] = xr[j];
//		}
//	}
//	free(xr);
//	free(xi);
//	free(si);
//	free(co);
//	free(br);
//}
//void FFT_init(int nx, float *si, float *co, unsigned short *brv){
//	double	d = 2*M_PI / nx;
//	int		i;
//	for (i = 0; i < nx / 4; i++) {
//		si[i] = (float)sin(d * i);
//		co[i + nx / 4] = -si[i];
//	}
//	for (i = nx / 4; i < nx / 2; i++) {
//		si[i] = (float)sin(d * i);
//		co[i - nx / 4] = si[i];
//	}
//	for (i = 0; i < nx; i++)
//		brv[i] = br(nx, (unsigned)i);
//}
//int br(int nx, unsigned nn){
//	unsigned	c, r = 0;
//	for (c = 1; c <= (unsigned)nx / 2; c <<= 1) {
//		r <<= 1;
//		if ((nn&c) != 0)
//			r++;
//	}
//	return(r);
//}
//void FFT(int ir, int nx, float *xr, float *xi, float *si, float *co, unsigned short *brv){
//// int		ir;  順変換(1)と逆変換(-1)
//	int		d = 1, g, i, j, j3, j4, k, l, ll, n1, n2 = nx;
//	float	a, b, c, s;
//	for (l = 1; l <= nx / 2; l *= 2, d += d) {
//		g = 0;
//		ll = n2;
//		n2 /= 2;
//		for (k = 1; k <= n2; k++) {
//			n1 = k - ll;
//			c = co[g];
//			s = -ir * si[g];
//			g += d;
//			for (j = ll; j <= nx; j += ll) {
//				j3 = j + n1 - 1;
//				j4 = j3 + n2;
//				a = xr[j3] - xr[j4];
//				b = xi[j3] - xi[j4];
//				xr[j3] += xr[j4];
//				xi[j3] += xi[j4];
//				xr[j4] = c * a + s * b;
//				xi[j4] = c * b - s * a;
//			}
//		}
//	}
//	bitrev(nx, xr, xi, brv);
//	if (ir == -1)
//	for (i = 0; i < nx; i++) {
//		xr[i] /= nx;
//		xi[i] /= nx;
//	}
//}
//void bitrev(int nx, float *xr, float *xi, unsigned short *brv){
//	int		i, j;
//	float	a, b;
//	for (i = 0; i < nx; i++){
//		j = brv[i];
//		if (i < j){
//			a = xr[i];
//			b = xi[i];
//			xr[i] = xr[j];
//			xi[i] = xi[j];
//			xr[j] = a;
//			xi[j] = b;
//		}
//	}
//}
//void Filter(float *xr, int px){
//	int		i;
//	double	h = M_PI / px;
//	for (i = 0; i < px / 2; i++)
//		xr[i] *= (float)(i * h);
//	for (i = px / 2; i < px; i++)
//		xr[i] *= (float)((px - i) * h);
//}