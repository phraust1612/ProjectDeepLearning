#include "training.h"
#if CUDAEXIST
__global__ void CudaTrainingThread(double *d_W, double *d_b, double *d_dLdW, double *d_dLdb)
{
	__shared__ double s[ /* size */ ];
	__shared__ bool delta[ /* size */ ];
	__shared__ double dW[ /* size */ ];
	__shared__ double db[ /* size */ ];
	int thisblock = threadIdx.x;
}

__global__ void CudaResetGradients(double *d_grad)
{
	int x = blockIdx.x;
	d_grad[x] = 0;
}

__global__ void CudaL2regularization(double *d_grad, double *d_W, double N, double lambda)
{
	int x = blockIdx.x;
	d_grad[x] = d_grad[x] / N;
	if(lambda) d_grad[x] = lambda * d_W[x];
}

__global__ void CudaOptimization(double *d_grad, double *d_W, double H)
{
	int x = blockIdx.x;
	d_W[x] = d_W[x] - (H * d_grad[x]);
}
#endif

CTraining::CTraining(CDataread* pD)
{
	pData = pD;
	N = pData->N;
	Nt = pData->Nt;
	loaded=0;
#if CUDAEXIST
	deviceProp.major = 0;
	deviceProp.minor = 0;
	cudaGetDeviceProperties(&deviceProp, cudadevice);
	cuda_err = cudaGetLastError();
	if(cuda_err != cudaSuccess) CUDAEXIST = 0;
#endif	
	alpha = 1;
	learningSize = 0;
	count = 0;
	DELTA = DELTADEFAULT;
	LAMBDA = LAMBDADEFAULT;
	H = HDEFAULT;
	MOMENTUMUPDATE = MUDEFAULT;
	Lold = 0;
	l = 0;
	L = 0;
	sizeW = 0;
	sizeb = 0;
	sizes = 0;
	automode = 'n';
	W = NULL;
	dLdW = NULL;
	b = NULL;
	dLdb = NULL;
	D = NULL;
	savefilename = NULL;
}

CTraining::~CTraining(){}

void CTraining::FreeMem()
{
	pData->FreeData();
	if(W != NULL) free(W);
	if(dLdW != NULL) free(dLdW);
	if(b != NULL) free(b);
	if(dLdb != NULL) free(dLdb);
	if(D != NULL) free(D);
}

int CTraining::WeightInit(int size, char* argv)
{
	int i;
	learningSize=size;
	savefilename = argv;
	printf("[(CONV -> ReLU) * 'A' -> POOL?] * 'B' -> (FC -> ReLU) * 'C' -> FC\n");
	printf("if B=0 : (CONV -> ReLU) * A -> (FC -> ReLU) * C -> FC\n");
	printf("input A\n>> ");
	scanf("%d", &A);
	getchar();
	printf("input B\n>> ");
	scanf("%d", &A);
	getchar();
	printf("input C\n>> ");
	scanf("%d", &A);
	getchar();
	alpha = C+1;
	
	if(B)
	{
		Size.width = (int*) malloc(sizeof(int) * ((A+1) * B +1));
		Size.height = (int*) malloc(sizeof(int) * ((A+1) * B +1));
		Size.depth = (int*) malloc(sizeof(int) * ((A+1) * B +1));
	}
	else
	{
		Size.width = (int*) malloc(sizeof(int) * (A+1));
		Size.height = (int*) malloc(sizeof(int) * (A+1));
		Size.depth = (int*) malloc(sizeof(int) * (A+1));
	}

	Size.width[0] = pData->row;
	Size.height[0] = pData->col;
	Size.depth[0] = pData->depth;
	
	// if use Pooling layer
	if(B)
	{
		for(i=0; i<(A+1)*B; i++)
		{
			printf("width_%d : %d, height_%d : %d\n", i, Size.width[i], i, Size.height[i]);
			// Pooling layer
			if((i+1)%(A+1) == 0)
			{
				while(true)
				{
					printf("input F_%d\n>> ", i);
					scanf("%d", &F[i]);
					getchar();
					printf("input S_%d\n>> ", i);
					scanf("%d", &S[i]);
					getchar();
					P[i] = 0;
					Size.width[i+1] = Size.width[i] - F[i];
					Size.height[i+1] = Size.height[i] - F[i];
					if((Size.width[i+1] % S[i]) == 0 && (Size.height[i+1] % S[i]) == 0)
					{
						Size.width[i+1] = Size.width[i+1] / S[i] + 1;
						Size.height[i+1] = Size.height[i+1] / S[i] + 1;
						break;
					}
					else printf("wrong sets...\n");
				}
				Size.depth[i+1] = Size.depth[i];
			}
			// Convolutional layer
			else
			{
				while(true)
				{
					printf("input F_%d\n>> ", i);
					scanf("%d", &F[i]);
					getchar();
					printf("input S_%d\n>> ", i);
					scanf("%d", &S[i]);
					getchar();
					printf("input P_%d\n>> ", i);
					scanf("%d", &P[i]);
					getchar();
					Size.width[i+1] = Size.width[i] - F[i] + 2*P[i];
					Size.height[i+1] = Size.height[i] - F[i] + 2*P[i];
					if((Size.width[i+1] % S[i]) == 0 && (Size.height[i+1] % S[i]) == 0)
					{
						Size.width[i+1] = Size.width[i+1] / S[i] + 1;
						Size.height[i+1] = Size.height[i+1] / S[i] + 1;
						break;
					}
					else printf("wrong sets...\n");
				}
				printf("input depth_%d\n>> ");
				scanf("%d", &Size.depth[i+1]);
				getchar();
			}
		}
	}
	// for the case if no pooling layer's used
	else
	{
		for(i=0; i<A; i++)
		{
			printf("width_%d : %d, height_%d : %d\n", i, Size.width[i], i, Size.height[i]);
			while(true)
			{
				printf("input F_%d\n>> ", i);
				scanf("%d", &F[i]);
				getchar();
				printf("input S_%d\n>> ", i);
				scanf("%d", &S[i]);
				getchar();
				printf("input P_%d\n>> ", i);
				scanf("%d", &P[i]);
				getchar();
				Size.width[i+1] = Size.width[i] - F[i] + 2*P[i];
				Size.height[i+1] = Size.height[i] - F[i] + 2*P[i];
				if((Size.width[i+1] % S[i]) == 0 && (Size.height[i+1] % S[i]) == 0)
				{
					Size.width[i+1] = Size.width[i+1] / S[i] + 1;
					Size.height[i+1] = Size.height[i+1] / S[i] + 1;
					break;
				}
				else printf("wrong sets...\n");
			}
			printf("input depth_%d\n>> ");
			scanf("%d", &Size.depth[i+1]);
			getchar();
		}
	}
	
	D = (int*) malloc(sizeof(int) * (C+2));
	D[0] = pData->D0;	// initializing D_0 = D, and D_i is dimension of i'th s
	D[alpha] = pData->M;	// initializing D_alpha = M
	// scan D_i's value from user
	for(i=1; i<alpha; i++)
	{
		printf("intput D_%d\n>> ", i);
		scanf("%d", &D[i]);
		getchar();
	}
	
	ParamAllocate();
	
	double U, V;
	U = (double) (RAND_MAX) / 2;
	V = sqrt((double) ((double) RAND_MAX * (double) (RAND_MAX+1) *(double) (2*RAND_MAX + 1)) / 6);

	srand(time(NULL));
	// W^i : D_(i+1) x D_i
	// b^i : D_(i+1)
	double sumW=0;
	for(i=0; i<sizeb; i++) b[i] = 0.01;
	for(i=0; i<sizeW; i++)
	{
		W[i] = (double)rand();
		sumW += W[i];
	}
	// choose random number according to normal distribution
	// and times sqrt(2/N)
	for(i=0; i<sizeW; i++)
		W[i] = (sqrt(2) * (W[i] - U) * (sumW - (double) sizeW * U)) / (V * sqrt(N));
				
	return 0;
}

int CTraining::WeightLoad(char* argv)
{
	int i,j,err;
	err = ERR_NONE;
	savefilename = argv;
	FILE* fpWeight = fopen(savefilename, "rb");
	
	char magic1,magic2;
	magic1 = (char) fgetc(fpWeight);
	magic2 = (char) fgetc(fpWeight);
	if(magic1 != 'P' || magic2 != 'D')
	{
		fclose(fpWeight);
		return ERR_CRACKED_FILE;
	}
	
	// read version
	int ver[2];
	fread(ver, sizeof(int), 2, fpWeight);
	if(ver[0] != 2 || ver[1] != 1)
	{
		fclose(fpWeight);
		return ERR_NOTSUPPORTEDVERSION;
	}
	
	// scan alpha - the number of layers including score layer
	fread(&alpha, sizeof(int), 1, fpWeight);
	// scan hyperparameters, etc.
	D = (int*) malloc(sizeof(int) * (alpha+1));
	fread(D, sizeof(int), alpha+1, fpWeight);
	fread(&H, sizeof(double), 1, fpWeight);
	fread(&DELTA, sizeof(double), 1, fpWeight);
	fread(&LAMBDA, sizeof(double), 1, fpWeight);
	fread(&MOMENTUMUPDATE, sizeof(double), 1, fpWeight);
	fread(&count, sizeof(int), 1, fpWeight);
	fread(&learningSize, sizeof(int), 1, fpWeight);
	if(count == learningSize) err = EXC_TRAININGDONE;
	fread(&l, sizeof(int), 1, fpWeight);
	fread(&L, sizeof(double), 1, fpWeight);
	fread(&Lold, sizeof(double), 1, fpWeight);
	
	// allocate memories
	ParamAllocate();
	
	// scan W,b etc
	fread(W, sizeof(double), sizeW, fpWeight);
	fread(b, sizeof(double), sizeb, fpWeight);
	// scan gradients
	fread(dLdW, sizeof(double), sizeW, fpWeight);
	fread(dLdb, sizeof(double), sizeb, fpWeight);
	fread(vecdW, sizeof(double), sizeW, fpWeight);
	fread(vecdb, sizeof(double), sizeb, fpWeight);
	fclose(fpWeight);
	loaded=1;
	
	return err;
}

// W^i_j,k = W[indexOfW(i,j,k)]
// return ERR_WRONGINDEX if index is out of its range
int CTraining::indexOfW(int i, int j, int k)
{
	if(i>=alpha) return ERR_WRONGINDEXW;
	if(j>=D[i+1]) return ERR_WRONGINDEXW;
	if(k>=D[i]) return ERR_WRONGINDEXW;
	
	int t, ans;
	ans = 0;
	for(t=0; t<i; t++) ans += D[t+1] * D[t];
	ans += D[i] * j;
	ans += k;
	if(ans>=sizeW) return ERR_WRONGINDEXW;
	
	return ans;
}

// b^i_j = b[indexOfb(i,j)]
// return ERR_WRONGINDEX if index is out of its range
int CTraining::indexOfb(int i, int j)
{
	if(i>=alpha) return ERR_WRONGINDEXB;
	if(j>=D[i+1]) return ERR_WRONGINDEXB;
	
	int t, ans;
	ans = 0;
	for(t=0; t<i; t++) ans += D[t+1];
	ans += j;
	if(ans>=sizeb) return ERR_WRONGINDEXB;
	
	return ans;
}

// s^i_j = s[indexOfs(i,j)]
// return ERR_WRONGINDEX if index is out of its range
int CTraining::indexOfs(int i, int j)
{
	if(i>alpha) return ERR_WRONGINDEXS;
	if(j>=D[i]) return ERR_WRONGINDEXS;
	
	int t, ans;
	ans = 0;
	for(t=0; t<i; t++) ans += D[t];
	ans += j;
	if(ans>=sizes) return ERR_WRONGINDEXS;
	
	return ans;
}

// ds^alpha_i / dW^m_j,k = dW[indexOfdW(m,i,j,k)]
// return ERR_WRONGINDEX if index is out of its range
int CTraining::indexOfdW(int m, int i, int j, int k)
{
	if(i>=D[alpha]) return ERR_WRONGINDEXDW;
	
	int t, ans;
	ans = i * sizeW;
	t = indexOfW(m,j,k);
	if(t<0) return t;
	ans += t;
	if(ans>=D[alpha] * sizeW) return ERR_WRONGINDEXDW;
	
	return ans;
}

// ds^alpha_i / dW^m_j = db[indexOfdb(m,i,j)]
// return ERR_WRONGINDEX if index is out of its range
int CTraining::indexOfdb(int m, int i, int j)
{
	if(i>=D[alpha]) return ERR_WRONGINDEXDB;
	
	int t, ans;
	ans = i * sizeb;
	t = indexOfb(m,j);
	if(t<0) return t;
	
	ans += t;
	if(ans>=D[alpha] * sizeb) return ERR_WRONGINDEXDB;
	
	return ans;
}

void CTraining::ParamAllocate()
{
	int i,j,k;
	
	sizeW = 0;
	for(i=0; i<alpha; i++) sizeW += D[i+1] * D[i];
	sizeb = 0;
	for(i=0; i<alpha; i++) sizeb += D[i+1];
	sizes = sizeb + D[0];
	
	// dLdW[indexOfW(i,j,k)] : dL / dW^i_j,k
	// dLdb[indexOfb(i,j)] : dL / db^i_j
	W = (double*) malloc(sizeof(double) * sizeW);
	b = (double*) malloc(sizeof(double) * sizeb);
	dLdW = (double*) malloc(sizeof(double) * sizeW);
	dLdb = (double*) malloc(sizeof(double) * sizeb);
	vecdW = (double*) malloc(sizeof(double) * sizeW);
	vecdb = (double*) malloc(sizeof(double) * sizeb);
	
	// initialize dL/db and dL/dW
	for(i=0; i<sizeb; i++)
	{
		dLdb[i] = 0;
		vecdb[i]=0;
	}
	for(i=0; i<sizeW; i++)
	{
		dLdW[i] = 0;
		vecdW[i]=0;
	}
}

void CTraining::Training(int threads)
{
	int i,j,k,tmp,hr,min,sec,startindexl,startindexcount;
	time_t starttime, endtime;
	double gap, Try, acc, h;
	bool cont = true;
	std::thread hThread[threads];
	// Callback function starts to catch key inputs
	Key = CKeyinter();
	Key.Start();
	endtime=0;
	ValidationParam valid;
	startindexl = l;
	startindexcount = count;
	starttime = clock();
	
#if CUDAEXIST
	// allocate device memories
	double *d_W, *d_b, *d_dLdW, *d_dLdb;
	cudaMalloc((void**)&d_W, sizeof(double) * sizeW);
	cudaMalloc((void**)&d_b, sizeof(double) * sizeb);
	cudaMalloc((void**)&d_dLdW, sizeof(double) * sizeW);
	cudaMalloc((void**)&d_dLdb, sizeof(double) * sizeb);
	
	// memcpy initial W,b from host to device
	cudaMemcpy(d_W, W, sizeof(double) * sizeW, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(double) * sizeb, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dLdW, dLdW, sizeof(double) * sizeW, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dLdb, dLdb, sizeof(double) * sizeb, cudaMemcpyHostToDevice);
#endif
	
	for(;count<learningSize && cont; count++) 
	{
		if(!loaded)
		{
			L = 0;
			l = 0;
#if CUDAEXIST
			// initialize gradient of Loss on device
			CudaResetGradients<<<sizeW, 1>>>(d_dLdW);
			CudaResetGradients<<<sizeb, 1>>>(d_dLdb);
#else
			for(i=0; i<sizeW; i++)
				dLdW[i] = 0;
			for(i=0; i<sizeb; i++)
				dLdb[i] = 0;
#endif
		}
		loaded = 0;
		
		// loop whose goal is to calculate gradient of L about each W,b
		while(l<N && cont)	// l is an index for each picture
		{
			// if callback function catched character 's'
			// save W,b parameters to a file
			tmp = Key.keysave;
			switch(tmp)
			{
				case 's':
					FileSave();
					Key.keysave = automode;
					printf("\nsave suceeded...");
					break;
				case 't':
					Key.keysave = automode;
					if(!WeightSave()) printf("\nWeight parameter save suceeded...");
					break;
				case 'c':
					printf("\nstart testing procedure...");
					acc = CheckAccuracy();
					Key.keysave = automode;
					printf("\naccuracy : %2.2lf%%!!!", acc);
					break;
				case 'h':
					ShowHelp();
					Key.keysave = automode;
					break;
				case 'p':
					Key.keysave = automode;
					// compute just proceed and I won't declare any other variables
					printf("\ncurrently %dth loop's running",count);
					acc = (double) (l - startindexl + N * (count - startindexcount));
					acc /= (double) (N * learningSize);
					printf("\nlearning procedure : %2.2lf%% done...\n", 100 * acc);
					
					// show expected time
					endtime = clock();
					gap = (double) (endtime - starttime)/(CLOCKS_PER_SEC);
					sec = (int) gap;
					hr = (int) sec/3600;
					sec -= (int) hr*3600;
					min = (int) sec/60;
					sec -= (int) min*60;
					printf("costed about %d : %d : %d time...\n", hr, min, sec);
					// acc : Try = costed : remained
					acc = (double) (l - startindexl + N * (count - startindexcount));
					Try = (double) (N - l + N * (learningSize - count - 1));
					if(acc!=0) Try /= acc;
					else break;
					sec = (int) gap * Try;
					hr = (int) sec/3600;
					sec -= (int) hr*3600;
					min = (int) sec/60;
					sec -= (int) min*60;
					printf("estimated remaining time %d : %d : %d...", hr, min, sec);
					break;
				case 'q':
					count--;
					cont = false;
					break;
				case 'n':
					if(automode != 'n') printf("\nmode off...");
					automode = 'n';
					Key.keysave = automode;
					break;
				case 'a':
					if(automode != 'a') printf("\nauto save mode...");
					automode = 'a';
					Key.keysave = automode;
					break;
				case 'g':
					if(automode != 'g') printf("\ngradient check mode...");
					automode = 'g';
					Key.keysave = automode;
					break;
				case 'm':
					if(automode != 'm') printf("\nmodify parameter mode...");
					automode = 'm';
					Key.keysave = automode;
					break;
				default:
					Key.keysave = automode;
					break;
			}
#if CUDAEXIST
			// to calculate gradient,
			// run multi-thread via gpgpu
			
			// TODO : do it
#else
			// run multi-thread via cpu
			if(cont)
			{
				for(i=0; i<threads && l+i<N; i++) hThread[i] = std::thread(&CTraining::TrainingThreadFunc, this, l+i);
				for(i=0; i<threads && l+i<N; i++) hThread[i].join();
				l += threads;
			}
#endif
		}
	
		if(cont)
		{	
#if CUDAEXIST
			// compute L2 regularization on device
			CudaL2Regularization<<<sizeW, 1>>>(d_dLdW, d_W, (double)N, LAMBDA);
			CudaL2Regularization<<<sizeb, 1>>>(d_dLdb, d_b, (double)N, 0);
			
			// and optimize it
			CudaOptimization<<<sizeW, 1>>>(d_dLdW, d_W, H);
			CudaOptimization<<<sizeb, 1>>>(d_dLdb, d_b, H);
#else
			// L2 regularization
			// and optimize next W, b according to momentum update
			// if you set MOMENTUMUPDATE as 0, it works exactly same as SGD
			L /= (double) N;
			for(i=0; i<sizeW; i++)
			{
				dLdW[i] /= (double) N;
				dLdW[i] += LAMBDA * W[i];
				L += LAMBDA * W[i] * W[i] * 0.5;
				vecdW[i] = MOMENTUMUPDATE * vecdW[i] - H * dLdW[i];
				W[i] += vecdW[i];
			}
			for(i=0; i<sizeb; i++)
			{
				dLdb[i] /= (double) N;
				vecdb[i] = MOMENTUMUPDATE * vecdb[i] - H * dLdb[i];
				b[i] += vecdb[i];
			}
			
			// show loss func and its difference with previous one
			printf("\nL = %lf", L);
			printf("\nincrement of L = %lf...", L-Lold);
				
			if(automode == 'a' && count%10 == 0)
			{
				FileSave();
				WeightSave();
			}
			if(automode == 'g')
			{
				Try = GradientCheck();
				printf("\ngradient check : %lf...", Try);
			}
			if(automode == 'm')
			{
				printf("\nto modify DELTA, input 1\n");
				printf("to modify LAMBDA, input 2\n");
				printf("to modify H, input 3\n");
				printf("to modify learning size, input 4\n");
				printf("to modify momentum update constance, input 5\n");
				printf("to continue, input 0\n>> ");
				scanf("%d", &valid);
				getchar();
				if(valid != None)
				{
					printf("input your hyperparam value\n>> ");
					scanf("%lf", &Try);
					getchar();
					SetHyperparam(valid, 0, Try);
				}
			}
			Lold = L;
			L=0;
#endif
		}
		// then retry
	}
	Key.Stop();
#if CUDAEXIST
	// copy from device to host
	cudaMemcpy(d_W, W, sizeof(double) * sizeW, cudaMemcpyDeviceToHost);
	cudaMemcpy(d_b, b, sizeof(double) * sizeb, cudaMemcpyDeviceToHost);
	cudaMemcpy(d_dLdW, dLdW, sizeof(double) * sizeW, cudaMemcpyDeviceToHost);
	cudaMemcpy(d_dLdb, dLdb, sizeof(double) * sizeb, cudaMemcpyDeviceToHost);
	
	// free cuda memories
	cudaFree(d_W);
	cudaFree(d_b);
	cudaFree(d_dLdb);
	cudaFree(d_dLdW);
#endif
}

// save alpha, D, W, b at file
// file format : 
// offset	type			description
// 0x0000	2byte			magic number : "PD"
// 0x0002	8byte			version (2 integers) 2 & 1
// 0x0010	4byte integer	alpha
// 0x0014	4byte integer	D[0]
// 0x0018	4byte integer	D[1]
// 				...
// 			4byte integer	D[alpha]
//			8byte double	H
//			8byte double	DELTA
//			8byte double	LAMBDA
//			8byte double	MOMENTUMUPDATE
//			4byte integer	count
//			4byte integer	learningsize
//			4byte int		l (do not write behind if training ended)
//			8byte double	L
//			8byte double	Lold
//			8byte double	W[0]
//				...
//			8byte double	W[sizeW-1]
//			8byte double	b[0]
//				...
//			8byte double	b[sizeb-1]
//			8byte double	dLdW[0]
//				...
//			8byte double	dLdW[sizeW-1]
//			8byte double	dLdb[0]
//				...
//			8byte double	dLdb[sizeb-1]
//			8byte double	vecdW[0]
//				...
//			8byte double	vecdW[sizeW-1]
//			8byte double	vecdb[0]
//				...
//			8byte double	vecdb[sizeb-1]
void CTraining::FileSave()
{
	int i, j;
	if(savefilename == NULL) return;
	FILE* fpResult = fopen(savefilename, "wb");
	char magic[2];
	int ver[2];
	magic[0] = 'P';
	magic[1] = 'D';
	ver[0] = 2;
	ver[1] = 1;
	fwrite(magic, sizeof(char), 2, fpResult);
	fwrite(ver, sizeof(int), 2, fpResult);
	fwrite(&alpha, sizeof(int), 1, fpResult);
	fwrite(D, sizeof(int), alpha+1, fpResult);
	fwrite(&H, sizeof(double), 1, fpResult);
	fwrite(&DELTA, sizeof(double), 1, fpResult);
	fwrite(&LAMBDA, sizeof(double), 1, fpResult);
	fwrite(&MOMENTUMUPDATE, sizeof(double), 1, fpResult);
	fwrite(&count, sizeof(int), 1, fpResult);
	fwrite(&learningSize, sizeof(int), 1, fpResult);
	fwrite(&l, sizeof(int), 1, fpResult);
	fwrite(&L, sizeof(double), 1, fpResult);
	fwrite(&Lold, sizeof(double), 1, fpResult);
	fwrite(W, sizeof(double), sizeW, fpResult);
	fwrite(b, sizeof(double), sizeb, fpResult);
	fwrite(dLdW, sizeof(double), sizeW, fpResult);
	fwrite(dLdb, sizeof(double), sizeb, fpResult);
	fwrite(vecdW, sizeof(double), sizeW, fpResult);
	fwrite(vecdb, sizeof(double), sizeb, fpResult);
	fclose(fpResult);
}

// return err if fails
int CTraining::WeightSave()
{
	int namesize = 100;
	char name[namesize];
	int cursor,i,j,k,printnum;
	FILE *fp;
	for(cursor=0; savefilename[cursor] != '.' && savefilename[cursor] != 0; cursor++)
		name[cursor] = savefilename[cursor];
//	printf("\ncursor %d",cursor);
	k = 10000; // maximum figure
	printnum = count;
	if(printnum >= 10*k) return ERR_TXTFILENAME;
	while(k>0)
	{
		j = printnum/k;
		if(j>0)
		{
			if(cursor>=namesize) return ERR_TXTFILENAME;
			name[cursor] = j+48;
			cursor++;
			printnum -= j*k;
		}
		k /= 10;
	}
	if(cursor>=namesize-4) return ERR_TXTFILENAME;
	
	for(i=0; i<alpha; i++)
	{
		name[cursor] = 'L';
		name[cursor+1] = i+48;
		name[cursor+2] = '.';
		name[cursor+3] = 't';
		name[cursor+4] = 'x';
		name[cursor+5] = 't';
		name[cursor+6] = 0;
		fp = fopen(name, "w");
		for(j=0; j<D[i+1]; j++)
		{
			for(k=0; k<D[i]; k++) fprintf(fp, "%.6lf\t", W[indexOfW(i,j,k)]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}
	name[cursor] = 'b';
	name[cursor+1] = '.';
	name[cursor+2] = 't';
	name[cursor+3] = 'x';
	name[cursor+4] = 't';
	name[cursor+5] = 0;
	fp = fopen(name, "w");
	for(i=0; i<alpha; i++)
	{
		for(j=0; j<D[i+1]; j++)
			fprintf(fp, "%.6lf\t", b[indexOfb(i,j)]);
		fprintf(fp, "\n");
	}
	fclose(fp);
	return 0;
}

// actually this function has an error mathematically
// so both analytical and numberical gradient would not match forever
// and I haven't found the right method to check gradient correctly
double CTraining::GradientCheck()
{
	int i;
	double analytical, numerical, max, diff, ans;;
	
	analytical = 0;
	for(i=0; i<sizeW; i++) analytical += dLdW[i];
	for(i=0; i<sizeb; i++) analytical += dLdb[i];
	
	numerical = 0;
	for(i=0; i<sizeW; i++) numerical += 1/vecdW[i];
	for(i=0; i<sizeb; i++) numerical += 1/vecdb[i];
	numerical *= (L-Lold);
	
	printf("\nanalytical : %lf, numerical %lf", analytical, numerical);
	
	// choose their absolute value
	if(analytical<0) analytical *= -1;
	if(numerical<0) numerical *= -1;
	
	if(analytical > numerical)
	{
		diff = analytical - numerical;
		max = analytical;
	}
	else
	{
		diff = numerical - analytical;
		max = numerical;
	}
	
	if(max == 0) return 0;
	ans = diff / max;
	return ans;
}

double CTraining::CheckAccuracy()
{
	int i, j, k, m;
	// memory allocation for s and delta
	double* s = (double*) malloc(sizeof(double) * sizes);
	bool* delta = (bool*) malloc(sizeof(bool) * sizes);
	
	int count = 0;	// number of how many images were correct
	int ans;			// temporary answer
	double highest;		// temporary score used for seeking highest score
	double proc;
	// computing score function for each test images
	for(m=0; m<Nt; m++)	// m is an index for each picture
	{
		j=0;
		// initializaing...
		for(i=0; i<D[0]; i++)
		{
			s[indexOfs(0,i)] = pData->xt[m][i];	// initializing s_0 = x_l
			delta[indexOfs(0,i)] = 1;	// initializing deltat^0_i,i = 1,
		}						// actually deltat is DxD matrix but I'll save memory
		
		// this loop is a procedure of score function
		for(i=1; i<=alpha; i++)
		{
			for(j=0; j<D[i]; j++)
			{
				s[indexOfs(i,j)] = 0;
				// st^i = W^(i-1) * deltat^(i-1) * st^(i-1) + b^(i-1)
				for(k=0; k<D[i-1]; k++)
					if(delta[indexOfs(i-1,k)])
						s[indexOfs(i,j)] += W[indexOfW(i-1,j,k)] * s[indexOfs(i-1,k)];
				s[indexOfs(i,j)] += b[indexOfb(i-1,j)];
				//deltat^i_j = 1 if st^i_j>0, 0 otherwise
				if(i==alpha) continue;	// because there is no deltat[alpha] memory allocated
				if(s[indexOfs(i,j)] > 0) delta[indexOfs(i,j)] = 1;
				else delta[indexOfs(i,j)] = 0;
			}
		}
		
		// compare with answer and calculate the accuracy
		// firstly find m'th image's highest score and its label
		ans=0;
		highest=s[indexOfs(alpha,0)];
		for(j=1; j<D[alpha]; j++)
		{
			if(s[indexOfs(alpha,j)] > highest)
			{
				ans = j;
				highest = s[indexOfs(alpha,j)];
			}
		}
		// accumulate count if ans is correct
		if(ans == pData->yt[m]) count++;
		
		proc = (double) 100 * m/Nt;
		printf("%2.2lf%%\b\b\b\b\b",proc);
		if(proc>9.995) printf("\b");
		if(proc>=99.995) printf("\b");
	}
	printf("done...");
	
	free(s);
	free(delta);
	highest = (double) (100*count)/Nt;
	return highest;
}

int CTraining::SetHyperparam(ValidationParam validateMode, int lPar, double hyperparam)
{
	switch(validateMode)
	{
	case None:
		return 0;
	case Delta:
		DELTA = hyperparam;
		return 0;
	case Lambda:
		LAMBDA = hyperparam;
		return 0;
	case LearningrateH:
		H = hyperparam;
		return 0;
	case LearningSize:
		learningSize = hyperparam;
		return 0;
	case MomentumUpdate:
		MOMENTUMUPDATE = hyperparam;
		return 0;
	default:
		return ERR_WRONG_VALID_PARAM;
	}
}

void CTraining::ShowHelp()
{
	printf("\nenter c whenever you want to check accuracy\n");
	printf("enter s whenever you want to save\n");
	printf("enter t whenever you want to save into txt files\n");
	printf("enter a if you want to save every 10 times automatically\n");
	printf("enter n if you want to stop automatically saving mode\n");
	printf("enter p whenever you want to check progress\n");
	printf("enter q whenever you want to quit the program\n");
	printf("enter h whenever you want to read this help message again...");
}

void CTraining::ConvThreadFunc()
{
	int i,j,k,a,b,c;
	double *X = (double*)malloc(sizeof(double) * sizeConv);
	double *confReLU = (double*) malloc(sizeof(double) * sizeConv);
	
	for(i=0; i<pData->D0; i++) X[i] = pData->x[index][i];

	// for the case if Pooling layer isn't used
	if(B == 0)
	{
		for(m=0; m<A; m++)
			for(a=0; a<Size.width[m+1]; a++)
			for(b=0; b<Size.height[m+1]; b++)
			for(c=0; b<Size.depth[m+1]; c++)
			{
				X[indexOfX(m+1,a,b,c)] = convb[indexOfconvb(m,c)];
				for(i=0; i<F[m]; i++)
				for(j=0; j<F[m]; j++)
				for(k=0; k<Size.depth[m]; k++)
				{
					X[indexOfX(m+1,a,b,c)] += X[indexOfX(m, a*S[m]+i, b*S[m]+j, k)] * convW[indexOfconvW(m,i,j,k)];
					if(X[indexOfX(m+1,a,b,c)] > 0) convReLU[indexOfX(m+1,a,b,c)] = 1;
					else
					{
						X[indexOfX(m+1,a,b,c)] = 0;
						convReLU[indexOfX(m+1,a,b,c)] = 0;
					}
				}
			}
	}
	else
	{
		for(m=0; m<(A+1)*B; m++)
		{
			// Pooling layer
			if((m+1)%(A+1) == 0)
			for(a=0; a<Size.width[m+1]; a++)
			for(b=0; b<Size.height[m+1]; b++)
			for(c=0; c<Size.depth[m+1]; c++)
			{
				X[indexOfX(m+1,a,b,c)] = 0;
				for(i=a*S[m]; i<a*S[m]+F[m]; i++)
				for(j=b*S[m]; j<b*S[m]+F[m]; j++)
				if(X[indexOfX(m+1,a,b,c)] < X[indexOfX(m,i,j,c)])
				{
					X[indexOfX(m+1,a,b,c)] = X[indexOfX(m,i,j,c)];
				}
			}
			
			// Conv layer
			else
			for(a=0; a<Size.width[m+1]; a++)
			for(b=0; b<Size.height[m+1]; b++)
			for(c=0; c<Size.depth[m+1]; c++)
			{
				X[indexOfX(m+1,a,b,c)] = convb[indexOfconvb(m,c)];
				for(i=0; i<F[m]; i++)
				for(j=0; j<F[m]; j++)
				for(k=0; k<Size.depth[m]; k++)
				{
					X[indexOfX(m+1,a,b,c)] += X[indexOfX(m, a*S[m]+i, b*S[m]+j, k)] * convW[indexOfconvW(m,i,j,k)];
					if(X[indexOfX(m+1,a,b,c)] > 0) convReLU[indexOfX(m+1,a,b,c)] = 1;
					else
					{
						X[indexOfX(m+1,a,b,c)] = 0;
						convReLU[indexOfX(m+1,a,b,c)] = 0;
					}
				}
			}
		}
	}
}

int CTraining::TrainingThreadFunc(int index)
{
	if(index >= N) return ERR_WRONGFCTHREADINDEX;
	
	int i,j,k,m,tmp;
	// memory allocation of s, delta, etc
	// s^(i+1) = W^i * delta^i * s^i + b^i
	double* s = (double*) malloc(sizeof(double) * sizes);
	// dW[indexOfW(m,i,j,k)] : ds^alpha_i / dW^m_j,k
	double* dW = (double*) malloc(sizeof(double) * D[alpha] * sizeW);
	// db[sizeOfb(m,i,j)] : ds^alpha_i / db^m_j
	// and this is exactly same as ds^alpha_i / ds^(m+1)_j mathematically
	double* db = (double*) malloc(sizeof(double) * D[alpha] * sizeb);
	// this delta is a deltachronical matrix, used at ReLU
	bool* delta = (bool*) malloc(sizeof(bool) * sizes);
	
	// initialize score function and delta chronicle
	for(j=0; j<D[0]; j++)
	{
		s[indexOfs(0,j)] = pData->x[index][j];	// initializing sBef = x_l
		delta[indexOfs(0,j)] = 1;	// initializing first deltaBef
	}
	
	// this loop is a procedure of score function
	for(i=0; i<alpha; i++)
	{
		for(j=0; j<D[i+1]; j++)
		{
			s[indexOfs(i+1, j)] = 0;
			// s^(i+1) = W^i * delta^i * s^i + b^i
			for(k=0; k<D[i]; k++)
				if(delta[indexOfs(i,k)])
					s[indexOfs(i+1,j)] += W[indexOfW(i,j,k)] * s[indexOfs(i,k)];
			s[indexOfs(i+1,j)] += b[indexOfb(i,j)];
			//delta^i_j = 1 if s^i_j>0, 0 otherwise
//			if(i>=alpha-1) continue;	// because there is no delta[alpha] memory allocated
			if(s[indexOfs(i+1,j)] > 0) delta[indexOfs(i+1,j)] = 1;
			else delta[indexOfs(i+1,j)] = 0;
		}
	}
	
	// initialize db and dW (for the case m=alpha-1)
	for(i=0; i<D[alpha]; i++)
	{
		for(j=0; j<D[alpha]; j++)
		{
			// ds^alpha_i / db^(alpha-1)_j = 1 if and only if i=j
			// otherwise it becomes 0
			if(i==j) db[indexOfdb(alpha-1,i,j)] = 1;
			else db[indexOfdb(alpha-1,i,j)] = 0;
			
			for(k=0; k<D[alpha-1]; k++)
			{
				if(delta[indexOfs(alpha-1,k)])
					dW[indexOfdW(alpha-1,i,j,k)] = db[indexOfdb(alpha-1,i,j)] * s[indexOfs(alpha-1,k)];
				else dW[indexOfdW(alpha-1,i,j,k)] = 0;
			}
		}
	}
	
	// calculating gradient for b,W in general
	for(m=alpha-2; m>=0; m--)
	{
		for(i=0; i<D[alpha]; i++)
		{
			for(j=0; j<D[m+1]; j++)
			{
				// compute ds^alpha_i / db^m_j
				// check up my blog for detail about how this comes
				db[indexOfdb(m,i,j)] = 0;
				if(delta[indexOfs(m+1,j)])
					for(k=0; k<D[m+2]; k++)
						db[indexOfdb(m,i,j)] += db[indexOfdb(m+1,i,k)] * W[indexOfW(m+1,k,j)];
						
				// compute ds^alpha_i / dW^m_j,k
				for(k=0; k<D[m]; k++)
				{
					if(delta[indexOfs(m,k)]) dW[indexOfdW(m,i,j,k)] = db[indexOfdb(m,i,j)] * s[indexOfs(m,k)];
					else dW[indexOfdW(m,i,j,k)] = 0;
				}
			}
		}
	}
	
	// this is a procedure of calculating loss function
	// according to SVM and its gradient about W,b
	// L_l = sig_i
	for(i=0; i<D[alpha]; i++)
	{
		if(i == pData->y[index]) continue;
		if((tmp = s[indexOfs(alpha,i)] - s[indexOfs(alpha,pData->y[index])] + DELTA) > 0)
		{
			L += tmp;
			for(m=0; m<alpha; m++)
			{
				for(j=0; j<D[m+1]; j++)
				{
					for(k=0; k<D[m]; k++)
						dLdW[indexOfW(m,j,k)] += dW[indexOfdW(m,i,j,k)] - dW[indexOfdW(m,pData->y[index],j,k)];
					dLdb[indexOfb(m,j)] += db[indexOfdb(m,i,j)] - db[indexOfdb(m,pData->y[index],j)];
				}
			}
		}
	}
	
	// free memories used at this thread
	free(dW);
	free(db);
	free(s);
	free(delta);
	
	return 0;
}
