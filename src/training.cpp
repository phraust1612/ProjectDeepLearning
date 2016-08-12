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
	DELTA = 0;
	LAMBDA = 0;
	Lold = 0;
	l = 0;
	L = 0;
	sizeW = 0;
	sizeb = 0;
	sizes = 0;
	W = NULL;
	dLdW = NULL;
	b = NULL;
	dLdb = NULL;
	D = NULL;
	H = NULL;
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
	if(H != NULL) free(H);
}

int CTraining::WeightInit(int size)
{
	int i,j,k;
	loaded = 0;
	count = 0;
	learningSize=size;
	DELTA = DELTADEFAULT;
	LAMBDA = LAMBDADEFAULT;
	printf("input alpha\n>> ");
	scanf("%d", &alpha);	// number of W's
	getchar();
	
	D = (int*) malloc(sizeof(int) * (alpha+1));
	H = (double*) malloc(sizeof(double) * alpha);
	D[0] = pData->D0;	// initializing D_0 = D, and D_i is dimension of i'th s
	D[alpha] = pData->M;	// initializing D_alpha = M
	// scan D_i's value from user
	for(i=1; i<alpha; i++)
	{
		printf("intput D_%d\n>> ", i);
		scanf("%d", &D[i]);
		getchar();
	}
	for(i=0; i<alpha; i++)
	{
		printf("intput H_%d\n>> ", i);
		scanf("%lf", &H[i]);
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
	int numW=0;
	for(i=0; i<alpha; i++)
	{
		for(j=0; j<D[i+1]; j++)
		{
			b[indexOfb(i,j)] = 0.01;
			for(k=0; k<D[i]; k++)
			{
				W[indexOfW(i,j,k)] = (double)rand();
				sumW += W[indexOfW(i,j,k)];
			}
		}
		numW += D[i+1] * D[i];
	}
	// choose random number according to normal distribution
	// and times sqrt(2/N)
	for(i=0; i<alpha; i++)
		for(j=0; j<D[i+1]; j++)
			for(k=0; k<D[i]; k++)
				W[indexOfW(i,j,k)] = (sqrt(2) * (W[indexOfW(i,j,k)] - U) * (sumW - (double) numW * U)) / (V * sqrt(N));
				
	return 0;
}

int CTraining::WeightLoad()
{
	int i,j,err;
	err = ERR_NONE;
	FILE* fpWeight = fopen("TrainedParam", "rb");
	
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
	if(ver[0] != 1 || ver[1] != 9)
	{
		fclose(fpWeight);
		return ERR_CRACKED_FILE;
	}
	
	// scan alpha - the number of layers including score layer
	fread(&alpha, sizeof(int), 1, fpWeight);
	
	// scan hyperparameters, etc.
	D = (int*) malloc(sizeof(int) * (alpha+1));
	H = (double*) malloc(sizeof(double) * alpha);
	fread(D, sizeof(int), alpha+1, fpWeight);
	fread(H, sizeof(double), alpha, fpWeight);
	
	fread(&DELTA, sizeof(double), 1, fpWeight);
	fread(&LAMBDA, sizeof(double), 1, fpWeight);
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
	if(ans>=sizeW) return ERR_WRONGINDEXDW;
	
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
	if(ans>=sizeb) return ERR_WRONGINDEXDB;
	
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
	
	// initialize dL/db and dL/dW
	for(i=0; i<alpha; i++)
	{
		for(j=0; j<D[i+1]; j++)
		{
			dLdb[indexOfb(i,j)] = 0;
			for(k=0; k<D[i]; k++) dLdW[indexOfW(i,j,k)] = 0;
		}
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
			for(i=0; i<alpha && cont; i++)
			{
				for(j=0; j<D[i+1]; j++)
				{
					dLdb[indexOfb(i,j)] = 0;
					for(k=0; k<D[i]; k++) dLdW[indexOfW(i,j,k)] = 0;
				}
			}
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
					Key.keysave = 'n';
					printf("\nsave suceeded...");
					break;
				case 'c':
					printf("\nstart testing procedure...");
					acc = CheckAccuracy();
					Key.keysave = 'n';
					printf("\naccuracy : %2.2lf%%!!!", acc);
					break;
				case 'h':
					ShowHelp();
					Key.keysave = 'n';
					break;
				case 'p':
					Key.keysave = 'n';
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
					l -= threads;
					cont = false;
					break;
				default:
					Key.keysave = 'n';
					break;
			}
#if CUDAEXIST
			// to calculate gradient,
			// run multi-thread via gpgpu
			
			// TODO : do it
#else
			// run multi-thread via cpu
			for(i=0; i<threads; i++) hThread[i] = std::thread(&CTraining::TrainingThreadFunc, this, l+i, 0);
			for(i=0; i<threads; i++) hThread[i].join();
			l += threads;
#endif
		}
		
#if CUDAEXIST
		// compute L2 regularization on device
		CudaL2Regularization<<<sizeW, 1>>>(d_dLdW, d_W, (double)N, LAMBDA);
		CudaL2Regularization<<<sizeb, 1>>>(d_dLdb, d_b, (double)N, 0);
		
		// and optimize it
		if(cont)
		{
			CudaOptimization<<<sizeW, 1>>>(d_dLdW, d_W, H);
			CudaOptimization<<<sizeb, 1>>>(d_dLdb, d_b, H);
		}
#else
		//i = target;
		// L2 regularization
		L /= (double) N;
		/* compute only target layer */
		for(i=0; i<alpha; i++)
		{
			for(j=0; j<D[i+1]; j++)
			{
				for(k=0; k<D[i]; k++)
				{
					dLdW[indexOfW(i,j,k)] /= (double) N;
					dLdW[indexOfW(i,j,k)] += LAMBDA * W[indexOfW(i,j,k)];
					L += (double) LAMBDA * LAMBDA * 0.5;
				}
				dLdb[indexOfb(i,j)] /= (double) N;
			}
		}
		// calculate gradient of L is done...
		
		// optimizing next W, b according to SGD
		if(cont)
		{
		/* compute only target layer */
			for(i=0; i<alpha; i++)
			{
				// optimize each layer's weight individually
				for(j=0; j<D[i+1]; j++)
				{
					for(k=0; k<D[i]; k++) W[indexOfW(i,j,k)] -= (H[i] * dLdW[indexOfW(i,j,k)]);
					b[indexOfb(i,j)] -= (H[i] * dLdb[indexOfb(i,j)]);
				}
			}
			
			// show loss func and its difference
			printf("\nL = %lf", L);
			printf("\nincrement of L = %lf...", L-Lold);
			Lold = L;
		}
#endif
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
// 0x0002	8byte			version (2 integers) 1 & 9
// 0x0010	4byte integer	alpha
// 0x0014	4byte integer	D[0]
// 0x0018	4byte integer	D[1]
// 				...
// 			4byte integer	D[alpha]
//			8byte double	H[0]
//				...
//			8byte double	H[alpha-1]
//			8byte double	DELTA
//			8byte double	LAMBDA
//			4byte integer	count
//			4byte integer	learningsize
//			4byte int		l (do not write behind if training ended)
//			8byte double	L
//			8byte double	Lold
//			8byte double	W[0][0][0]
//				...
//			8byte double	W[alpha-1][D[alpha]][D[alpha-1]]
//			8byte double	b[0][0]
//				...
//			8byte double	dLdW[0][0][0]
//				...
//			8byte double	dLdW[alpha-1][D[alpha]][D[alpha-1]]
//			8byte double	dLdb[0][0]
//				...
void CTraining::FileSave()
{
	int i, j;
	FILE* fpResult = fopen("TrainedParam", "wb");
	char magic[2];
	int ver[2];
	magic[0] = 'P';
	magic[1] = 'D';
	ver[0] = 1;
	ver[1] = 9;
	fwrite(magic, sizeof(char), 2, fpResult);
	fwrite(ver, sizeof(int), 2, fpResult);
	fwrite(&alpha, sizeof(int), 1, fpResult);
	fwrite(D, sizeof(int), alpha+1, fpResult);
	fwrite(H, sizeof(double), alpha, fpResult);
	fwrite(&DELTA, sizeof(double), 1, fpResult);
	fwrite(&LAMBDA, sizeof(double), 1, fpResult);
	fwrite(&count, sizeof(int), 1, fpResult);
	fwrite(&learningSize, sizeof(int), 1, fpResult);
	fwrite(&l, sizeof(int), 1, fpResult);
	fwrite(&L, sizeof(double), 1, fpResult);
	fwrite(&Lold, sizeof(double), 1, fpResult);
	fwrite(W, sizeof(double), sizeW, fpResult);
	fwrite(b, sizeof(double), sizeb, fpResult);
	fwrite(dLdW, sizeof(double), sizeW, fpResult);
	fwrite(dLdb, sizeof(double), sizeb, fpResult);
	fclose(fpResult);
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
		if(lPar<0) return ERR_WRONG_VALID_PARAM;
		if(lPar>=alpha) return ERR_WRONG_VALID_PARAM;
		H[lPar] = hyperparam;
		return 0;
	case LearningSize:
		learningSize = hyperparam;
		return 0;
	default:
		return ERR_WRONG_VALID_PARAM;
	}
}

void CTraining::ShowHelp()
{
	printf("\nenter c whenever you want to check accuracy\n");
	printf("enter s whenever you want to save\n");
	printf("enter p whenever you want to check progress\n");
	printf("enter q whenever you want to quit the program\n");
	printf("enter h whenever you want to read this help message again...");
}

void CTraining::TrainingThreadFunc(int index, int targetlayer)
{
	int i,j,k,m,tmp;
	// memory allocation of s, delta, etc
	// s^(i+1) = W^i * delta^i * s^i + b^i
	double* s = (double*) malloc(sizeof(double) * sizes);
	// dW[indexOfW(m,i,j,k)] : ds^alpha_i / dW^m_j,k
	double* dW = (double*) malloc(sizeof(double) * sizeW);
	// db[sizeOfb(m,i,j)] : ds^alpha_i / db^m_j
	// and this is exactly same as ds^alpha_i / ds^(m+1)_j mathematically
	double* db = (double*) malloc(sizeof(double) * sizeb);
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
				if(delta[indexOfs(alpha-1,k)])
					dW[indexOfdW(alpha-1,i,j,k)] = db[indexOfdb(alpha-1,i,j)] * s[indexOfs(alpha-1,k)];
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
}
