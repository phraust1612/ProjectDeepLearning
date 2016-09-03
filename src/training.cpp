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

// constructor : resets member parameters
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
	A = 0;
	B = 0;
	C = 0;
	alpha = 1;
	beta = 0;
	learningSize = 0;
	count = 0;
	DELTA = DELTADEFAULT;
	LAMBDA = LAMBDADEFAULT;
	H = HDEFAULT;
	MOMENTUMUPDATE = MUDEFAULT;
	Lold = 0;
	l = 0;
	L = 0;
	sizeW = NULL;
	sizeb = NULL;
	sizes = NULL;
	sizeConvW = NULL;
	sizeConvb = NULL;
	sizeConvX = NULL;
	sizePool = NULL;
	automode = 'n';
	W = NULL;
	b = NULL;
	dLdW = NULL;
	dLdb = NULL;
	vecdW = NULL;
	vecdb = NULL;
	ConvW = NULL;
	Convb = NULL;
	ConvdLdW = NULL;
	ConvdLdb = NULL;
	vecConvdW = NULL;
	vecConvdb = NULL;
	width = NULL;
	height = NULL;
	depth = NULL;
	D = NULL;
	F = NULL;
	S = NULL;
	P = NULL;
	savefilename = NULL;
}

// destructor does nothing but please use FreeMem() instead
CTraining::~CTraining(){}

// free allocated memories
// please call this after end Training() returns
void CTraining::FreeMem()
{
	pData->FreeData();
	if(W != NULL) free(W);
	if(b != NULL) free(b);
	if(dLdW != NULL) free(dLdW);
	if(dLdb != NULL) free(dLdb);
	if(vecdW != NULL) free(vecdW);
	if(vecdb != NULL) free(vecdb);
	if(ConvW != NULL) free(ConvW);
	if(Convb != NULL) free(Convb);
	if(ConvdLdW != NULL) free(ConvdLdW);
	if(ConvdLdb != NULL) free(ConvdLdb);
	if(vecConvdW != NULL) free(vecConvdW);
	if(vecConvdb != NULL) free(vecConvdb);
	if(width != NULL) free(width);
	if(height != NULL) free(height);
	if(depth != NULL) free(depth);
	if(sizeW != NULL) free(sizeW);
	if(sizeb != NULL) free(sizeb);
	if(sizes != NULL) free(sizes);
	if(sizeConvW != NULL) free(sizeConvW);
	if(sizeConvb != NULL) free(sizeConvb);
	if(sizeConvX != NULL) free(sizeConvX);
	if(sizePool != NULL) free(sizePool);
	if(D != NULL) free(D);
	if(F != NULL) free(F);
	if(S != NULL) free(S);
	if(P != NULL) free(P);
}


// Set learning outline and initiate weights
// each weight parameter gets random value following normal distribution
int CTraining::WeightInit(int size, char* argv)
{
	int i,j;
	learningSize=size;
	savefilename = argv;
	printf("[(CONV -> ReLU) * A -> POOL?] * B -> (FC -> ReLU) * C -> FC\n");
	printf("if B=0 : (CONV -> ReLU) * A -> (FC -> ReLU) * C -> FC\n");
	printf("input A\n>> ");
	scanf("%d", &A);
	getchar();
	printf("input B\n>> ");
	scanf("%d", &B);
	getchar();
	printf("input C\n>> ");
	scanf("%d", &C);
	getchar();
	alpha = C+1;
	if(B) beta = (A+1)*B;
	else beta = A;
	
	width = (int*) malloc(sizeof(int) * (beta+1));
	height = (int*) malloc(sizeof(int) * (beta+1));
	depth = (int*) malloc(sizeof(int) * (beta+1));
	F = (int*) malloc(sizeof(int) * beta);
	S = (int*) malloc(sizeof(int) * beta);
	P = (int*) malloc(sizeof(int) * beta);

	width[0] = pData->row;
	height[0] = pData->col;
	depth[0] = pData->depth;
	
	for(i=0; i<beta; i++)
	{
		printf("width_%d : %d, height_%d : %d\n", i, width[i], i, height[i]);
		// Pooling layer
		if((B>0) && !((i+1)%(A+1)))
		{
			while(true)
			{
				printf("<Pooling Layer>\n");
				printf("input F_%d\n>> ", i);
				scanf("%d", &F[i]);
				getchar();
				printf("input S_%d\n>> ", i);
				scanf("%d", &S[i]);
				getchar();
				P[i] = 0;
				width[i+1] = width[i] - F[i];
				height[i+1] = height[i] - F[i];
				if((width[i+1] % S[i]) == 0 && (height[i+1] % S[i]) == 0)
				{
					width[i+1] = width[i+1] / S[i] + 1;
					height[i+1] = height[i+1] / S[i] + 1;
					break;
				}
				else printf("wrong sets...\n");
			}
			depth[i+1] = depth[i];
		}
		// Convolutional layer
		else
		{
			while(true)
			{
				printf("<Convolutional Layer>\n");
				printf("input F_%d\n>> ", i);
				scanf("%d", &F[i]);
				getchar();
				printf("input S_%d\n>> ", i);
				scanf("%d", &S[i]);
				getchar();
				printf("input P_%d\n>> ", i);
				scanf("%d", &P[i]);
				getchar();
				width[i+1] = width[i] - F[i] + 2*P[i];
				height[i+1] = height[i] - F[i] + 2*P[i];
				if((width[i+1] % S[i]) == 0 && (height[i+1] % S[i]) == 0)
				{
					width[i+1] = width[i+1] / S[i] + 1;
					height[i+1] = height[i+1] / S[i] + 1;
					break;
				}
				else printf("wrong sets...\n");
			}
			printf("input depth_%d\n>> ",i+1);
			scanf("%d", &depth[i+1]);
			getchar();
		}
	}
	
	D = (int*) malloc(sizeof(int) * (alpha+1));
	D[0] = width[beta] * height[beta] * depth[beta];
	D[alpha] = pData->M;	// initializing D_alpha = M
	// scan D_i's value from user
	for(i=1; i<alpha; i++)
	{
		printf("intput D_%d\n>> ", i);
		scanf("%d", &D[i]);
		getchar();
	}
	
	ParamAllocate();
	
	double U, V, sumW;
	int numW;
	U = (double) (RAND_MAX) / 2;
	V = sqrt((double) ((double) RAND_MAX * (double) (RAND_MAX+1) *(double) (2*RAND_MAX + 1)) / 6);

	srand(time(NULL));
	// W^i : D_(i+1) x D_i
	// b^i : D_(i+1)
	for(i=0; i<sizeb[alpha]; i++) b[i] = 0.01;
	for(i=0; i<sizeConvb[beta]; i++) b[i] = 0.01;
	// choose random number according to normal distribution
	// and times sqrt(2/N)
	for(i=0; i<alpha; i++)
	{
		sumW = 0;
		numW = 0;
		for(j=indexOfW(i,0,0); j<indexOfW(i+1,0,0); j++)
		{
			W[j] = (double)rand();
			sumW += W[j];
			numW++;
		}
		for(j=indexOfW(i,0,0); j<indexOfW(i+1,0,0); j++)
			W[j] = (sqrt(2) * (W[j] - U) * (sumW - (double) sizeW[alpha] * U)) / (V * sqrt(numW));
	}
	for(i=0; i<beta; i++)
	{
		sumW = 0;
		numW = 0;
		for(j=indexOfConvW(i,0,0,0,0); j<indexOfConvW(i+1,0,0,0,0); j++)
		{
			ConvW[j] = (double)rand();
			sumW += ConvW[j];
			numW++;
		}
		for(j=indexOfConvW(i,0,0,0,0); j<indexOfConvW(i+1,0,0,0,0); j++)
			ConvW[j] = (sqrt(2) * (ConvW[j] - U) * (sumW - (double) sizeW[alpha] * U)) / (V * sqrt(numW));
	}
				
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
	if(ver[0] == 2 && ver[1] == 1)
	{
		printf("%s loaded ver 2.1\n",savefilename);
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
		A = 0;
		B = 0;
		C = alpha-1;
		beta = 0;
		
		width = (int*) malloc(sizeof(int) * (beta+1));
		height = (int*) malloc(sizeof(int) * (beta+1));
		depth = (int*) malloc(sizeof(int) * (beta+1));
		
		width[0] = pData->row;
		height[0] = pData->col;
		depth[0] = pData->depth;
		
		// allocate memories
		ParamAllocate();
		
		// scan W,b etc
		fread(W, sizeof(double), sizeW[alpha], fpWeight);
		fread(b, sizeof(double), sizeb[alpha], fpWeight);
		// scan gradients
		fread(dLdW, sizeof(double), sizeW[alpha], fpWeight);
		fread(dLdb, sizeof(double), sizeb[alpha], fpWeight);
		fread(vecdW, sizeof(double), sizeW[alpha], fpWeight);
		fread(vecdb, sizeof(double), sizeb[alpha], fpWeight);
	}
	else if(ver[0] == 2 && ver[1] == 2)
	{
		fread(&A, sizeof(int), 1, fpWeight);
		fread(&B, sizeof(int), 1, fpWeight);
		fread(&C, sizeof(int), 1, fpWeight);
		alpha = C+1;
		if(B) beta = (A+1)*B;
		else beta = A;
		
		D = (int*) malloc(sizeof(int) * (alpha+1));
		F = (int*) malloc(sizeof(int) * beta);
		S = (int*) malloc(sizeof(int) * beta);
		P = (int*) malloc(sizeof(int) * beta);
		width = (int*) malloc(sizeof(int) * (beta+1));
		height = (int*) malloc(sizeof(int) * (beta+1));
		depth = (int*) malloc(sizeof(int) * (beta+1));
		
		width[0] = pData->row;
		height[0] = pData->col;
		depth[0] = pData->depth;
		
		fread(D, sizeof(int), alpha+1, fpWeight);
		fread(F, sizeof(int), beta, fpWeight);
		fread(S, sizeof(int), beta, fpWeight);
		fread(P, sizeof(int), beta, fpWeight);
		fread(depth+1, sizeof(int), beta, fpWeight);
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
		
		for(i=0; i<beta; i++)
		{
			width[i+1] = width[i] - F[i] + 2*P[i];
			height[i+1] = height[i] - F[i] + 2*P[i];
			if((width[i+1] % S[i]) == 0 && (height[i+1] % S[i]) == 0)
			{
				width[i+1] = width[i+1] / S[i] + 1;
				height[i+1] = height[i+1] / S[i] + 1;
			}
			else err = ERR_UNDIVISIBLESTRIDE;
		}
		
		// allocate memories
		ParamAllocate();
		
		// scan W,b etc
		fread(W, sizeof(double), sizeW[alpha], fpWeight);
		fread(b, sizeof(double), sizeb[alpha], fpWeight);
		fread(ConvW, sizeof(double), sizeW[beta], fpWeight);
		fread(Convb, sizeof(double), sizeb[beta], fpWeight);
		// scan gradients
		fread(dLdW, sizeof(double), sizeW[alpha], fpWeight);
		fread(dLdb, sizeof(double), sizeb[alpha], fpWeight);
		fread(vecdW, sizeof(double), sizeW[alpha], fpWeight);
		fread(vecdb, sizeof(double), sizeb[alpha], fpWeight);
		fread(ConvdLdW, sizeof(double), sizeW[beta], fpWeight);
		fread(ConvdLdb, sizeof(double), sizeb[beta], fpWeight);
		fread(vecConvdW, sizeof(double), sizeW[beta], fpWeight);
		fread(vecConvdb, sizeof(double), sizeb[beta], fpWeight);
	}
	else err = ERR_NOTSUPPORTEDVERSION;
	
	fclose(fpWeight);
	loaded=1;
	return err;
}

// W^i_j,k = W[indexOfW(i,j,k)]
// return ERR_WRONGINDEX if index is out of its range
int CTraining::indexOfW(int i, int j, int k)
{
	if(j>=D[i+1]) return ERR_WRONGINDEXW;
	if(k>=D[i]) return ERR_WRONGINDEXW;
	
	int t, ans;
	ans = sizeW[i];
	ans += D[i] * j;
	ans += k;
	if(ans>=sizeW[alpha]) return ERR_WRONGINDEXW;
	
	return ans;
}

// b^i_j = b[indexOfb(i,j)]
// return ERR_WRONGINDEX if index is out of its range
int CTraining::indexOfb(int i, int j)
{
	if(i>=alpha) return ERR_WRONGINDEXB;
	if(j>=D[i+1]) return ERR_WRONGINDEXB;
	
	int t, ans;
	ans = sizeb[i];
	ans += j;
	if(ans>=sizeb[alpha]) return ERR_WRONGINDEXB;
	
	return ans;
}

// s^i_j = s[indexOfs(i,j)]
// return ERR_WRONGINDEX if index is out of its range
int CTraining::indexOfs(int i, int j)
{
	if(i>alpha) return ERR_WRONGINDEXS;
	if(j>=D[i]) return ERR_WRONGINDEXS;
	
	int t, ans;
	ans = sizes[i];
	ans += j;
	if(ans>=sizes[alpha+1]) return ERR_WRONGINDEXS;
	
	return ans;
}

// ds^alpha_i / dW^m_j,k = dW[indexOfdW(m,i,j,k)]
// return ERR_WRONGINDEX if index is out of its range
int CTraining::indexOfdW(int m, int i, int j, int k)
{
	if(i>=D[alpha]) return ERR_WRONGINDEXDW;
	
	int t, ans;
	ans = i * sizeW[alpha];
	t = indexOfW(m,j,k);
	if(t<0) return t;
	ans += t;
	if(ans>=D[alpha] * sizeW[alpha]) return ERR_WRONGINDEXDW;
	
	return ans;
}

// ds^alpha_i / dW^m_j = db[indexOfdb(m,i,j)]
// return ERR_WRONGINDEX if index is out of its range
int CTraining::indexOfdb(int m, int i, int j)
{
	if(i>=D[alpha]) return ERR_WRONGINDEXDB;
	
	int t, ans;
	ans = i * sizeb[alpha];
	t = indexOfb(m,j);
	if(t<0) return t;
	
	ans += t;
	if(ans>=D[alpha] * sizeb[alpha]) return ERR_WRONGINDEXDB;
	
	return ans;
}

// return sizeConvX[u] + width[u]*height[u]*k + width[u]*j + i;
// which doesn't consider about zero-padding yet.
int CTraining::indexOfConvX(int u, int i, int j, int k)
{
	int ans = height[u]*k + j;
	ans *= width[u];
	ans += i + sizeConvX[u];
	return ans;
}

// return sizeConvW[u] + F[u]*F[u]*depth[u]*v + F[u]*F[u]*k + F[u]*j + i
int CTraining::indexOfConvW(int u, int v, int i, int j, int k)
{
	int ans = depth[u] * v + k;
	ans *= F[u];
	ans += j;
	ans *= F[u];
	ans += i + sizeConvW[u];
	return ans;
}

int CTraining::indexOfConvb(int u, int v)
{
	return sizeConvb[u] + v;
}

int CTraining::indexOfPool(int m, int i, int j, int k)
{
	int ans, t;
	t = m/(A+1);
	ans = height[m]*k + j;
	ans *= width[m];
	ans += i + sizePool[t];
	return ans;
}

int CTraining::indexOfConvdX(int u, int m, int i, int j, int k)
{
	int ans = u * sizeConvX[beta+1];
	ans += indexOfConvX(m,i,j,k);
	return ans;
}

int CTraining::indexOfConvdW(int u, int m, int v, int i, int j, int k)
{
	int ans = u * sizeConvW[beta];
	ans += indexOfConvW(m,v,i,j,k);
	return ans;
}

int CTraining::indexOfConvdb(int u, int m, int v)
{
	int ans = u * sizeConvb[beta];
	ans += indexOfConvb(m,v);
	return ans;
}

void CTraining::ParamAllocate()
{
	int i,j,k;
	
	sizeW = (int*) malloc(sizeof(int) * (alpha+1));
	sizeb = (int*) malloc(sizeof(int) * (alpha+1));
	sizes = (int*) malloc(sizeof(int) * (alpha+2));
	sizeConvX = (int*) malloc(sizeof(int) * (beta+2));
	sizeConvW = (int*) malloc(sizeof(int) * (beta+1));
	sizeConvb = (int*) malloc(sizeof(int) * (beta+1));
	sizePool = (int*) malloc(sizeof(int) * (B+1));
	
	sizeW[0] = 0;
	sizeb[0] = 0;
	sizes[0] = 0;
	for(i=0; i<alpha; i++)
	{
		sizeW[i+1] = sizeW[i] + D[i+1]*D[i];
		sizeb[i+1] = sizeb[i] + D[i+1];
		sizes[i+1] = sizes[i] + D[i];
	}
	sizes[alpha+1] = sizes[alpha] + D[alpha];
	
	sizeConvX[0] = 0;
	sizeConvW[0] = 0;
	sizeConvb[0] = 0;
	sizePool[0] = 0;
	for(i=0; i<beta; i++)
	{
		sizeConvX[i+1] = sizeConvX[i] + width[i] * height[i] * depth[i];
		sizeConvW[i+1] = sizeConvW[i];
		sizeConvb[i+1] = sizeConvb[i];
		// conv layer
		if(!B || ((i+1)%(A+1) > 0))
		{
			sizeConvW[i+1] += F[i]*F[i]*depth[i]*depth[i+1];
			sizeConvb[i+1] += depth[i+1];
		}
		else
		{
			j = (i+1)/(A+1);
			sizePool[j] = sizePool[j-1] + width[i+1] * height[i+1] * depth[i+1];
		}
	}
	sizeConvX[beta+1] = sizeConvX[beta] + width[beta]*height[beta]*depth[beta];
	
	// dLdW[indexOfW(i,j,k)] : dL / dW^i_j,k
	// dLdb[indexOfb(i,j)] : dL / db^i_j
	W = (double*) malloc(sizeof(double) * sizeW[alpha]);
	b = (double*) malloc(sizeof(double) * sizeb[alpha]);
	dLdW = (double*) malloc(sizeof(double) * sizeW[alpha]);
	dLdb = (double*) malloc(sizeof(double) * sizeb[alpha]);
	vecdW = (double*) malloc(sizeof(double) * sizeW[alpha]);
	vecdb = (double*) malloc(sizeof(double) * sizeb[alpha]);
	ConvW = (double*) malloc(sizeof(double) * sizeConvW[beta]);
	Convb = (double*) malloc(sizeof(double) * sizeConvb[beta]);
	ConvdLdW = (double*) malloc(sizeof(double) * sizeConvW[beta]);
	ConvdLdb = (double*) malloc(sizeof(double) * sizeConvb[beta]);
	vecConvdW = (double*) malloc(sizeof(double) * sizeConvW[beta]);
	vecConvdb = (double*) malloc(sizeof(double) * sizeConvb[beta]);
	
	// initialize dL/db and dL/dW
	for(i=0; i<sizeb[alpha]; i++)
	{
		dLdb[i] = 0;
		vecdb[i]=0;
	}
	for(i=0; i<sizeW[alpha]; i++)
	{
		dLdW[i] = 0;
		vecdW[i]=0;
	}
	for(i=0; i<sizeConvW[beta]; i++)
	{
		ConvdLdW[i] = 0;
		vecConvdW[i] = 0;
	}
	for(i=0; i<sizeConvb[beta]; i++)
	{
		ConvdLdb[i] = 0;
		vecConvdb[i] = 0;
	}
}

void CTraining::Training(int threads)
{
	int i,j,k,tmp,hr,min,sec,startindexl,startindexcount;
	time_t starttime, endtime;
	double gap, Try, acc, h;
	bool cont = true;
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
	cudaMalloc((void**)&d_W, sizeof(double) * sizeW[alpha]);
	cudaMalloc((void**)&d_b, sizeof(double) * sizeb[alpha]);
	cudaMalloc((void**)&d_dLdW, sizeof(double) * sizeW[alpha]);
	cudaMalloc((void**)&d_dLdb, sizeof(double) * sizeb[alpha]);
	
	// memcpy initial W,b from host to device
	cudaMemcpy(d_W, W, sizeof(double) * sizeW[alpha], cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(double) * sizeb[alpha], cudaMemcpyHostToDevice);
	cudaMemcpy(d_dLdW, dLdW, sizeof(double) * sizeW[alpha], cudaMemcpyHostToDevice);
	cudaMemcpy(d_dLdb, dLdb, sizeof(double) * sizeb[alpha], cudaMemcpyHostToDevice);
#endif
	
	for(;count<learningSize && cont; count++) 
	{
		if(!loaded)
		{
			L = 0;
			l = 0;
#if CUDAEXIST
			// initialize gradient of Loss on device
			CudaResetGradients<<<sizeW[alpha], 1>>>(d_dLdW);
			CudaResetGradients<<<sizeb[alpha], 1>>>(d_dLdb);
#else
			for(i=0; i<sizeW[alpha]; i++) dLdW[i] = 0;
			for(i=0; i<sizeb[alpha]; i++) dLdb[i] = 0;
			for(i=0; i<sizeConvW[beta]; i++) ConvdLdW[i] = 0;
			for(i=0; i<sizeConvb[beta]; i++) ConvdLdb[i] = 0;
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
					Key.keysave = automode;
					acc = CheckAccuracy(threads);
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
					break;/*
				case 'm':
					if(automode != 'm') printf("\nmodify parameter mode...");
					automode = 'm';
					Key.keysave = automode;
					break;*/
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
				std::thread* hThread = new std::thread[threads];
				for(i=0; i<threads && l+i<N; i++) hThread[i] = std::thread(&CTraining::CNNThreadFunc, this, l+i);
				for(i=0; i<threads && l+i<N; i++) hThread[i].join();
				l += threads;
				delete [] hThread;
			}
#endif
		}
	
		if(cont)
		{	
#if CUDAEXIST
			// compute L2 regularization on device
			CudaL2Regularization<<<sizeW[alpha], 1>>>(d_dLdW, d_W, (double)N, LAMBDA);
			CudaL2Regularization<<<sizeb[alpha], 1>>>(d_dLdb, d_b, (double)N, 0);
			
			// and optimize it
			CudaOptimization<<<sizeW[alpha], 1>>>(d_dLdW, d_W, H);
			CudaOptimization<<<sizeb[alpha], 1>>>(d_dLdb, d_b, H);
#else
			// L2 regularization
			// and optimize next W, b according to momentum update
			// if you set MOMENTUMUPDATE as 0, it works exactly same as SGD
			L /= (double) N;
			for(i=0; i<sizeW[alpha]; i++)
			{
				dLdW[i] /= (double) N;
				dLdW[i] += LAMBDA * W[i];
				L += LAMBDA * W[i] * W[i] * 0.5;
				vecdW[i] = MOMENTUMUPDATE * vecdW[i] - H * dLdW[i];
				W[i] += vecdW[i];
			}
			for(i=0; i<sizeb[alpha]; i++)
			{
				dLdb[i] /= (double) N;
				vecdb[i] = MOMENTUMUPDATE * vecdb[i] - H * dLdb[i];
				b[i] += vecdb[i];
			}
			for(i=0; i<sizeConvW[beta]; i++)
			{
				ConvdLdW[i] /= (double) N;
				ConvdLdW[i] += LAMBDA * ConvW[i];
				L += LAMBDA * ConvW[i] * ConvW[i] * 0.5;
				vecConvdW[i] = MOMENTUMUPDATE * vecConvdW[i] - H * ConvdLdW[i];
				ConvW[i] += vecConvdW[i];
			}
			for(i=0; i<sizeConvb[beta]; i++)
			{
				ConvdLdb[i] /= (double) N;
				vecConvdb[i] = MOMENTUMUPDATE * vecConvdb[i] - H * ConvdLdb[i];
				Convb[i] += vecConvdb[i];
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
			}/*
			if(automode == 'm')
			{
				Key.Stop();
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
				Key.Start();
				Key.keysave = 'n';
				automode = 'n';
			}*/
			Lold = L;
			L=0;
#endif
		}
		// then retry
	}
	Key.Stop();
#if CUDAEXIST
	// copy from device to host
	cudaMemcpy(d_W, W, sizeof(double) * sizeW[alpha], cudaMemcpyDeviceToHost);
	cudaMemcpy(d_b, b, sizeof(double) * sizeb[alpha], cudaMemcpyDeviceToHost);
	cudaMemcpy(d_dLdW, dLdW, sizeof(double) * sizeW[alpha], cudaMemcpyDeviceToHost);
	cudaMemcpy(d_dLdb, dLdb, sizeof(double) * sizeb[alpha], cudaMemcpyDeviceToHost);
	
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
// 0x0002	8byte			version (2 integers) 2 & 2
// 0x0010	4byte integer	A
// 0x0010	4byte integer	B
// 0x0010	4byte integer	C
// 0x0014	4byte integer	D[0]
// 0x0018	4byte integer	D[1]
// 				...
// 			4byte integer	D[alpha]
//			4byte integer	F[0]
//			4byte integer	F[1]
//				...
//			4byte integer	F[beta-1]
//			4byte integer	S[0]
//			4byte integer	S[1]
//				...
//			4byte integer	S[beta-1]
//			4byte integer	P[0]
//			4byte integer	P[1]
//				...
//			4byte integer	P[beta-1]
//			4byte integer	depth[1]
//			4byte integer	depth[2]
//				...
//			4byte integer	depth[beta]
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
//			8byte double	W[1]
//				...
//			8byte double	W[sizeW[alpha]-1]
//			8byte double	b[0]
//			8byte double	b[1]
//				...
//			8byte double	b[sizeb[alpha]-1]
//			8byte double	ConvW[0]
//			8byte double	ConvW[1]
//				...
//			8byte double	ConvW[sizeConvW[beta]-1]
//			8byte double	Convb[0]
//			8byte double	Convb[1]
//				...
//			8byte double	Convb[sizeConvb[beta]-1]
//			8byte double	dLdW[0]
//				...
//			8byte double	dLdb[0]
//				...
//			8byte double	vecdW[0]
//				...
//			8byte double	vecdb[0]
//				...
//			8byte double	ConvdLdW[0]
//				...
//			8byte double	ConvdLdb[0]
//				...
//			8byte double	vecConvdW[0]
//				...
//			8byte double	vecConvdb[0]
//				...
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
	ver[1] = 2;
	fwrite(magic, sizeof(char), 2, fpResult);
	fwrite(ver, sizeof(int), 2, fpResult);
	fwrite(&A, sizeof(int), 1, fpResult);
	fwrite(&B, sizeof(int), 1, fpResult);
	fwrite(&C, sizeof(int), 1, fpResult);
	fwrite(D, sizeof(int), alpha+1, fpResult);
	fwrite(F, sizeof(int), beta, fpResult);
	fwrite(S, sizeof(int), beta, fpResult);
	fwrite(P, sizeof(int), beta, fpResult);
	fwrite(depth+1, sizeof(int), beta, fpResult);
	fwrite(&H, sizeof(double), 1, fpResult);
	fwrite(&DELTA, sizeof(double), 1, fpResult);
	fwrite(&LAMBDA, sizeof(double), 1, fpResult);
	fwrite(&MOMENTUMUPDATE, sizeof(double), 1, fpResult);
	fwrite(&count, sizeof(int), 1, fpResult);
	fwrite(&learningSize, sizeof(int), 1, fpResult);
	fwrite(&l, sizeof(int), 1, fpResult);
	fwrite(&L, sizeof(double), 1, fpResult);
	fwrite(&Lold, sizeof(double), 1, fpResult);
	fwrite(W, sizeof(double), sizeW[alpha], fpResult);
	fwrite(b, sizeof(double), sizeb[alpha], fpResult);
	fwrite(ConvW, sizeof(double), sizeConvW[beta], fpResult);
	fwrite(Convb, sizeof(double), sizeConvb[beta], fpResult);
	fwrite(dLdW, sizeof(double), sizeW[alpha], fpResult);
	fwrite(dLdb, sizeof(double), sizeb[alpha], fpResult);
	fwrite(vecdW, sizeof(double), sizeW[alpha], fpResult);
	fwrite(vecdb, sizeof(double), sizeb[alpha], fpResult);
	fwrite(ConvdLdW, sizeof(double), sizeW[beta], fpResult);
	fwrite(ConvdLdb, sizeof(double), sizeb[beta], fpResult);
	fwrite(vecConvdW, sizeof(double), sizeW[beta], fpResult);
	fwrite(vecConvdb, sizeof(double), sizeb[beta], fpResult);
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
	for(i=0; i<sizeW[alpha]; i++) analytical += dLdW[i];
	for(i=0; i<sizeb[alpha]; i++) analytical += dLdb[i];
	
	numerical = 0;
	for(i=0; i<sizeW[alpha]; i++) numerical += 1/vecdW[i];
	for(i=0; i<sizeb[alpha]; i++) numerical += 1/vecdb[i];
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

double CTraining::CheckAccuracy(int threads)
{
	int t,i,tmp;
	
	t = 0;	// t is an index for each picture
	loaded = 0;	// number of how many images were correct
	double proc;
	// computing score function for each test images
	while(t<Nt)
	{
		std::thread* hThr = new std::thread[threads];
		
		for(i=0; i<threads && t+i<Nt; i++)
			hThr[i] = std::thread(&CTraining::CheckThreadFunc, this, t+i);
		for(i=0; i<threads && t+i<Nt; i++) hThr[i].join();
		t += threads;
		delete [] hThr;
		
		proc = (double) 100 * t/Nt;
		printf("%2.2lf%%\b\b\b\b\b",proc);
		if(proc>9.995) printf("\b");
		if(proc>=99.995) printf("\b");
	}
	printf("done...");
	
	proc = (double) (100*loaded)/Nt;
	loaded = 0;
	return proc;
}

int CTraining::CheckThreadFunc(int index)
{
	int m,i,j,k,i2,j2,k2,I,J,ans;
	double highest;		// temporary score used for seeking highest score
	double* X = (double*) malloc(sizeof(double) * sizes[alpha+1]);
	
	// For CNN case
	if(A)
	{
		double *ConvX = (double*)malloc(sizeof(double) * sizeConvX[beta+1]);
		// initialize
		for(i=0; i<pData->D0; i++)
			ConvX[i] = pData->xt[index][i];
		
		// Conv and Pooling layer procedure
		for(m=0; m<beta; m++)
		{
			// Pooling layer
			if((B>0) && !((m+1)%(A+1)))
			for(i2=0; i2<width[m+1]; i2++){
			for(j2=0; j2<height[m+1]; j2++){
			for(k2=0; k2<depth[m+1]; k2++)
			{
				ConvX[indexOfConvX(m+1,i2,j2,k2)] = 0;
				for(I=i2*S[m]; I<i2*S[m]+F[m]; I++){
				for(J=j2*S[m]; J<j2*S[m]+F[m]; J++){
				if(ConvX[indexOfConvX(m+1,i2,j2,k2)] < ConvX[indexOfConvX(m,I,J,k2)])
				{
					ConvX[indexOfConvX(m+1,i2,j2,k2)] = ConvX[indexOfConvX(m,I,J,k2)];
				}}}
			}}}
			
			// Conv layer
			else
			for(i2=0; i2<width[m+1]; i2++){
			for(j2=0; j2<height[m+1]; j2++){
			for(k2=0; k2<depth[m+1]; k2++)
			{
				ConvX[indexOfConvX(m+1,i2,j2,k2)] = Convb[indexOfConvb(m,k2)];
				for(i=0; i<F[m]; i++){
				for(j=0; j<F[m]; j++){
				for(k=0; k<depth[m]; k++)
				{
					I = i+i2*S[m]-P[m];
					J = j+j2*S[m]-P[m];
					if(I<0 || I>=width[m] || J<0 || J>=height[m]) continue;
						ConvX[indexOfConvX(m+1,i2,j2,k2)] += ConvX[indexOfConvX(m,I,J,k)] * ConvW[indexOfConvW(m,k2,i,j,k)];
				}}}
				if(ConvX[indexOfConvX(m+1,i2,j2,k2)] < 0) ConvX[indexOfConvX(m+1,i2,j2,k2)] = 0;
			}}}
		}
		// Conv & Pool layer ended
		
		i = indexOfConvW(beta,0,0,0,0);
		for(j=0; j<D[0]; j++)
			X[j] = ConvX[j+i];
		free(ConvX);
	}
	// RNN init
	else
	{
		for(i=0; i<pData->D0; i++)
			X[i] = pData->xt[index][i];
	}
	
	// FC layers
	for(i=0; i<C; i++)
	{
		for(j=0; j<D[i+1]; j++)
		{
			X[indexOfs(i+1,j)] = b[indexOfb(i,j)];
			// X^(i+1) = W^i * ReLU^i * X^i + b^i
			for(k=0; k<D[i]; k++)
				X[indexOfs(i+1,j)] += W[indexOfW(i,j,k)] * X[indexOfs(i,k)];
			//ReLU^i_j = 1 if X^i_j>0, 0 otherwise
			if(X[indexOfs(i+1,j)] < 0) X[indexOfs(i+1,j)] = 0;
		}
	}
	
	for(j=0; j<D[alpha]; j++)
	{
		X[indexOfs(alpha,j)] = b[indexOfb(C,j)];
		for(k=0; k<D[C]; k++)
			X[indexOfs(alpha,j)] += W[indexOfW(C,j,k)] * X[indexOfs(C,k)];
	}
	
	// compare with answer and calculate the accuracy
	// firstly find index'th image's highest score and its label
	ans=0;
	highest=X[indexOfs(alpha,0)];
	for(j=1; j<D[alpha]; j++)
	{
		if(X[indexOfs(alpha,j)] > highest)
		{
			ans = j;
			highest = X[indexOfs(alpha,j)];
		}
	}
	free(X);
	
	if(ans == pData->yt[index]) loaded++;
	return 0;
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

void CTraining::CNNThreadFunc(int index)
{
	int m,i,j,k,i2,j2,k2,I,J,u,tmp;
	double *ConvX = (double*)malloc(sizeof(double) * sizeConvX[beta+1]);
	bool *ConvReLU = (bool*)malloc(sizeof(bool) * sizeConvX[beta+1]);
	int *Pooledi = (int*) malloc(sizeof(int) * sizePool[B]);
	int *Pooledj = (int*) malloc(sizeof(int) * sizePool[B]);
	double *ConvdX = (double*)malloc(sizeof(double) * D[alpha] * sizeConvX[beta+1]);
	double *ConvdW = (double*)malloc(sizeof(double) * D[alpha] * sizeConvW[beta]);
	double *Convdb = (double*)malloc(sizeof(double) * D[alpha] * sizeConvb[beta]);
	double* X = (double*) malloc(sizeof(double) * sizes[alpha+1]);
	double* dW = (double*) malloc(sizeof(double) * D[alpha] * sizeW[alpha]);
	double* db = (double*) malloc(sizeof(double) * D[alpha] * sizeb[alpha]);
	bool* ReLU = (bool*) malloc(sizeof(bool) * sizes[alpha+1]);
	
	if(!A && !B)
	{
		RNNThreadFunc(index);
		return;
	}
	
	// initialize
	for(i=0; i<pData->D0; i++)
	{
		ConvX[i] = pData->x[index][i];
		ConvReLU[i] = 1;
	}
	
	// Conv and Pooling layer procedure
	for(m=0; m<beta; m++)
	{
		// Pooling layer
		if((B>0) && !((m+1)%(A+1)))
		for(i2=0; i2<width[m+1]; i2++){
		for(j2=0; j2<height[m+1]; j2++){
		for(k2=0; k2<depth[m+1]; k2++)
		{
			ConvX[indexOfConvX(m+1,i2,j2,k2)] = 0;
			for(I=i2*S[m]; I<i2*S[m]+F[m]; I++){
			for(J=j2*S[m]; J<j2*S[m]+F[m]; J++){
			if(ConvX[indexOfConvX(m+1,i2,j2,k2)] < ConvX[indexOfConvX(m,I,J,k2)])
			{
				ConvX[indexOfConvX(m+1,i2,j2,k2)] = ConvX[indexOfConvX(m,I,J,k2)];
				Pooledi[indexOfPool(m+1,i2,j2,k2)] = I;
				Pooledj[indexOfPool(m+1,i2,j2,k2)] = J;
			}}}
		}}}
		
		// Conv layer
		else
		for(i2=0; i2<width[m+1]; i2++){
		for(j2=0; j2<height[m+1]; j2++){
		for(k2=0; k2<depth[m+1]; k2++)
		{
			ConvX[indexOfConvX(m+1,i2,j2,k2)] = Convb[indexOfConvb(m,k2)];
			for(i=0; i<F[m]; i++){
			for(j=0; j<F[m]; j++){
			for(k=0; k<depth[m]; k++)
			{
				I = i+i2*S[m]-P[m];
				J = j+j2*S[m]-P[m];
				if(I<0 || I>=width[m] || J<0 || J>=height[m]) continue;
				if(ConvReLU[indexOfConvX(m,I,J,k)])
					ConvX[indexOfConvX(m+1,i2,j2,k2)] += ConvX[indexOfConvX(m,I,J,k)] * ConvW[indexOfConvW(m,k2,i,j,k)];
			}}}
			if(ConvX[indexOfConvX(m+1,i2,j2,k2)] < 0) ConvReLU[indexOfConvX(m+1,i2,j2,k2)] = 0;
			else ConvReLU[indexOfConvX(m+1,i2,j2,k2)] = 1;
		}}}
	}
	// Conv & Pool layer ended
	
	i = indexOfConvW(beta,0,0,0,0);
	for(j=0; j<D[0]; j++)
	{
		X[j] = ConvX[j+i];
		ReLU[j] = ConvReLU[j+i];
	}
	
	// this loop is a procedure of score function
	for(i=0; i<C; i++)
	{
		for(j=0; j<D[i+1]; j++)
		{
			X[indexOfs(i+1,j)] = b[indexOfb(i,j)];
			// X^(i+1) = W^i * ReLU^i * X^i + b^i
			for(k=0; k<D[i]; k++)
				if(ReLU[indexOfs(i,k)])
					X[indexOfs(i+1,j)] += W[indexOfW(i,j,k)] * X[indexOfs(i,k)];
			//ReLU^i_j = 1 if X^i_j>0, 0 otherwise
			if(X[indexOfs(i+1,j)] > 0) ReLU[indexOfs(i+1,j)] = 1;
			else ReLU[indexOfs(i+1,j)] = 0;
		}
	}
	
	for(j=0; j<D[alpha]; j++)
	{
		X[indexOfs(alpha,j)] = b[indexOfb(C,j)];
		// X^(i+1) = W^i * ReLU^i * X^i + b^i
		for(k=0; k<D[C]; k++)
			if(ReLU[indexOfs(C,k)])
				X[indexOfs(alpha,j)] += W[indexOfW(C,j,k)] * X[indexOfs(C,k)];
		//ReLU^i_j = 1 if X^i_j>0, 0 otherwise
		ReLU[indexOfs(alpha,j)] = 1;
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
				dW[indexOfdW(alpha-1,i,j,k)] = db[indexOfdb(alpha-1,i,j)] * X[indexOfs(alpha-1,k)];
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
				if(ReLU[indexOfs(m+1,j)])
					for(k=0; k<D[m+2]; k++)
						db[indexOfdb(m,i,j)] += db[indexOfdb(m+1,i,k)] * W[indexOfW(m+1,k,j)];
						
				// compute ds^alpha_i / dW^m_j,k
				for(k=0; k<D[m]; k++)
				{
					if(ReLU[indexOfs(m,k)]) dW[indexOfdW(m,i,j,k)] = db[indexOfdb(m,i,j)] * X[indexOfs(m,k)];
					else dW[indexOfdW(m,i,j,k)] = 0;
				}
			}
		}
	}
	// computing gradients of FC Weights is done
	
	for(i=0; i<D[alpha] * sizeConvX[beta+1]; i++) ConvdX[i] = 0;
	for(i=0; i<D[alpha] * sizeConvW[beta]; i++) ConvdW[i] = 0;
	for(i=0; i<D[alpha] * sizeConvb[beta]; i++) Convdb[i] = 0;
	// calculate initial gradient of convdX
	for(u=0; u<D[alpha]; u++)
	{
		tmp = indexOfConvdX(u,beta,0,0,0);
		for(int v=0; v<D[0]; v++)
		{
			if(ReLU[indexOfs(0,v)])
			for(i=0; i<D[1]; i++)
				ConvdX[tmp+v] += db[indexOfdb(0,u,i)] * W[indexOfW(0,i,v)];
		}
	}
	
	// calculate gradient of CNN for general cases
	for(m=beta-1; m>=0; m--)
	{
		// Pooling layer
		if(B && !((m+1)%(A+1)))
		{
			for(I=0; I<width[m]; I++){
			for(J=0; J<height[m]; j++){
			for(k=0; k<depth[m]; k++)
			{
				for(i2=(int)(I-F[m])/S[m] + 1; i2<(int)I/S[m]; i2++){
				for(j2=(int)(J-F[m])/S[m] + 1; j2<(int)J/S[m]; j2++){
				if(I==Pooledi[indexOfPool(m+1,i2,j2,k)] && J==Pooledj[indexOfPool(m+1,i2,j2,k)])
				for(u=0; u<D[alpha]; u++)
					ConvdX[indexOfConvdX(u,m,I,J,k)] += ConvdX[indexOfConvdX(u,m+1,i2,j2,k)];
			}}}}}
		}
	
		// Conv layer
		else
		{
			for(i2=0; i2<width[m+1]; i2++){
			for(j2=0; j2<height[m+1]; j2++){
			for(k2=0; k2<depth[m+1]; k2++)
			{
				for(i=0; i<F[m]; i++){
				for(j=0; j<F[m]; j++){
				for(k=0; k<depth[m]; k++)
				{
					I = i+i2*S[m]-P[m];
					J = j+j2*S[m]-P[m];
					if(I<0 || I>=width[m] || J<0 || J>=height[m]) continue;
					if(ConvReLU[indexOfConvX(m,I,J,k)])
					for(u=0; u<D[alpha]; u++)
					{
						// ConvdX
						ConvdX[indexOfConvdX(u,m,I,J,k)] += ConvdX[indexOfConvdX(u,m+1,i2,j2,k2)] * ConvW[indexOfConvW(m,k2,i,j,k)];
						// ConvdW
						ConvdW[indexOfConvdW(u,m,k2,i,j,k)] += ConvdX[indexOfConvdX(u,m+1,i2,j2,k2)] * ConvX[indexOfConvX(m,I,J,k)];
					}
				}}}
				for(u=0; u<D[alpha]; u++)
					Convdb[indexOfConvdb(u,m,k2)] += ConvdX[indexOfConvdX(u,m+1,i2,j2,k2)];
			}}}
		}
	}
	
	// this is a procedure of calculating loss function
	// according to SVM and its gradient about W,b
	// L_l = sig_i
	for(u=0; u<D[alpha]; u++)
	{
		if(u == pData->y[index]) continue;
		if((tmp = X[indexOfs(alpha,u)] - X[indexOfs(alpha,pData->y[index])] + DELTA) > 0)
		{
			L += tmp;
			for(m=0; m<alpha; m++)
			{
				for(j=0; j<D[m+1]; j++)
				{
					for(k=0; k<D[m]; k++)
						dLdW[indexOfW(m,j,k)] += dW[indexOfdW(m,u,j,k)] - dW[indexOfdW(m,pData->y[index],j,k)];
					dLdb[indexOfb(m,j)] += db[indexOfdb(m,u,j)] - db[indexOfdb(m,pData->y[index],j)];
				}
			}
			for(m=0; m<beta; m++)
			if(!B || ((m+1)%(A+1)))
			for(k2=0; k2<depth[m+1]; k2++)
			{
				for(i=0; i<F[m]; i++)
				for(j=0; j<F[m]; j++)
				for(k=0; k<depth[m]; k++)
					ConvdLdW[indexOfConvW(m,k2,i,j,k)] += ConvdW[indexOfConvdW(u,m,k2,i,j,k)] - ConvdW[indexOfConvdW(pData->y[index],m,k2,i,j,k)];
				ConvdLdb[indexOfConvb(m,k2)] += Convdb[indexOfConvdb(u,m,k2)] - Convdb[indexOfConvdb(pData->y[index],m,k2)];
			}
		}
	}
	
	// free memories used at this thread
	free(dW);
	free(db);
	free(X);
	free(ReLU);
	free(ConvX);
	free(ConvReLU);
	free(Pooledi);
	free(Pooledj);
	free(ConvdX);
	free(ConvdW);
	free(Convdb);
}

int CTraining::RNNThreadFunc(int index)
{
	int i,j,k,m,tmp;
	// X^(i+1) = W^i * delta^i * X^i + b^i
	double* X = (double*) malloc(sizeof(double) * sizes[alpha+1]);
	// dW[indexOfW(m,i,j,k)] : ds^alpha_i / dW^m_j,k
	double* dW = (double*) malloc(sizeof(double) * D[alpha] * sizeW[alpha]);
	// db[sizeOfb(m,i,j)] : ds^alpha_i / db^m_j
	// and this is exactly same as ds^alpha_i / ds^(m+1)_j mathematically
	double* db = (double*) malloc(sizeof(double) * D[alpha] * sizeb[alpha]);
	bool* ReLU = (bool*) malloc(sizeof(bool) * sizes[alpha+1]);
	
	// initialize score function and ReLU chronicle
	for(j=0; j<D[0]; j++)
	{
		X[indexOfs(0,j)] = pData->x[index][j];	// initializing sBef = x_l
		ReLU[indexOfs(0,j)] = 1;	// initializing first deltaBef
	}
	
	// this loop is a procedure of score function
	for(i=0; i<C; i++)
	{
		for(j=0; j<D[i+1]; j++)
		{
			X[indexOfs(i+1,j)] = b[indexOfb(i,j)];
			// X^(i+1) = W^i * ReLU^i * X^i + b^i
			for(k=0; k<D[i]; k++)
				if(ReLU[indexOfs(i,k)])
					X[indexOfs(i+1,j)] += W[indexOfW(i,j,k)] * X[indexOfs(i,k)];
			//ReLU^i_j = 1 if X^i_j>0, 0 otherwise
			if(X[indexOfs(i+1,j)] > 0) ReLU[indexOfs(i+1,j)] = 1;
			else ReLU[indexOfs(i+1,j)] = 0;
		}
	}
	
	for(j=0; j<D[alpha]; j++)
	{
		X[indexOfs(alpha,j)] = b[indexOfb(C,j)];
		// X^(i+1) = W^i * ReLU^i * X^i + b^i
		for(k=0; k<D[C]; k++)
			if(ReLU[indexOfs(C,k)])
				X[indexOfs(alpha,j)] += W[indexOfW(C,j,k)] * X[indexOfs(C,k)];
		//ReLU^i_j = 1 if X^i_j>0, 0 otherwise
		ReLU[indexOfs(alpha,j)] = 1;
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
				dW[indexOfdW(alpha-1,i,j,k)] = db[indexOfdb(alpha-1,i,j)] * X[indexOfs(alpha-1,k)];
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
				if(ReLU[indexOfs(m+1,j)])
					for(k=0; k<D[m+2]; k++)
						db[indexOfdb(m,i,j)] += db[indexOfdb(m+1,i,k)] * W[indexOfW(m+1,k,j)];
						
				// compute ds^alpha_i / dW^m_j,k
				for(k=0; k<D[m]; k++)
				{
					if(ReLU[indexOfs(m,k)]) dW[indexOfdW(m,i,j,k)] = db[indexOfdb(m,i,j)] * X[indexOfs(m,k)];
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
		if((tmp = X[indexOfs(alpha,i)] - X[indexOfs(alpha,pData->y[index])] + DELTA) > 0)
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
	free(X);
	free(ReLU);
	
	return 0;
}
