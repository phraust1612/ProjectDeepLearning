#include "training.h"

CTraining::CTraining(CDataread* pD)
{
	pData = pD;
	N = pData->N;
	Nt = pData->Nt;
	loaded=0;
	
	alpha = 1;
	learningSize = 0;
	count = 0;
	DELTA = 0;
	LAMBDA = 0;
	H = 0;
	Lold = 0;
	l = 0;
	L = 0;
}

CTraining::~CTraining()
{
	int i,j,k,u,v;
	for(i=0; i<alpha; i++)
	{
		for(j=0; j<D[i+1]; j++)
		{
			free(W[i][j]);
			free(dLdW[i][j]);
		}
		free(W[i]);
		free(dLdW[i]);
		free(b[i]);
		free(dLdb[i]);
	}
	free(W);
	free(dLdW);
	free(b);
	free(dLdb);
	free(D);
}

int CTraining::WeightInit(int size)
{
	int i,j,k;
	count = 0;
	learningSize=size;
	DELTA = DELTADEFAULT;
	LAMBDA = LAMBDADEFAULT;
	H = HDEFAULT;
	printf("input alpha\n>> ");
	scanf("%d", &alpha);	// number of W's
	getchar();
	
	D = (int*) malloc(sizeof(int) * (alpha+1));
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
	int numW=0;
	for(i=0; i<alpha; i++)
	{
		for(j=0; j<D[i+1]; j++)
		{
			b[i][j] = 0.01;
			for(k=0; k<D[i]; k++)
			{
				W[i][j][k] = (double)rand();
				sumW += W[i][j][k];
			}
		}
		numW += D[i+1] * D[i];
	}
	// choose random number according to normal distribution
	// and times sqrt(2/N)
	for(i=0; i<alpha; i++)
		for(j=0; j<D[i+1]; j++)
			for(k=0; k<D[i]; k++)
				W[i][j][k] = (sqrt(2) * (W[i][j][k] - U) * (sumW - (double) numW * U)) / (V * sqrt(N));
				
	return 0;
}

int CTraining::WeightLoad()
{
	int i,j;
	FILE* fpWeight = fopen("TrainedParam", "rb");
	fread(&alpha, sizeof(int), 1, fpWeight);	// scan alpha
	// allocate D and scan them
	D = (int*) malloc(sizeof(int) * (alpha+1));
	fread(D, sizeof(int), alpha+1, fpWeight);
	
	// allocate memories
	ParamAllocate();
	
	// scan W
	for(i=0; i<alpha; i++)
		for(j=0; j<D[i+1]; j++)
			fread(W[i][j], sizeof(double), D[i], fpWeight);
	// scan b
	for(i=0; i<alpha; i++)
		fread(b[i], sizeof(double), D[i+1], fpWeight);
	
	fread(&H, sizeof(double), 1, fpWeight);
	if(!H) return ERR_CRACKED_FILE;
	fread(&DELTA, sizeof(double), 1, fpWeight);
	fread(&LAMBDA, sizeof(double), 1, fpWeight);
	fread(&count, sizeof(int), 1, fpWeight);
	fread(&learningSize, sizeof(int), 1, fpWeight);
	fread(&l, sizeof(int), 1, fpWeight);
	fread(&L, sizeof(double), 1, fpWeight);
	fread(&Lold, sizeof(double), 1, fpWeight);
	// scan W
	for(i=0; i<alpha; i++)
		for(j=0; j<D[i+1]; j++)
			fread(dLdW[i][j], sizeof(double), D[i], fpWeight);
	// scan b
	for(i=0; i<alpha; i++)
		fread(dLdb[i], sizeof(double), D[i+1], fpWeight);
	fclose(fpWeight);
	loaded=1;
	
	return 0;
}

void CTraining::ParamAllocate()
{
	int i,j,k,u;
	
	// dLdW[i][j][k] : dL / dW^i_j,k
	// dLdb[i][j] : dL / db^i_j
	W = (double***) malloc(sizeof(double**) * alpha);
	b = (double**) malloc(sizeof(double*) * alpha);
	dLdW = (double***) malloc(sizeof(double**) * alpha);
	dLdb = (double**) malloc(sizeof(double*) * alpha);
	for(i=0; i<alpha; i++)
	{
		W[i] = (double**) malloc(sizeof(double*) * D[i+1]);
		b[i] = (double*) malloc(sizeof(double) * D[i+1]);
		dLdW[i] = (double**) malloc(sizeof(double*) * D[i+1]);
		dLdb[i] = (double*) malloc(sizeof(double) * D[i+1]);
		for(j=0; j<D[i+1]; j++)
		{
			W[i][j] = (double*) malloc(sizeof(double) * D[i]);
			dLdb[i][j] = 0;
			dLdW[i][j] = (double*) malloc(sizeof(double) * D[i]);
		}
	}
}

void CTraining::Training(int threads)
{
	int i,j,k,u,v,m,tmp,hr,min,sec,startindexl,startindexcount;
	time_t starttime, endtime;
	double gap, Try, acc;
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
	
	for(;count<learningSize && cont; count++) 
	{
		// initializing dL/dW^i_j,k and dL/db^i_j
		// I'm wondering if I have to memorize whole L_l
		// so I'll just accumulate all gradient
		if(!loaded)
		{
			L = 0;
			l=0;
			for(i=0; i<alpha && cont; i++)
			{
				for(j=0; j<D[i+1]; j++)
				{
					dLdb[i][j] = 0;
					for(k=0; k<D[i]; k++) dLdW[i][j][k] = 0;
				}
			}
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
					printf("\n");
					cont = false;
					break;
				default:
					Key.keysave = 'n';
					break;
			}
			
			// run multi-thread to calculate gradients
			for(i=0; i<threads; i++) hThread[i] = std::thread(&CTraining::TrainingThreadFunc, this, l+i);
			for(i=0; i<threads; i++) hThread[i].join();
			l += threads;
		}
		
		// L2 regularization
		for(i=0; i<alpha; i++)
		{
			for(j=0; j<D[i+1]; j++)
			{
				for(k=0; k<D[i]; k++)
				{
					dLdW[i][j][k] /= (double) N;
					dLdW[i][j][k] += LAMBDA * W[i][j][k];
				}
				dLdb[i][j] /= (double) N;
			}
		}
		// calculate gradient of L is done...
		
		// optimizing next W, b according to SGD
		if(cont)
		{
			for(i=0; i<alpha; i++)
			{
				for(j=0; j<D[i+1]; j++)
				{
					for(k=0; k<D[i]; k++) W[i][j][k] -= (H * dLdW[i][j][k]);
					b[i][j] -= (H * dLdb[i][j]);
				}
			}
			// then retry
			
			// show loss func and its difference
			printf("\nL = %lf\n", L/N);
			printf("increment of L = %lf...", (L-Lold)/N);
			Lold = L;
		}
	}
	Key.Stop();
}

// save alpha, D, W, b at file
// file format : 
// offset	type			description
// 0x0000	4byte integer	alpha
// 0x0004	4byte integer	D[0]
// 0x0008	4byte integer	D[1]
// 				...
// 			4byte integer	D[alpha]
//			8byte double	W[0][0][0]
//				...
//			8byte double	W[alpha-1][D[alpha]][D[alpha-1]]
//			8byte double	b[0][0]
//				...
//			8byte double	H
//			8byte double	DELTA
//			8byte double	LAMBDA
//			4byte integer	count
//			4byte integer	learningsize
//			4byte int		l (do not write behind if training ended)
//			8byte double	L
//			8byte double	Lold
//			8byte double	dLdW[0][0][0]
//				...
//			8byte double	dLdW[alpha-1][D[alpha]][D[alpha-1]]
//			8byte double	dLdb[0][0]
//				...
void CTraining::FileSave()
{
	int i, j;
	FILE* fpResult = fopen("TrainedParam", "wb");
	fwrite(&alpha, sizeof(int), 1, fpResult);
	fwrite(D, sizeof(int), alpha+1, fpResult);
	for(i=0; i<alpha; i++)
		for(j=0; j<D[i+1]; j++)
			fwrite(W[i][j], sizeof(double), D[i], fpResult);
	for(i=0; i<alpha; i++) fwrite(b[i], sizeof(double), D[i+1], fpResult);
	fwrite(&H, sizeof(double), 1, fpResult);
	fwrite(&DELTA, sizeof(double), 1, fpResult);
	fwrite(&LAMBDA, sizeof(double), 1, fpResult);
	fwrite(&count, sizeof(int), 1, fpResult);
	fwrite(&learningSize, sizeof(int), 1, fpResult);
	fwrite(&l, sizeof(int), 1, fpResult);
	fwrite(&L, sizeof(double), 1, fpResult);
	fwrite(&Lold, sizeof(double), 1, fpResult);
	for(i=0; i<alpha; i++)
		for(j=0; j<D[i+1]; j++)
			fwrite(dLdW[i][j], sizeof(double), D[i], fpResult);
	for(i=0; i<alpha; i++) fwrite(dLdb[i], sizeof(double), D[i+1], fpResult);
	fclose(fpResult);
}

double CTraining::CheckAccuracy()
{
	int i, j, k, l;
	// memory allocation for s and delta
	double** st = (double**) malloc(sizeof(double*) * (alpha+1));
	for(i=0; i<=alpha; i++) st[i] = (double*) malloc(sizeof(double) * D[i]);
	bool** deltat = (bool**) malloc(sizeof(bool*) * alpha);
	for(i=0; i<alpha; i++) deltat[i] = (bool*) malloc(sizeof(bool) * D[i]);
	
	int count = 0;	// number of how many images were correct
	int ans;			// temporary answer
	double highest;		// temporary score used for seeking highest score
	// computing score function for each test images
	for(l=0; l<Nt; l++)	// l is an index for each picture
	{
		j=0;
		// initializaing...
		for(i=0; i<D[0]; i++)
		{
			st[0][i] = pData->xt[l][i];	// initializing s_0 = x_l
			deltat[0][i] = 1;	// initializing deltat^0_i,i = 1,
		}						// actually deltat is DxD matrix but I'll save memory
		
		// this loop is a procedure of score function
		for(i=1; i<=alpha; i++)
		{
			for(j=0; j<D[i]; j++)
			{
				st[i][j] = 0;
				// st^i = W^(i-1) * deltat^(i-1) * st^(i-1) + b^(i-1)
				for(k=0; k<D[i-1]; k++)
					if(deltat[i-1][k])
						st[i][j] += W[i-1][j][k] * st[i-1][k];
				st[i][j] += b[i-1][j];
				//deltat^i_j = 1 if st^i_j>0, 0 otherwise
				if(i==alpha) continue;	// because there is no deltat[alpha] memory allocated
				if(st[i][j] > 0) deltat[i][j] = 1;
				else deltat[i][j] = 0;
			}
		}
		
		// compare with answer and calculate the accuracy
		// firstly find l'th image's highest score and its label
		ans=0;
		highest=st[alpha][0];
		for(j=1; j<D[alpha]; j++)
		{
			if(st[alpha][j] > highest)
			{
				ans = j;
				highest = st[alpha][j];
			}
		}
		// accumulate count if ans is correct
		if(ans == pData->yt[l]) count++;
	}
	
	for(i=0; i<=alpha; i++) free(st[i]);
	for(i=0; i<alpha; i++) free(deltat[i]);
	free(st);
	free(deltat);
	highest = (double) (100*count)/Nt;
	return highest;
}

int CTraining::SetHyperparam(ValidationParam validateMode, double hyperparam)
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

void CTraining::TrainingThreadFunc(int l)
{
	int i,j,k,u,v,m,tmp;
	// memory allocation of s, delta, etc
	// s^(i+1) = W^i * delta^i * s^i + b^i
	double** s = (double**) malloc(sizeof(double*) * (alpha+1));
	// dW[m][i][j][k] : ds^alpha_i / dW^m_j,k
	double**** dW = (double****) malloc(sizeof(double***) * alpha);
	// db[m][i][j] : ds^alpha_i / db^m_j
	// and this is exactly same as ds^alpha_i / ds^(m+1)_j mathematically
	double*** db = (double***) malloc(sizeof(double**) * alpha);
	// this delta is a deltachronical matrix, used at ReLU
	bool** delta = (bool**) malloc(sizeof(bool*) * (alpha+1));
	for(i=0; i<=alpha; i++)
	{
		s[i] = (double*) malloc(sizeof(double) * D[i]);
		delta[i] = (bool*) malloc(sizeof(bool) * D[i]);
	}
	
	for(m=0; m<alpha; m++)
	{
		dW[m] = (double***) malloc(sizeof(double**) * D[alpha]);
		db[m] = (double**) malloc(sizeof(double*) * D[alpha]);
		for(i=0; i<D[alpha]; i++)
		{
			dW[m][i] = (double**) malloc(sizeof(double*) * D[m+1]);
			db[m][i] = (double*) malloc(sizeof(double) * D[m+1]);
			for(j=0; j<D[m+1]; j++) dW[m][i][j] = (double*) malloc(sizeof(double) * D[m]);
		}
	}
	
	// initialize score function and delta chronicle
	for(j=0; j<D[0]; j++)
	{
		s[0][j] = pData->x[l][j];	// initializing sBef = x_l
		delta[0][j] = 1;	// initializing first deltaBef
	}
	
	// this loop is a procedure of score function
	for(i=0; i<alpha; i++)
	{
		for(j=0; j<D[i+1]; j++)
		{
			s[i+1][j] = 0;
			// s^(i+1) = W^i * delta^i * s^i + b^i
			for(k=0; k<D[i]; k++)
				if(delta[i][k])
					s[i+1][j] += W[i][j][k] * s[i][k];
			s[i+1][j] += b[i][j];
			//delta^i_j = 1 if s^i_j>0, 0 otherwise
//			if(i>=alpha-1) continue;	// because there is no delta[alpha] memory allocated
			if(s[i+1][j] > 0) delta[i+1][j] = 1;
			else delta[i+1][j] = 0;
		}
	}
	
	// initialize db and dW (for the case m=alpha-1)
	for(i=0; i<D[alpha]; i++)
	{
		for(j=0; j<D[alpha]; j++)
		{
			// ds^alpha_i / db^(alpha-1)_j = 1 if and only if i=j
			// otherwise it becomes 0
			if(i==j) db[alpha-1][i][j] = 1;
			else db[alpha-1][i][j] = 0;
			
			for(k=0; k<D[alpha-1]; k++)
				if(delta[alpha-1][k])
					dW[alpha-1][i][j][k] = db[alpha-1][i][j] * s[alpha-1][k];
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
				db[m][i][j] = 0;
				if(delta[m+1][j])
					for(k=0; k<D[m+2]; k++)
						db[m][i][j] += db[m+1][i][k] * W[m+1][k][j];
						
				// compute ds^alpha_i / dW^m_j,k
				for(k=0; k<D[m]; k++)
				{
					if(delta[m][k]) dW[m][i][j][k] = db[m][i][j] * s[m][k];
					else dW[m][i][j][k] = 0;
				}
			}
		}
	}
	
	// this is a procedure of calculating loss function
	// according to SVM and its gradient about W,b
	// L_l = sig_i
	for(i=0; i<D[alpha]; i++)
	{
		if(i == pData->y[l]) continue;
		if((tmp = s[alpha][i] - s[alpha][pData->y[l]] + DELTA) > 0)
		{
			L += tmp;
			for(m=0; m<alpha; m++)
			{
				for(j=0; j<D[m+1]; j++)
				{
					for(k=0; k<D[m]; k++)
						dLdW[m][j][k] += dW[m][i][j][k] - dW[m][pData->y[l]][j][k];
					dLdb[m][j] += db[m][i][j] - db[m][pData->y[l]][j];
				}
			}
		}
	}
	
	// free memories used at this thread
	for(m=0; m<alpha; m++)
	{
		for(i=0; i<D[alpha]; i++)
		{
			for(j=0; j<D[m+1]; j++) free(dW[m][i][j]);
			free(dW[m][i]);
			free(db[m][i]);
		}
		free(dW[m]);
		free(db[m]);
	}
	free(dW);
	free(db);
	for(i=0; i<=alpha; i++)
	{
		free(s[i]);
		free(delta[i]);
	}
	free(s);
	free(delta);
}
