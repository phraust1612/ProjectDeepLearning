#include "training.h"
#include <math.h>

CKeyinter Key;
int cont, loaded;
void KeyIntercept();

CTraining::CTraining(CDataread* pD)
{
	pData = pD;
	N = pData->N;
	Nt = pData->Nt;
	cont = 1;
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
	for(i=0; i<=alpha; i++)
	{
		free(s[i]);
		free(delta[i]);
	}
	free(s);
	free(delta);
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
	for(i=0; i<alpha; i++)
	{
		for(j=0; j<D[i]; j++)
		{
			for(k=0; k<i; k++)
			{
				for(u=0; u<D[k+1]; u++) free(dW[i][j][k][u]);
				free(dW[i][j][k]);
				free(db[i][j][k]);
			}
			free(dW[i][j]);
			free(db[i][j]);
		}
		free(dW[i]);
		free(db[i]);
	}
	free(dW);
	free(db);
	free(D);
}

int CTraining::WeightInit(int size)
{
	int i,j,k;
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
	if(!DELTA) return ERR_CRACKED_FILE;
	fread(&LAMBDA, sizeof(double), 1, fpWeight);
	if(!LAMBDA) return ERR_CRACKED_FILE;
	fread(&learningSize, sizeof(int), 1, fpWeight);
	if(learningSize)
	{
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
	}
	fclose(fpWeight);
	loaded=1;
	
	return 0;
}

void CTraining::ParamAllocate()
{
	int i,j,k,u;
	// memory allocation of s, delta, etc
	// s^(i+1) = W^i * delta^i * s^i + b^i
	s = (double**) malloc(sizeof(double*) * (alpha+1));
	delta = (bool**) malloc(sizeof(bool*) * (alpha+1));
	for(i=0; i<=alpha; i++)
	{
		s[i] = (double*) malloc(sizeof(double) * D[i]);
		delta[i] = (bool*) malloc(sizeof(bool) * D[i]);
	}
	
	// dW[i][j][k][u][v] : ds^(i+1)_j / dW^k_u,v
	// db[i][j][k][u] : ds^(i+1)_j / db^k_u
	dW = (double*****) malloc(sizeof(double****) * alpha);
	db = (double****) malloc(sizeof(double***) * alpha);
	for(i=0; i<alpha; i++)
	{
		dW[i] = (double****) malloc(sizeof(double***) * D[i]);
		db[i] = (double***) malloc(sizeof(double**) * D[i]);
		for(j=0; j<D[i]; j++)
		{
			dW[i][j] = (double***) malloc(sizeof(double**) * (i+1));
			db[i][j] = (double**) malloc(sizeof(double*) * (i+1));
			for(k=0; k<=i; k++)
			{
				dW[i][j][k] = (double**) malloc(sizeof(double*) * D[k+1]);
				db[i][j][k] = (double*) malloc(sizeof(double) * D[k+1]);
				for(u=0; u<D[k+1]; u++) dW[i][j][k][u] = (double*) malloc(sizeof(double) * D[k]);
			}
		}
	}
	
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

void CTraining::Training()
{
	int i,j,k,u,v,m,tmp;
	double acc;
	// Callback function starts to catch key inputs
	Key = CKeyinter();
	Key.Start();
	Key.SetCallbackFunction(KeyIntercept);
	endtime=0;
	double gap, Try;
	int hr, min, sec;
	starttime = clock();
	ValidationParam valid;
	
	for(count=0; count<learningSize && cont; count++) 
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
		for(; l<N && cont; l++)	// l is an index for each picture
		{
			// if callback function catched character 's'
			// save W,b parameters to a file
			if(cont==2)
			{
				FileSave();
				cont=1;
				printf("save suceeded...\n>> ");
			}
			// if callback function catched character 'c'
			// check current accuracy
			if(cont==3)
			{
				printf("start testing procedure...\n");
				acc = CheckAccuracy();
				cont=1;
				printf("accuracy : %2.2lf%%!!!\n>> ", acc);
			}
			if(cont==4)
			{
				ShowHelp();
				cont=1;
			}
			if(cont==6)
			{
				printf("loop %2.2lf%%\n>> ", (double)(100*l)/N);
				cont=1;
			}
			
			// initializaing...
			for(j=0; j<D[0] && cont; j++)
			{
				s[0][j] = pData->x[l][j];	// initializing sBef = x_l
				delta[0][j] = 1;	// initializing first deltaBef
			}
			
			for(i=0; i<alpha && cont; i++)
			{
				// this loop is a procedure of score function
				for(j=0; j<D[i+1] && cont; j++)
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
					
					// loop for calculating gradient
					for(k=0; k<i; k++)
					{
						for(u=0; u<D[k+1]; u++)
						{
							db[i][j][k][u] = 0;
							for(v=0; v<D[i]; v++)
							{
								dW[i][j][k][u][v] = 0;
								for(m=0; m<D[i]; m++)
									if(delta[i][m]) dW[i][j][k][u][v] += W[i][j][m] * dW[i-1][m][k][u][v];
							}
							for(m=0; m<D[i]; m++)
								if(delta[i][m]) db[i][j][k][u] += W[i][j][m] * db[i-1][m][k][u];
						}
					}

					// for k=i-1, ds^(i+1)_j/dW^i_j,v = delta^i_k * s^i_k
					// ds^(i+1)_j/db^i_j,v = 1
					// otherwise 0
					k=i;
					for(u=0; u<D[k+1]; u++)
					{
						for(v=0; v<D[k]; v++)
						{
							if(u==j && delta[i][v]) dW[i][j][k][u][v] = s[i][v];
							else dW[i][j][k][u][v] = 0;
						}
						if(u==j) db[i][j][k][u] = 1;
						else db[i][j][k][u] = 0;
					}
				}
			}
			
			// this is a procedure of calculating loss function
			// according to SVM and its gradient about W,b
			// L_l = sig_i
			for(j=0; j<D[alpha]; j++)
			{
				if(j == pData->y[l]) continue;
				if((tmp = s[alpha][j] - s[alpha][pData->y[l]] + DELTA) > 0)
				{
					L += tmp;
					for(k=0; k<alpha; k++)
					{
						for(u=0; u<D[k+1]; u++)
						{
							for(v=0; v<D[k]; v++)
								// applying L2 Regularization
								dLdW[k][u][v] += dW[alpha-1][j][k][u][v] - dW[alpha-1][pData->y[l]][k][u][v];
							dLdb[k][u] += db[alpha-1][j][k][u] - db[alpha-1][pData->y[l]][k][u];
						}
					}
				}
			}
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
			
			printf("\nlearning procedure : %2.2lf%% done...\n", (double)(100 * (count+1))/(learningSize));
			// show loss func and its difference
			printf("general s = %lf\n", s[alpha][0]);
			printf("L = %lf\n", L/N);
			printf("increment of L = %lf\n", (L-Lold)/N);
			Lold = L;
			
			// show expected time
			endtime = clock();
			gap = (double) (endtime - starttime)/(CLOCKS_PER_SEC);
			sec = (int) gap;
			hr = (int) sec/3600;
			sec -= (int) hr*3600;
			min = (int) sec/60;
			sec -= (int) min*60;
			printf("costed about %d : %d : %d time...\n", hr, min, sec);
			sec = (int) gap*(learningSize - count-1)/(count+1);
				hr = (int) sec/3600;
			sec -= (int) hr*3600;
			min = (int) sec/60;
			sec -= (int) min*60;
			printf("estimated remaining time %d : %d : %d...\n>> ", hr, min, sec);
		}
		if(cont == 5)
		{
			printf("to modify DELTA, input 1\n");
			printf("to modify LAMBDA, input 2\n");
			printf("to modify H, input 3\n>> ");
			scanf("%d", &valid);
			getchar();
			
			printf("input trial hyperparam value\n>> ");
			scanf("%lf", &Try);
			getchar();
			tmp = SetHyperparam(valid, Try);
			if(tmp) printf("wrong query!!!\n");
			cont=1;
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
//			4byte integer	learningSize(remaining)
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
	i = learningSize - count;	// temporary learningSize to save without influencing current process
	if(!cont) i++;
	fwrite(&i, sizeof(int), 1, fpResult);	// learningSize
	if(i)
	{
		fwrite(&l, sizeof(int), 1, fpResult);
		fwrite(&L, sizeof(double), 1, fpResult);
		fwrite(&Lold, sizeof(double), 1, fpResult);
		for(i=0; i<alpha; i++)
			for(j=0; j<D[i+1]; j++)
				fwrite(dLdW[i][j], sizeof(double), D[i], fpResult);
		for(i=0; i<alpha; i++) fwrite(dLdb[i], sizeof(double), D[i+1], fpResult);
	}
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
	printf("enter c whenever you want to check accuracy\n");
	printf("enter s whenever you want to save\n");
	printf("enter q whenever you want to quit the program\n");
	printf("enter m whenever you want to modify hyperparameters\n");
	printf("enter h whenever you want to read this help message again\n>> ");
}

void KeyIntercept()
{
	int temp = Key.keysave;
	switch(temp)
	{
		case 'q':
			cont=0;
			break;
		case 's':
			cont=2;
			break;
		case 'c':
			cont=3;
			break;
		case 'h':
			cont=4;
			break;
		case 'm':
			cont=5;
			break;
		case 'p':
			cont=6;
			break;
		default:
			break;
	}
}
