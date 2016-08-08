#ifndef _TRAINING_H
#define _TRAINING_H
#include "keyinter.h"
#include "dataread.h"
#include "pdlerror.h"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#define DELTADEFAULT	1.0
#define LAMBDADEFAULT	0.001
#define HDEFAULT		0.001
#define CUDAEXIST	0
// set CUDAEXIST 1 if you've installed cuda before compiling
// otherwise set 0
#if CUDAEXIST
#include <cuda.h>
#endif

class CTraining
{
private:
	// N is the number of training sets
	// and Nt is the number of test sets
	int alpha, N, Nt, *D, count, l, learningSize, loaded;
	int sizeW, sizeb, sizes, sizedW, sizedb;
	// H, DELTA, LAMBDA are hyperparameters
	// dW, db each stands for ds/dW, ds/db matrices
	double H, *W, *b, *dLdW, *dLdb, L, Lold, DELTA, LAMBDA;
	CDataread *pData;
	void ParamAllocate();
	int indexOfW(int i, int j, int k);
	int indexOfb(int i, int j);
	int indexOfs(int i, int j);
	int indexOfdW(int m, int i, int j, int k);
	int indexOfdb(int m, int i, int j);
	CKeyinter Key;
#if CUDAEXIST
#define CUDABLOCKS	1000
	cudaError_t cuda_err;
	int cudadevice;
	cudaDeviceProp deviceProp;
#endif
public:
	
	CTraining(CDataread* pD);
	~CTraining();
	int WeightInit(int size);
	int WeightLoad();
	void Training(int threads);
	void FileSave();
	void ShowHelp();
	void TrainingThreadFunc(int index, int targetlayer);
	int SetHyperparam(ValidationParam validateMode, double hyperparam);
	double CheckAccuracy();
};
#endif
