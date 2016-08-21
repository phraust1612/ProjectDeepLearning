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
#define MUDEFAULT		0.9
#define CUDAEXIST	0
// set CUDAEXIST 1 if you've installed cuda before compiling
// otherwise set 0
#if CUDAEXIST
#include <cuda.h>
#endif
typedef struct
{
	int *width,
	int *height,
	int *depth
}ConvSizeStruct;

class CTraining
{
private:
	char *savefilename, automode;
	// alpha is the number of layers including hidden layers and the final score layer
	// N is the number of training sets
	// and Nt is the number of test sets
	// D is the dimension of each layer (D[0] becomes the dimension of input layer)
	int alpha, N, Nt, *D, count, l, learningSize, loaded;
	// each value is the size of W,b,s
	int sizeW, sizeb, sizes;
	// dW, db each stands for ds/dW, ds/db matrices
	// dLdW, dLdb corresponds to dL/dW, dL/db
	// vecdW, vecdb are used for momentum update
	// olddLdW, olddLdb, oldvecdW, oldvecdb are used for gradient check
	double *W, *b, *dLdW, *dLdb, *vecdW, *vecdb;
	// DELTA, LAMBDA, MOMENTUMUPDATE are hyperparameters
	// L is loss function value, Lold is previous loss value
	// H is the learning rate, which is also kind of hyperparameters
	double H, L, Lold, DELTA, LAMBDA, MOMENTUMUPDATE;
	CDataread *pData;
	void ParamAllocate();
	int indexOfW(int i, int j, int k);
	int indexOfb(int i, int j);
	int indexOfs(int i, int j);
	int indexOfdW(int m, int i, int j, int k);
	int indexOfdb(int m, int i, int j);
	double GradientCheck();
	CKeyinter Key;
	ConvSizeStruct Size;
#if CUDAEXIST
#define CUDABLOCKS	1000
	cudaError_t cuda_err;
	int cudadevice;
	cudaDeviceProp deviceProp;
#endif
public:
	
	CTraining(CDataread* pD);
	~CTraining();
	int WeightInit(int size, char* argv);
	int WeightLoad(char* argv);
	int WeightSave();
	void Training(int threads);
	void FileSave();
	void ShowHelp();
	int TrainingThreadFunc(int index);
	void ConvThreadFunc();
	void FreeMem();
	int SetHyperparam(ValidationParam validateMode, int lPar, double hyperparam);
	double CheckAccuracy();
};
#endif
