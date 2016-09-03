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
// and this CUDA programing is not ready yet (16.09.01 ver 2.2)
#if CUDAEXIST
#include <cuda.h>
#endif
class CTraining
{
private:
	// saverfilename : Loaded file name string
	// automode : holded command
	char *savefilename, automode;
	// A,B,C,D,F,S,P are hyperparameters to decide Wieght's initial size
	// please check README file for more detail
	// alpha is the number of FC layers including score layer
	// beta is the number of Conv and Pooling layers
	// N is the number of training sets
	// and Nt is the number of test sets
	// loaded is 1 when it first loads file
	// count is an index of nth learning
	// l is an index of nth images
	int A, B, C, *D, *F, *S, *P, alpha, beta, N, Nt, count, l, learningSize, loaded;
	// each width, height, depth corresponds to nth layer's
	// sizeX is the sum of X's size under nth X
	int *width, *height, *depth, *sizeW, *sizeb, *sizes, *sizeConvX, *sizeConvW, *sizeConvb, *sizePool;
	// W,b,ConvW,Convb are weight and bias parameters for each FC and Conv layers
	// dW, db each stands for ds/dW, ds/db matrices
	// dLdW, dLdb corresponds to dL/dW, dL/db
	// vecdW, vecdb are used for momentum update
	double *W, *b, *dLdW, *dLdb, *vecdW, *vecdb, *ConvW, *Convb, *ConvdLdW, *ConvdLdb, *vecConvdW, *vecConvdb;
	// L is loss function value, Lold is previous loss value
	// other uppercase words are hyperparameters
	// check README for more detail
	double H, L, Lold, DELTA, LAMBDA, MOMENTUMUPDATE;
	CDataread *pData;
	// allocate memories for gradient and weights
	void ParamAllocate();
	// W^i_j,k = W[indexOfW(i,j,k)]
	// b^i,j = b[indexOfb(i,j)]
	// X^i,j = X[indexOfs(i,j)]
	// ds^alpha_i / dW^m_j,k = dW[indexOfdW(m,i,j,k)]
	// ds^alpha_i / db^m_j = db[indexOfdb(m,i,j)]
	// ConvX^m_i,j,k = ConvX[indexOfConvX(m,i,j,k)]
	// Pool^m_i,j,k = Pool[indexOfPool(m,i,j,k)]
	// ConvW^m,k2_i,j,k = ConvW[indexOfConvW(m,k2,i,j,k)]
	// Convb^m_k2 = Convb[indexOfConvb(m,k2)]
	// ds^alpha_u / dConvX^m_i,j,k = ConvdX[indexOfConvdX(u,m,i,j,k)]
	// ds^alpha_u / dConvW^m,k2_i,j,k = ConvdW[indexOfConvdW(u,m,k2,i,j,k)]
	// ds^alpha_u / dConvb^m_k2 = Convdb[indexOfConvdb(u,m,k2)]
	// Check my blog for index range - http://blog.naver.com/phraust1612
	int indexOfW(int i, int j, int k);
	int indexOfb(int i, int j);
	int indexOfs(int i, int j);
	int indexOfdW(int m, int i, int j, int k);
	int indexOfdb(int m, int i, int j);
	int indexOfConvX(int u, int i, int j, int k);
	int indexOfConvW(int u, int v, int i, int j, int k);
	int indexOfConvb(int u, int v);
	int indexOfPool(int m, int i, int j, int k);
	int indexOfConvdX(int u, int m, int i, int j, int k);
	int indexOfConvdW(int u, int m, int v, int i, int j, int k);
	int indexOfConvdb(int u, int m, int v);
	double GradientCheck();
	int RNNThreadFunc(int index);
	// callee function runs on a thread, called from CheckAccuracy func
	int CheckThreadFunc(int index);
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
	// one of WeightInit or WeightLoad must be called before Training starts
	int WeightInit(int size, char* argv);
	int WeightLoad(char* argv);
	int WeightSave();
	// caller function of each threads and intercept key press during the loop
	// you can set the number of threads to use
	void Training(int threads);
	// save hyperparameters, weights and gradients into a binary file
	void FileSave();
	void ShowHelp();
	// callee function : run a training loop for given index'th imaage
	// RNNThread is only called when both A and B are 0 from CNNThreadFunc
	void CNNThreadFunc(int index);
	// call this function after whole training procedure is done to free memories
	void FreeMem();
	// public member function to set hyperparameter value
	// lPar isn't used yet
	int SetHyperparam(ValidationParam validateMode, int lPar, double hyperparam);
	// CheckAccuracy returns the accuracy of current weight parameters
	double CheckAccuracy(int threads);
};
#endif
