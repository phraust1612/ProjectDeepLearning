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
#define LAMBDADEFAULT	0.003
#define HDEFAULT		0.0001


class CTraining
{
private:
	// N is the number of training sets
	// and Nt is the number of test sets
	int alpha, N, Nt, *D, count, l, learningSize, loaded;
	// H, DELTA, LAMBDA are hyperparameters
	// dW, db each stands for ds/dW, ds/db matrices
	double H, ***W, **b, *****dW, ***dLdW, ****db, **dLdb, L, Lold, DELTA, LAMBDA;
	time_t starttime, endtime;
	CDataread *pData;
	void ParamAllocate();
	CKeyinter Key;
public:
	
	CTraining(CDataread* pD);
	~CTraining();
	int WeightInit(int size);
	int WeightLoad();
	void Training(int threads);
	void FileSave();
	void ShowHelp();
	void TrainingThreadFunc(int l);
	int SetHyperparam(ValidationParam validateMode, double hyperparam);
	double CheckAccuracy();
};
#endif
