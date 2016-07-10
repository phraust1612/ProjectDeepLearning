#include <stdio.h>
#include <stdlib.h>
#include "training.h"
#include "dataread.h"
#define VERSION		 1.3

int main()
{
	printf("DeepLearning ver %1.1f start\n", VERSION);
	int tmp;
	FileSetMode filemode;
	ValidationParam valid;
	CDataread hData = CDataread();
	printf("MNIST : 1\n");
	printf(">> ");
	scanf("%d", &filemode);
	getchar();
	tmp = hData.SetMode(filemode);
	if(tmp)
	{
		printf("error %d occured!!!\n", tmp);
		return tmp;
	}
	
	printf("to validate DELTA, input 1\n");
	printf("to validate LAMBDA, input 2\n");
	printf("to validate H, input 3\n");
	printf("not to make validation set, input 0\n");
	printf(">> ");
	scanf("%d", &valid);
	getchar();
	printf("loading training set...\n");
	tmp = hData.ReadTrainingSet(valid);
	if(tmp)
	{
		printf("error %d occured!!!\n", tmp);
		return tmp;
	}
	if(!valid)
	{
		printf("loading test set...\n");
		hData.ReadTestSet();
		if(tmp)
		{
			printf("error %d occured!!!\n", tmp);
			return tmp;
		}
	}
	printf("scanning all picture done...\n");
	
	CTraining hTrain = CTraining(&hData);
	
	// if inputs 1, load saved data of W and b
	// otherwise user inputs each dimension of layers
	// and randomly choose variables of W and b
	printf("to load previous weight, enter 1\n");
	printf("or to start with new random var, enter 0\n");
	printf(">> ");
	scanf("%d", &tmp);
	getchar();
	if(tmp) hTrain.WeightLoad();
	else if(valid) hTrain.WeightInit(100);
	else hTrain.WeightInit(1000);
	printf("Weight parameters load done...\n");
	
	// allocates memories of W, b, etc
	hTrain.ParamAllocate();
	printf("memory allocation done...\n");
	
	if(valid!=None)
	{
		printf("input trial hyperparam value\n>> ");
		double Try;
		scanf("%lf", &Try);
		getchar();
		tmp = hTrain.SetHyperparam(valid, Try);
		if(tmp)
		{
			printf("error %d occured!!!\n", tmp);
			return tmp;
		}
	}
	// start training
	printf("start learning procedure...\n");
	hTrain.ShowHelp();
	hTrain.Training();
	
	// save trained parameters
	printf("learning ended...\n");
	hTrain.FileSave();
	double acc = hTrain.CheckAccuracy();
	printf("accuracy : %2.2lf%%!!!\n", acc);
	
	// free memories
	// hTrain.~CTraining();
	printf("end of the programm!!\n");
	
	system("pause");
	return 0;
}
