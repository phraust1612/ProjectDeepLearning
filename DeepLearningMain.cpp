#include <stdio.h>
#include <stdlib.h>
#include "training.h"
#include "dataread.h"
#define VERSION		 1.7

int main()
{
	printf("DeepLearning ver %1.1f start\n", VERSION);
	int tmp;
	double Try;
	FileSetMode filemode;
	ValidationParam valid;
	CDataread hData = CDataread();
	
	// step 1 : choose which dataset to use
	printf("MNIST : 1\n");
	printf(">> ");
	scanf("%d", &filemode);
	getchar();
	tmp = hData.SetMode(filemode);
	if(tmp)
	{
		printf(pdl_error(tmp));
		system("pause");
		return tmp;
	}
	
	// step 2 : choose whether use validation set or not
	printf("to use validation set, input 1\n");
	printf("otherwise input 0\n>> ");
	scanf("%d", &tmp);
	getchar();
	printf("loading training set...\n");
	tmp = hData.ReadTrainingSet(tmp);
	if(tmp)
	{
		printf(pdl_error(tmp));
		system("pause");
		return tmp;
	}
	if(!valid)
	{
		printf("loading test set...\n");
		hData.ReadTestSet();
		if(tmp)
		{
			printf(pdl_error(tmp));
			system("pause");
			return tmp;
		}
	}
	printf("scanning all picture done...\n");
	
	CTraining hTrain = CTraining(&hData);
	
	// step 3 : choose whether load previous weights or not
	// if inputs 1, load saved data of W and b
	// otherwise user inputs each dimension of layers
	// and randomly choose variables of W and b
	printf("to load previous weight, enter 1\n");
	printf("or to start with new random var, enter 0\n");
	printf(">> ");
	scanf("%d", &tmp);
	getchar();
	if(tmp) tmp = hTrain.WeightLoad();
	else
	{
		// step 4 : set learning size
		printf("set learning size\n>> ");
		scanf("%d", &tmp);
		getchar();
		tmp = hTrain.WeightInit(tmp);
	}
	if(tmp)
	{
		printf(pdl_error(tmp));
		system("pause");
		return tmp;
	}
	printf("Weight parameters load done...\n");
	
	// step 5 : choose whetherinput your own hyperparameter values or not
	valid = None;
	while(true)
	{
		printf("to validate DELTA, input 1\n");
		printf("to validate LAMBDA, input 2\n");
		printf("to validate H, input 3\n");
		printf("to start, input 0\n>> ");
		scanf("%d", &valid);
		getchar();
		if(valid == None) break;
		printf("input your hyperparam value\n>> ");
		scanf("%lf", &Try);
		getchar();
		tmp = hTrain.SetHyperparam(valid, Try);
		if(tmp)
		{
			printf(pdl_error(tmp));
			system("pause");
			return tmp;
		}
	}
	
	// step 6 : choose how many threads you're gonna use
	printf("how many threads do you want to set?\n>> ");
	scanf("%d", &tmp);
	getchar();
	if(tmp<1)
	{
		printf("you must use at least one thread!\n");
		system("pause");
		return tmp;
	}
	
	// step 7 : start training
	// input q to quit or other commands
	printf("start learning procedure...\n");
	hTrain.ShowHelp();
	hTrain.Training(tmp);
	
	// save trained parameters
	printf("learning ended...\n");
	hTrain.FileSave();
	
	// free memories
	// hTrain.~CTraining();
	printf("end of the programm!!\n");
	return 0;
}
