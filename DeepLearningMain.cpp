#include <stdio.h>
#include <stdlib.h>
#include "training.h"
#include "dataread.h"
#define VERSION		 1.4
const char* pdl_error(int Err);

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
		printf(pdl_error(tmp));
		system("pause");
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
	
	// if inputs 1, load saved data of W and b
	// otherwise user inputs each dimension of layers
	// and randomly choose variables of W and b
	printf("to load previous weight, enter 1\n");
	printf("or to start with new random var, enter 0\n");
	printf(">> ");
	scanf("%d", &tmp);
	getchar();
	if(tmp) tmp = hTrain.WeightLoad();
	else if(valid) tmp = hTrain.WeightInit(100);
	else tmp = hTrain.WeightInit(1000);
	if(tmp)
	{
		printf(pdl_error(tmp));
		system("pause");
		return tmp;
	}
	printf("Weight parameters load done...\n");
	
	if(valid!=None)
	{
		printf("input trial hyperparam value\n>> ");
		double Try;
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
	
	printf("how many threads do you want to set?\n>> ");
	scanf("%d", &tmp);
	getchar();
	
	// start training
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

const char* pdl_error(int Err)
{
	switch(Err)
	{
		case ERR_NONE:
			return "no error occured\n";
		case ERR_UNAPPROPRIATE_INPUT:
			return "unappropriate input\n";
		case ERR_FILELOAD_FAILED:
			return "image file load failed\n";
		case ERR_FILE_DISCORDED:
			return "training images format and test set doesn't match'\n";
		case ERR_WRONG_DIMENSION:
			return "training image's dimension and test set's doesn't match\n";
		case ERR_WRONG_VALID_PARAM:
			return "unknown validation parameter chosen\n";
		case ERR_CRACKED_FILE:
			return "saved file cracked\n";
		default:
			return "unknown error code\n";
	}
}
