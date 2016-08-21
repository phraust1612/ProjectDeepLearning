#include <stdio.h>
#include <stdlib.h>
#include "training.h"
#include "dataread.h"
#define VERSION	2.1

int main(int argc, char* argv[])
{
	printf("DeepLearning ver %1.1f start\n", VERSION);
	int tmp, err;
	double Try;
	FileSetMode filemode;
	ValidationParam valid;
	CDataread hData = CDataread();
	
	// step 1 : choose which dataset to use
	printf("MNIST : 1\n");
	printf("CIFAR-10 : 2\n");
	printf(">> ");
	scanf("%d", &filemode);
	getchar();
	err = hData.SetMode(filemode);
	if(err < 0)
	{
		printf(pdl_error(err));
		system("pause");
		return err;
	}
	
	// step 2 : choose whether use validation set or not
	printf("to use validation set, input 1\n");
	printf("otherwise input 0\n>> ");
	scanf("%d", &tmp);
	getchar();
	printf("loading training set...");
	err = hData.ReadTrainingSet(tmp);
	if(err < 0)
	{
		printf(pdl_error(err));
		hData.FreeData();
		system("pause");
		return err;
	}
	if(!tmp)
	{
		printf("\nloading test set...");
		err = hData.ReadTestSet();
		if(err < 0)
		{
			printf(pdl_error(err));
			hData.FreeData();
			system("pause");
			return err;
		}
	}
	printf("\nscanning all picture done...\n");
	
	CTraining hTrain = CTraining(&hData);
	
	switch(argc)
	{
		case 2:
			// automatically load weight parameters
			err = (int) hTrain.WeightLoad(argv[1]);
			break;
		case 1:
			// step 3: set your own file's name
			// it will initiate weight parameters randomly
			char savefilename[50];
			printf("set the name of auto saved file\n>> ");
			scanf("%s", savefilename);
			getchar();
			// step 4 : set learning size
			printf("set learning size\n>> ");
			scanf("%d", &tmp);
			getchar();
			err = hTrain.WeightInit(tmp,savefilename);
			break;
		default:
			err = ERR_WRONGPROGRAMEXEC;
			break;
	}
	
	if(err == EXC_TRAININGDONE)
	{
		printf("training proc's already done, check accuracy and end?\n>> ");
		scanf("%d", &tmp);
		getchar();
		if(tmp)
		{
			Try = hTrain.CheckAccuracy();
			printf("accuracy : %2.2lf%%!\n", Try);
			system("pause");
			return err;
		}
	}
	if(err < 0)
	{
		printf(pdl_error(err));
		hTrain.FreeMem();
		system("pause");
		return err;
	}
	
	// step 5 : choose whetherinput your own hyperparameter values or not
	valid = None;
	while(true)
	{
		printf("to modify DELTA, input 1\n");
		printf("to modify LAMBDA, input 2\n");
		printf("to modify H, input 3\n");
		printf("to modify learning size, input 4\n");
		printf("to modify momentum update constance, input 5\n");
		printf("to start, input 0\n>> ");
		scanf("%d", &valid);
		getchar();
		if(valid == None) break;
		printf("input your hyperparam value\n>> ");
		scanf("%lf", &Try);
		getchar();
		err = hTrain.SetHyperparam(valid, 0, Try);
		if(err < 0)
		{
			printf(pdl_error(err));
			hTrain.FreeMem();
			system("pause");
			return err;
		}
	}
	
	// step 6 : choose how many threads you're gonna use
	printf("how many threads do you want to set?\n>> ");
	scanf("%d", &tmp);
	getchar();
	if(tmp<1)
	{
		err = ERR_UNAPPROPTHREADS;
		printf(pdl_error(err));
		hTrain.FreeMem();
		system("pause");
		return err;
	}
	
	// step 7 : start training
	// input q to quit or other commands
	printf("start learning procedure...");
	printf("\npress h for help");
	hTrain.Training(tmp);
	
	// save trained parameters
	printf("\nlearning ended...");
	hTrain.FileSave();
	
	// free memories
	hTrain.FreeMem();
	printf("\nend of the programm!!");
	return 0;
}
