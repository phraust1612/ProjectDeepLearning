#ifndef _DATAREAD_H
#define _DATAREAD_H
#include <stdio.h>
#include <stdlib.h>
typedef enum
{
	MNIST=1,
	CIFAR10
} FileSetMode;

typedef enum
{
	None=0,
	Delta,
	Lambda,
	LearningrateH,
	LearningSize
}ValidationParam;

class CDataread
{
private:
	int row, col;
	FileSetMode mode;
	int ReadMNISTTrainingSet(int validateMode);
	int ReadMNISTTestSet();
	int ReadCIFAR10TrainingSet(int validateMode);
	int ReadCIFAR10TestSet();
public:
	double **x, **xt;
	unsigned char *y, *yt;
	int D0, N, M, Nt;
	CDataread();
	~CDataread();
	int SetMode(FileSetMode mod);
	int ReadTrainingSet(int validateMode);
	int ReadTestSet();
};
#endif
