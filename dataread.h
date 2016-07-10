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
	LearningrateH
}ValidationParam;

class CDataread
{
private:
	int row, col;
	FileSetMode mode;
	int ReadMNISTTrainingSet(ValidationParam validateMode);
	int ReadMNISTTestSet();
public:
	double **x, **xt;
	unsigned char *y, *yt;
	int D0, N, M, Nt;
	CDataread();
	~CDataread();
	int SetMode(FileSetMode mod);
	int ReadTrainingSet(ValidationParam validateMode);
	int ReadTestSet();
};
#endif
