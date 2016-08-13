#include "dataread.h"
#include "pdlerror.h"

CDataread::CDataread()
{
	mode = MNIST;
	N=0;
	Nt=0;
	row=0;
	col=0;
	x=NULL;
	xt=NULL;
	y=NULL;
	yt=NULL;
}

CDataread::~CDataread(){}

void CDataread::FreeData()
{
	int i;
	if(x!=NULL)
	{
		for(i=0; i<N; i++) free(x[i]);
		free(x);
	}
	if(y!=NULL) free(y);
	if(xt!=NULL)
	{
		for(i=0; i<Nt; i++) free(xt[i]);
		free(xt);
	}
	if(yt!=NULL) free(yt);
}

int CDataread::SetMode(FileSetMode mod)
{
	if(mode = mod) return 0;
	else return ERR_UNAPPROPRIATE_INPUT;
}

int CDataread::ReadTrainingSet(int validateMode)
{
	switch(mode)
	{
	case MNIST:
		return ReadMNISTTrainingSet(validateMode);
	case CIFAR10:
		return ReadCIFAR10TrainingSet(validateMode);
	default:
		return ERR_UNAPPROPRIATE_INPUT;
	}
}

int CDataread::ReadTestSet()
{
	switch(mode)
	{
	case MNIST:
		return ReadMNISTTestSet();
	case CIFAR10:
		return ReadCIFAR10TestSet();
	default:
		return ERR_UNAPPROPRIATE_INPUT;
	}
}

// if validateMode=0, load MNIST training set and return 0 if no error occurs
// if validateMode=1, load training set as validate set for 10%,
// and use last 90% for training set
// in this case, do not use test set
int CDataread::ReadMNISTTrainingSet(int validateMode)
{
	//if((validateMode!=0) && (validateMode!=1)) return ERR_UNAPPROPRIATE_INPUT;
	int i,j,tmp;
	double proc;
	// read file buffer
	FILE* fpTrainImage = fopen("MNIST/train-images.idx3-ubyte", "rb");
	FILE* fpTrainLabel = fopen("MNIST/train-labels.idx1-ubyte", "rb");
	fread(&tmp, sizeof(int), 1, fpTrainImage);	// read magic number
	if(tmp!=0x03080000) return ERR_FILELOAD_FAILED;
	fread(&tmp, sizeof(int), 1, fpTrainLabel);	// read magic number
	if(tmp!=0x01080000) return ERR_FILELOAD_FAILED;
	fread(&N, sizeof(int), 1, fpTrainImage);	// read N, the number of pictures
	fread(&tmp, sizeof(int), 1, fpTrainLabel);	// read N, the number of pictures
	if(tmp!=N) return ERR_FILE_DISCORDED;
	fread(&row, sizeof(int), 1, fpTrainImage);	// read number of rows
	fread(&col, sizeof(int), 1, fpTrainImage);	// read number of columns
	M = 10;	// number of possible labels
	D0 = row * col;	// dimension of each picture
	if(D0<=0) return ERR_WRONG_DIMENSION;
	
	// memory allocation of x, y
	if(validateMode)
	{
		Nt = N - N*0.9;
		N *= 0.9;
		// memory allocation of x, y
		xt = (double**) malloc(sizeof(double*) * Nt);
		yt = (unsigned char*) malloc(sizeof(unsigned char) * Nt);
		for(i=0; i<Nt; i++) xt[i] = (double*) malloc(sizeof(double) * D0);
		
		// scanning pictures and their answers
		for(i=0; i<Nt; i++)
		{
			for(j=0; j<D0; j++) xt[i][j] = (double) (fgetc(fpTrainImage) - 128)/128;
			yt[i] = (unsigned char) fgetc(fpTrainLabel);
			proc = (double) 100*i/N;
			printf("%2.2lf%%\b\b\b\b\b", proc);
			if(proc>9.995) printf("\b");
			if(proc>=99.995) printf("\b");
		}
	}
	x = (double**) malloc(sizeof(double*) * N);
	y = (unsigned char*) malloc(sizeof(double) * N);
	for(i=0; i<N; i++) x[i] = (double*) malloc(sizeof(double) * D0);
	
	// scanning pictures and their answers
	for(i=0; i<N; i++)
	{
		for(j=0; j<D0; j++) x[i][j] = (double) (fgetc(fpTrainImage) - 128)/128;
		y[i] = (unsigned char)fgetc(fpTrainLabel);
		if(validateMode) proc = (double) 100*(i+Nt)/(N+Nt);
		else proc = (double) 100*i/N;
		printf("%2.2lf%%\b\b\b\b\b", proc);
		if(proc>9.995) printf("\b");
		if(proc>=99.995) printf("\b");
	}
	printf("done...");
	fclose(fpTrainImage);
	fclose(fpTrainLabel);
	return 0;
}

int CDataread::ReadMNISTTestSet()
{
	int i, j, tmp;
	double proc;
	if(xt!=NULL || yt!=NULL) return 0;
	// Load test images and labels
	FILE* fpTestImage = fopen("MNIST/t10k-images.idx3-ubyte", "rb");
	FILE* fpTestLabel = fopen("MNIST/t10k-labels.idx1-ubyte", "rb");
	fread(&tmp, sizeof(int), 1, fpTestImage);	// read magic number
	if(tmp!=0x03080000) return ERR_FILELOAD_FAILED;
	fread(&tmp, sizeof(int), 1, fpTestLabel);	// read magic number
	if(tmp!=0x01080000) return ERR_FILELOAD_FAILED;
	fread(&Nt, sizeof(int), 1, fpTestImage);	// read Nt, the number of pictures
	fread(&tmp, sizeof(int), 1, fpTestLabel);	// read Nt, the number of pictures
	if(tmp!=Nt) return ERR_FILE_DISCORDED;
	fread(&tmp, sizeof(int), 1, fpTestImage);	// read number of rows
	if(tmp!=row) return ERR_WRONG_DIMENSION;
	fread(&tmp, sizeof(int), 1, fpTestImage);	// read number of columns
	if(tmp!=row) return ERR_WRONG_DIMENSION;
	
	// memory allocation of x, y
	xt = (double**) malloc(sizeof(double*) * Nt);
	yt = (unsigned char*) malloc(sizeof(unsigned char) * Nt);
	for(i=0; i<Nt; i++) xt[i] = (double*) malloc(sizeof(double) * D0);
	
	// scanning pictures and their answers
	for(i=0; i<Nt; i++)
	{
		for(j=0; j<D0; j++) xt[i][j] = (double) (fgetc(fpTestImage) - 128)/128;
		yt[i] = (unsigned char) fgetc(fpTestLabel);
		proc = (double) 100*i/Nt;
		printf("%2.2lf%%\b\b\b\b\b", proc);
		if(proc>9.995) printf("\b");
		if(proc>=99.995) printf("\b");
	}
	printf("done...");
	
	fclose(fpTestImage);
	fclose(fpTestLabel);
	
	return 0;
}

int CDataread::ReadCIFAR10TrainingSet(int validateMode)
{
	int i,j;
	double proc;
	FILE* fpTraining1 = fopen("cifar-10/data_batch_1.bin","rb");
	FILE* fpTraining2 = fopen("cifar-10/data_batch_2.bin","rb");
	FILE* fpTraining3 = fopen("cifar-10/data_batch_3.bin","rb");
	FILE* fpTraining4 = fopen("cifar-10/data_batch_4.bin","rb");
	FILE* fpTraining5 = fopen("cifar-10/data_batch_5.bin","rb");
	
	M = 10;
	D0 = 3072;
	
	if(validateMode) N = 40000;
	else N = 50000;
	x = (double**) malloc(sizeof(double*) * N);
	y = (unsigned char*) malloc(sizeof(unsigned char) * N);
	for(i=0; i<N; i++) x[i] = (double*) malloc(sizeof(double) * D0);
	
	for(i=0; i<10000; i++)
	{
		y[i] = (unsigned char) fgetc(fpTraining1);
		for(j=0; j<D0; j++) x[i][j] = (double) (fgetc(fpTraining1) - 128)/128;
		proc = (double) 100*i/50000;
		printf("%2.2lf%%\b\b\b\b\b", proc);
		if(proc>9.995) printf("\b");
		if(proc>=99.995) printf("\b");
	}
	
	for(; i<20000; i++)
	{
		y[i] = (unsigned char) fgetc(fpTraining2);
		for(j=0; j<D0; j++) x[i][j] = (double) (fgetc(fpTraining2) - 128)/128;
		proc = (double) 100*i/50000;
		printf("%2.2lf%%\b\b\b\b\b", proc);
		if(proc>9.995) printf("\b");
		if(proc>=99.995) printf("\b");
	}
	
	for(; i<30000; i++)
	{
		y[i] = (unsigned char) fgetc(fpTraining3);
		for(j=0; j<D0; j++) x[i][j] = (double) (fgetc(fpTraining3) - 128)/128;
		proc = (double) 100*i/50000;
		printf("%2.2lf%%\b\b\b\b\b", proc);
		if(proc>9.995) printf("\b");
		if(proc>=99.995) printf("\b");
	}
	
	for(; i<40000; i++)
	{
		y[i] = (unsigned char) fgetc(fpTraining4);
		for(j=0; j<D0; j++) x[i][j] = (double) (fgetc(fpTraining4) - 128)/128;
		proc = (double) 100*i/50000;
		printf("%2.2lf%%\b\b\b\b\b", proc);
		if(proc>9.995) printf("\b");
		if(proc>=99.995) printf("\b");
	}
	
	if(validateMode)
	{
		Nt = 10000;
		xt = (double**) malloc(sizeof(double*) * Nt);
		yt = (unsigned char*) malloc(sizeof(unsigned char) * Nt);
		for(i=0; i<Nt; i++) xt[i] = (double*) malloc(sizeof(double) * D0);
		for(i=0; i<Nt; i++)
		{
			yt[i] = (unsigned char) fgetc(fpTraining5);
			for(j=0; j<D0; j++) xt[i][j] = (double) (fgetc(fpTraining5) - 128)/128;
			proc = (double) 100*(i+N)/50000;
			printf("%2.2lf%%\b\b\b\b\b", proc);
			if(proc>9.995) printf("\b");
			if(proc>=99.995) printf("\b");
		}
	}
	else
	{
		for(i=40000; i<N; i++)
		{
			y[i] = (unsigned char) fgetc(fpTraining5);
			for(j=0; j<D0; j++) x[i][j] = (double) (fgetc(fpTraining5) - 128)/128;
			proc = (double) 100*i/50000;
			printf("%2.2lf%%\b\b\b\b\b", proc);
			if(proc>9.995) printf("\b");
			if(proc>=99.995) printf("\b");
		}
	}
	printf("done...");
	
	fclose(fpTraining1);
	fclose(fpTraining2);
	fclose(fpTraining3);
	fclose(fpTraining4);
	fclose(fpTraining5);
	
	return 0;
}

int CDataread::ReadCIFAR10TestSet()
{
	int i,j;
	double proc;
	FILE* fpTest = fopen("cifar-10/test_batch.bin","rb");
	
	D0 = 3072;
	Nt = 10000;
	
	// memory allocation of x, y
	xt = (double**) malloc(sizeof(double*) * Nt);
	yt = (unsigned char*) malloc(sizeof(unsigned char) * Nt);
	for(i=0; i<Nt; i++) xt[i] = (double*) malloc(sizeof(double) * D0);
	
	for(i=0; i<Nt; i++)
	{
		yt[i] = (unsigned char) fgetc(fpTest);
		for(j=0; j<D0; j++) xt[i][j] = (double) (fgetc(fpTest) - 128)/128;
		proc = (double) 100*i/Nt;
		printf("%2.2lf%%\b\b\b\b\b", proc);
		if(proc>9.995) printf("\b");
		if(proc>=99.995) printf("\b");
	}
	printf("done...");
	
	fclose(fpTest);
	
	return 0;
}
