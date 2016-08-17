#include <stdio.h>
#include <stdlib.h>

int indexOfW(int i, int j, int k, int *D)
{	
	int t, ans;
	ans = 0;
	for(t=0; t<i; t++) ans += D[t+1] * D[t];
	ans += D[i] * j;
	ans += k;
	
	return ans;
}

int main()
{
	int i, j, k, err, sizeW, sizeb, alpha, *D, count, learningSize, l, cursor;
	double *W, *b, *dLdW, *dLdb, *H, DELTA, LAMBDA, L, Lold;
	char name[40];
	
	printf("file name : ");
	scanf("%s", name);
	getchar();
	
	FILE *fp = fopen(name, "rb");
	
	char magic1,magic2;
	magic1 = (char) fgetc(fp);
	magic2 = (char) fgetc(fp);
	if(magic1 != 'P' || magic2 != 'D')
	{
		fclose(fp);
		return 1;
	}
	
	// read version
	int ver[2];
	fread(ver, sizeof(int), 2, fp);
	if(ver[0] != 1 || ver[1] != 9)
	{
		fclose(fp);
		return 1;
	}
	
	// scan alpha - the number of layers including score layer
	fread(&alpha, sizeof(int), 1, fp);
	
	// scan hyperparameters, etc.
	D = (int*) malloc(sizeof(int) * (alpha+1));
	H = (double*) malloc(sizeof(double) * alpha);
	fread(D, sizeof(int), alpha+1, fp);
	fread(H, sizeof(double), alpha, fp);
	
	sizeW = 0;
	sizeb = 0;
	for(i=0; i<alpha; i++)
	{
		sizeW += D[i+1] * D[i];
		sizeb += D[i+1];
	}
	
	fread(&DELTA, sizeof(double), 1, fp);
	fread(&LAMBDA, sizeof(double), 1, fp);
	fread(&count, sizeof(int), 1, fp);
	fread(&learningSize, sizeof(int), 1, fp);
	if(count == learningSize) err = 2;
	fread(&l, sizeof(int), 1, fp);
	fread(&L, sizeof(double), 1, fp);
	fread(&Lold, sizeof(double), 1, fp);
	
	// allocate memories
	W = (double*) malloc(sizeof(double) * sizeW);
	b = (double*) malloc(sizeof(double) * sizeb);
	dLdW = (double*) malloc(sizeof(double) * sizeW);
	dLdb = (double*) malloc(sizeof(double) * sizeb);
	
	// scan W,b etc
	fread(W, sizeof(double), sizeW, fp);
	fread(b, sizeof(double), sizeb, fp);
	// scan gradients
	//fread(dLdW, sizeof(double), sizeW, fp);
	//fread(dLdb, sizeof(double), sizeb, fp);
	for(i=0; i<sizeW; i++) dLdW[i] = 0;
	for(i=0; i<sizeb; i++) dLdb[i] = 0;
	fclose(fp);
	
	fp = fopen(name, "wb");
	fwrite(&magic1, sizeof(char), 1, fp);
	fwrite(&magic2, sizeof(char), 1, fp);
	
	// read version
	fwrite(ver, sizeof(int), 2, fp);
	
	// scan alpha - the number of layers including score layer
	fwrite(&alpha, sizeof(int), 1, fp);
	
	// scan hyperparameters, etc.
	fwrite(D, sizeof(int), alpha+1, fp);
	fwrite(H, sizeof(double), alpha, fp);
	
	fwrite(&DELTA, sizeof(double), 1, fp);
	fwrite(&LAMBDA, sizeof(double), 1, fp);
	fwrite(&count, sizeof(int), 1, fp);
	fwrite(&learningSize, sizeof(int), 1, fp);
	l = 0;
	L = 0;
	fwrite(&l, sizeof(int), 1, fp);
	fwrite(&L, sizeof(double), 1, fp);
	fwrite(&Lold, sizeof(double), 1, fp);
	
	// scan W,b etc
	fwrite(W, sizeof(double), sizeW, fp);
	fwrite(b, sizeof(double), sizeb, fp);
	// scan gradients
	fwrite(dLdW, sizeof(double), sizeW, fp);
	fwrite(dLdb, sizeof(double), sizeb, fp);
	
	fclose(fp);
	free(D);
	free(H);
	free(W);
	free(b);
	free(dLdW);
	free(dLdb);
	
	return err;
}
