// this program transfers old parameter files to the newest version
#include <stdio.h>
#include <stdlib.h>

int main()
{
	int i, j, k, sizeW, sizeb, alpha, *D, count, learningSize, l;
	double *W, *b, *dLdW, *dLdb, *H, DELTA, LAMBDA, L, Lold;
	char name[20];
	
	printf("file name : ");
	scanf("%s", name);
	getchar();
	
	FILE *fp = fopen(name, "rb");
	
	fread(&alpha, sizeof(int), 1, fp);
	if(alpha<1) return 1;
	
	D = (int*) malloc(sizeof(int) * (alpha+1));
	H = (double*) malloc(sizeof(double) * alpha);
	
	fread(D, sizeof(int), alpha+1, fp);
	
	sizeW = 0;
	sizeb = 0;
	for(i=0; i<alpha; i++)
	{
		sizeW += D[i+1] * D[i];
		sizeb += D[i+1];
	}
	
	W = (double*) malloc(sizeof(double) * sizeW);
	b = (double*) malloc(sizeof(double) * sizeb);
	dLdW = (double*) malloc(sizeof(double) * sizeW);
	dLdb = (double*) malloc(sizeof(double) * sizeb);
	
	fread(W, sizeof(double), sizeW, fp);
	fread(b, sizeof(double), sizeb, fp);
	fread(H, sizeof(double), 1, fp);
	for(i=1; i<alpha; i++) H[i] = H[i-1] * H[0];
	fread(&DELTA, sizeof(double), 1, fp);
	fread(&LAMBDA, sizeof(double), 1, fp);
	fread(&count, sizeof(int), 1, fp);
	fread(&learningSize, sizeof(int), 1, fp);
	fread(&l, sizeof(int), 1, fp);
	fread(&L, sizeof(double), 1, fp);
	fread(&Lold, sizeof(double), 1, fp);
	fread(dLdW, sizeof(double), sizeW, fp);
	fread(dLdb, sizeof(double), sizeb, fp);
	fclose(fp);
	
	FILE* fpResult = fopen(name, "wb");
	char magic[2];
	int ver[2];
	magic[0] = 'P';
	magic[1] = 'D';
	ver[0] = 1;
	ver[1] = 9;
	fwrite(magic, sizeof(char), 2, fpResult);
	fwrite(ver, sizeof(int), 2, fpResult);
	fwrite(&alpha, sizeof(int), 1, fpResult);
	fwrite(D, sizeof(int), alpha+1, fpResult);
	fwrite(H, sizeof(double), alpha, fpResult);
	fwrite(&DELTA, sizeof(double), 1, fpResult);
	fwrite(&LAMBDA, sizeof(double), 1, fpResult);
	fwrite(&count, sizeof(int), 1, fpResult);
	fwrite(&learningSize, sizeof(int), 1, fpResult);
	fwrite(&l, sizeof(int), 1, fpResult);
	fwrite(&L, sizeof(double), 1, fpResult);
	fwrite(&Lold, sizeof(double), 1, fpResult);
	fwrite(W, sizeof(double), sizeW, fpResult);
	fwrite(b, sizeof(double), sizeb, fpResult);
	fwrite(dLdW, sizeof(double), sizeW, fpResult);
	fwrite(dLdb, sizeof(double), sizeb, fpResult);
	fclose(fpResult);
	free(W);
	free(dLdW);
	free(b);
	free(dLdb);
	free(D);
	free(H);
	
	return 0;
}
