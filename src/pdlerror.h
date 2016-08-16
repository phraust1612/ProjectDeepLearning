#ifndef _PDLERROR_H
#define _PDLERROR_H
#define ERR_NONE				0x00000000
#define EXC_TRAININGDONE		0x00000001
#define ERR_UNAPPROPRIATE_INPUT	0x80000001
#define ERR_FILELOAD_FAILED		0x80000002
#define	ERR_FILE_DISCORDED		0x80000003
#define ERR_WRONG_DIMENSION		0x80000004
#define ERR_WRONG_VALID_PARAM	0x80000005
#define ERR_CRACKED_FILE		0x80000006
#define ERR_UNAPPROPTHREADS		0x80000007
#define ERR_WRONGINDEXW			0x80000008
#define ERR_WRONGINDEXB			0x80000009
#define ERR_WRONGINDEXS			0x80000010
#define ERR_WRONGINDEXDW		0x80000011
#define ERR_WRONGINDEXDB		0x80000012
#define ERR_WRONGPROGRAMEXEC	0x80000013

static const char* pdl_error(int Err)
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
		case ERR_UNAPPROPTHREADS:
			return "you must use at least one thread!\n";
		default:
			return "unknown error code\n";
	}
}
#endif
