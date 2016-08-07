#ifndef _PDLERROR_H
#define _PDLERROR_H
#define ERR_NONE				0x0000
#define ERR_UNAPPROPRIATE_INPUT	0xF001
#define ERR_FILELOAD_FAILED		0xF002
#define	ERR_FILE_DISCORDED		0xF003
#define ERR_WRONG_DIMENSION		0xF004
#define ERR_WRONG_VALID_PARAM	0xF005
#define ERR_CRACKED_FILE		0xF006
#define ERR_WRONGINDEX			0xF007

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
		default:
			return "unknown error code\n";
	}
}
#endif
