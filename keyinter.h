#ifndef _KEYINTER_H
#define _KEYINTER_H
#include <process.h>
#include <windows.h>
#include <stdio.h>
#include <conio.h>

typedef void (*pfCallback)();

class CKeyinter
{
public:
	pfCallback m_cb;
	int keysave;
	bool bStop;
	CKeyinter();
	~CKeyinter();
	void Start();
	void Stop();
};
#endif
