#include "keyinter.h"
#include <process.h>
#include <windows.h>
#include <stdio.h>

void CallbackFunc(void *vp);

CKeyinter::CKeyinter()
{
	bStop = TRUE;
	m_cb = NULL;
	keysave = (int)'c';
}

CKeyinter::~CKeyinter(){}

void CKeyinter::Start()
{
	_beginthread(CallbackFunc, 0, this);
}

void CKeyinter::Stop()
{
	bStop = TRUE;
}

void CKeyinter::SetCallbackFunction(pfCallback cbFunc)
{
	m_cb = cbFunc;
}

void CallbackFunc(void *vp)
{
	int temp=0;
	CKeyinter* p = (CKeyinter*)vp;
	while(p->bStop!=FALSE)
	{
		p->keysave=fgetc(stdin);
		while((temp = getchar())!='\n');
		if(p->m_cb) p->m_cb();
	}
}
