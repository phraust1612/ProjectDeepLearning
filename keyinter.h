#ifndef _KEYINTER_H
#define _KEYINTER_H

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
	void SetCallbackFunction(pfCallback cbFunc);
};
#endif
