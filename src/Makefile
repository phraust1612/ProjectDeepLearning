# ProjectDeepLearning

CPP      = g++.exe
OBJ      = DeepLearningMain.o training.o dataread.o keyinter.o
LIBS     = -L"C:/Program Files/Dev-Cpp/MinGW64/x86_64-w64-mingw32/lib32" -m32
INCS     = -I"C:/Program Files/Dev-Cpp/MinGW64/include" -I"C:/Program Files/Dev-Cpp/MinGW64/x86_64-w64-mingw32/include" -I"C:/Program Files/Dev-Cpp/MinGW64/lib/gcc/x86_64-w64-mingw32/4.9.2/include" -I"C:/Program Files/Dev-Cpp/MinGW64/lib/gcc/x86_64-w64-mingw32/4.9.2/include/c++"
CFLAGS   = $(INCS) -m32 -std=gnu++11
BIN      = ../PDL.exe
RM       = rm.exe -f

.PHONY: all all-before all-after clean clean-custom

all: all-before $(BIN) all-after

clean: clean-custom
	${RM} $(OBJ)

$(BIN): $(OBJ)
	$(CPP) $(OBJ) -o $(BIN) $(LIBS)

keyinter.o: keyinter.cpp
	$(CPP) -c keyinter.cpp -o keyinter.o $(CFLAGS)

DeepLearningMain.o: DeepLearningMain.cpp
	$(CPP) -c DeepLearningMain.cpp -o DeepLearningMain.o $(CFLAGS)

training.o: training.cpp
	$(CPP) -c training.cpp -o training.o $(CFLAGS)

dataread.o: dataread.cpp
	$(CPP) -c dataread.cpp -o dataread.o $(CFLAGS)
