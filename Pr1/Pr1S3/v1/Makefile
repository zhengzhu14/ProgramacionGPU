CXX = g++
CC = g++
NVCC = nvcc

CFLAGS=-O3 -I. -I/usr/local/cuda/include 
CXXFLAGS=-O3 -I. -I/usr/local/cuda/include

LDFLAGS = -L/usr/lib/wsl/lib -L/usr/local/cuda-12.8/targets/x86_64-linux/lib
LIBS = -lm -lcuda -lcudart

SRC =  main.o matrix_mul.o
	
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $

%.o: %.cu
	$(NVCC) $(CFLAGS) -c -o $@ $

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $

matrix_mul: $(SRC) 
	$(CC) -o matrix_mult.out  $(SRC) $(CXXFLAGS) $(LDFLAGS) $(LIBS) 

clean:
	rm -f *.o matrix_mult