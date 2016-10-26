all:
	g++ GraphCannySeg.cpp -std=c++11 -c 
	nvcc -lassimp `gsl-config --cflags  --libs` `pkg-config --cflags --libs opencv` -use_fast_math -lineinfo -O3 -std=c++11 GraphCannySeg.o main.cu -o main




