all:
	nvcc -arch=sm_35 -rdc=true matVec.cu -lcudadevrt -o matVec
	# nvcc -O3 -march=native matvec.cu -arch=sm_35 -o matvec
debug:
	nvcc -g -G -arch=sm_35 matVec.cu -o matVec
