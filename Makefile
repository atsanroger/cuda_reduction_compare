################################################################################
#        @author  Wei-Lun Chen (wlchen) 
#                 $LastChangedBy: wlchen $
#        @date    $LastChangedDate:: 2024-10-02 19:34:22 #$
#        @version $LastChangedRevision: 2562 $
################################################################################
SHELL := /bin/bash

ARCH = -gencode=arch=compute_60,code=sm_60 \
			 -gencode=arch=compute_70,code=sm_70 \
       -gencode=arch=compute_80,code=sm_80 \
       -gencode=arch=compute_86,code=sm_86 \
       -gencode=arch=compute_90,code=sm_90

TARGET = test_norm
CUFLAGS = -O3 -Xcompiler -fopenmp $(ARCH)
LDFLAGS =

SRC = test_norm.cu
OBJ = $(SRC:.cpp=.o)

$(TARGET): $(OBJ)
	nvcc $(CUFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp kernels.cuh
	nvcc $(CUFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) *.o