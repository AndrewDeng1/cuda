NVCC = nvcc
TARGET = tensor_test
CUDA_SRC = tensor_kernels.cu
CPP_SRC = tensor.cpp main.cpp
CUDA_OBJ = tensor_kernels.o

all: $(TARGET)

$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) -c $(CUDA_SRC) -o $(CUDA_OBJ)

$(TARGET): $(CUDA_OBJ) $(CPP_SRC)
	$(NVCC) $(CPP_SRC) $(CUDA_OBJ) -o $(TARGET)

clean:
	-del /Q $(CUDA_OBJ) $(TARGET).exe $(TARGET).lib $(TARGET).exp 2>nul
