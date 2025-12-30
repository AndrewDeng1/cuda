NVCC = nvcc
NVCC_FLAGS = -std=c++17
TARGET = tensor_test
CUDA_SRC = tensor_kernels.cu
CPP_SRC = tensor.cpp nn.cpp main.cpp
CUDA_OBJ = tensor_kernels.o

all: $(TARGET)

$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) -c $(CUDA_SRC) -o $(CUDA_OBJ)

$(TARGET): $(CUDA_OBJ) $(CPP_SRC)
	$(NVCC) $(NVCC_FLAGS) $(CPP_SRC) $(CUDA_OBJ) -o $(TARGET)

clean:
	-del /Q $(CUDA_OBJ) $(TARGET).exe $(TARGET).lib $(TARGET).exp 2>nul
