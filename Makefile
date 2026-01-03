NVCC = nvcc
NVCC_FLAGS = -std=c++17 -Xcompiler -std=c++17
TARGET = tensor_test
GPT_TARGET = gpt
CUDA_SRC = tensor_kernels.cu
CPP_SRC = tensor.cpp nn.cpp optim.cpp main.cpp
CUDA_OBJ = tensor_kernels.o

all: $(TARGET)

$(GPT_TARGET): $(CUDA_OBJ) $(CPP_SRC)
	$(NVCC) $(NVCC_FLAGS) $(CPP_SRC) $(CUDA_OBJ) -o $(GPT_TARGET)

$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) -c $(CUDA_SRC) -o $(CUDA_OBJ)

$(TARGET): $(CUDA_OBJ) $(CPP_SRC)
	$(NVCC) $(NVCC_FLAGS) $(CPP_SRC) $(CUDA_OBJ) -o $(TARGET)

clean:
	rm -f $(CUDA_OBJ) $(TARGET) $(GPT_TARGET) $(TARGET).exe $(TARGET).lib $(TARGET).exp

.PHONY: all gpt clean
