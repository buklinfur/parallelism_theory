TARGET     = gpu_cub
NVCC       = nvcc
CXXFLAGS   = -O3 -std=c++17
BOOST_FLAGS= -lboost_program_options -lcudart

.PHONY: all clean profile

all: $(TARGET)

$(TARGET): gpu_cub.cu
	$(NVCC) gpu_cub.cu -o $(TARGET) $(CXXFLAGS) $(BOOST_FLAGS)

profile: $(TARGET)
	nsys profile --trace=nvtx,cuda --stats=true ./$(TARGET) --max_iterations=50

clean:
	rm -f $(TARGET)