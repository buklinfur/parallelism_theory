gpu_cublas:
	pgc++ -o gpu_cublas -lboost_program_options -acc=gpu -Minfo=all -fast \
		-I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include \
		-I/usr/local/cuda/include \
		-L/usr/local/cuda/lib64 \
		-I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/include \
		-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/lib \
		-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/lib64 \
		-lcublas -lcudart main_cublas.cpp

clear:
	rm gpu_cublas *.nsys-rep *.sqlite

profile:
	nsys profile --trace=nvtx,cuda,openacc --stats=true ./gpu_cublas --max_iterations=50