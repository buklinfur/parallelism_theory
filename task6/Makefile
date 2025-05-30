cpu_sequential:
	pgc++ -o task_cpu -lboost_program_options -acc=host -Minfo=all -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include task_cpu.cpp
	./task_cpu --size=128 --accuracy=0.000001 --max_iterations=1000000

cpu_multicore:
	pgc++ -o task_multicore -lboost_program_options -acc=multicore -Minfo=all -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include task_cpu.cpp
	./task_multicore --size=128 --accuracy=0.000001 --max_iterations=1000000

gpu_optim1:
	pgc++ -o task_gpu_optim1 -lboost_program_options -acc=gpu -Minfo=all -fast -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include task_gpu_optim1.cpp
	./task_gpu_optim1 --size=512 --accuracy=0.000001 --max_iterations=1000000

gpu_optim2:
	pgc++ -o task_gpu_optim2 -lboost_program_options -acc=gpu -Minfo=all -fast -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include task_gpu_optim2.cpp
	./task_gpu_optim2 --size=512 --accuracy=0.000001 --max_iterations=1000000

gpu_optim3:
	pgc++ -o task_gpu_optim3 -lboost_program_options -acc=gpu -Minfo=all -fast -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include task_gpu_optim3.cpp
	./task_gpu_optim3 --size=512 --accuracy=0.000001 --max_iterations=1000000

clear:
	rm task_cpu task_multicore task_gpu_optim1 task_gpu_optim2 task_gpu_optim3

profile:
	nsys profile --trace=nvtx,cuda,openacc --stats=true ./task_gpu_optim3 --size=256 --accuracy=0.0001 --max_iterations=50