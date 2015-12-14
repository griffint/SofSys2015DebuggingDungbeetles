all : gpu_fix cpu
gpu_fix: gpu_fix.cu
	nvcc -std=c++11 gpu_fix.cu -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_cudaarithm -o gpu_fix

cpu: cpu.cu
	nvcc -std=c++11 cpu.cu -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_cudaarithm -o cpu
