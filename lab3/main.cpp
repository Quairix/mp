
#define CL_KERNEL_FILE "kernel.cl"
// Define OpenCL compiler options, such as "-cl-nv-maxrregcount=127"
#define COMPILER_OPTIONS ""
#define CL_INCLUDE_FILE "Settings.h"

#define CL_PTX_FILE "bin/myGEMM.cl.ptx"
// Threadblock sizes (e.g. for kernels myGEMM1 or myGEMM2)
#define TS 32

#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

// Forward declaration of the OpenCL error checking function
void checkError(cl_int error, int line);

std::vector<cl_device_id> all_devices;

int getDevices() {
	std::vector<cl_device_id> discrete_gpu, integrated_gpu, cpu;
	int i, j;

	cl_uint deviceCount;
	cl_device_id* devices;
	cl_platform_id platform = 0;
	cl_device_type value;
	size_t valueSize;
	cl_uint maxComputeUnits;
	clGetPlatformIDs(1, &platform, NULL);
	// get all devices
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
	devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

	// for each device print critical attributes
	for (j = 0; j < deviceCount; j++) {
		clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(cl_device_type), &value, NULL);
		if ((cl_device_type)value == CL_DEVICE_TYPE_CPU)
		{
			cpu.push_back(devices[j]);
		}
		else if ((cl_device_type)value == CL_DEVICE_HOST_UNIFIED_MEMORY)
		{
			integrated_gpu.push_back(devices[j]);
		}
		else {
			discrete_gpu.push_back(devices[j]);
		}
	}
	free(devices);

	all_devices.insert(all_devices.end(), discrete_gpu.begin(), discrete_gpu.end());
	all_devices.insert(all_devices.end(), integrated_gpu.begin(), integrated_gpu.end());
	all_devices.insert(all_devices.end(), cpu.begin(), cpu.end());

	return deviceCount;
}

// Load an OpenCL kernel from file
char* readKernelFile(const char* filename, long* _size) {

	// Open the file
	FILE* file = fopen(filename, "r");
	if (!file) {
		printf("-- Error opening file %s\n", filename);
		exit(1);
	}

	// Get its size
	fseek(file, 0, SEEK_END);
	long size = ftell(file);
	rewind(file);

	// Read the kernel code as a string
	char* source = (char*)malloc((size + 1) * sizeof(char));
	fread(source, 1, size * sizeof(char), file);
	source[size] = '\0';
	fclose(file);

	// Save the size and return the source string
	*_size = (size + 1);
	return source;
}

int main(int argc, char* argv[]) {
	if (argc != 5) {
		std::cerr << "Wrong count of arguments" << std::endl;
		return 1;
	}

	int device_number = atoi(argv[1]);
	int algorithm = atoi(argv[4]);

	std::ifstream input(argv[2]);
	if (!input.is_open()) {
		std::cerr << "Cannot open file" << std::endl;
		return 1;
	}

	size_t n, k, m;
	input >> n >> k >> m;

	float* A = (float*)malloc(m * k * sizeof(float*));
	float* B = (float*)malloc(k * n * sizeof(float*));
	float* C = (float*)malloc(m * n * sizeof(float*));

	for (size_t i = 0; i < m; ++i)
		for (size_t j = 0; j < k; ++j)
			input >> A[i * k + j];
	for (size_t i = 0; i < k; ++i)
		for (size_t j = 0; j < n; ++j)
			input >> B[i * n + j];

	input.close();

	int total_devices = getDevices();

	if (total_devices == 0) {
		std::cerr << "No devices found. Check OpenCL installation!\n";
		return 1;
	}

	if (device_number < 0 || device_number >= total_devices) {
		device_number = 0;
	}

	// Configure the OpenCL environment
	printf(">>> Initializing OpenCL...\n");
	cl_platform_id platform = 0;
	clGetPlatformIDs(1, &platform, NULL);
	cl_device_id device = all_devices[device_number];
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
	cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
	char deviceName[1024];
	clGetDeviceInfo(device, CL_DEVICE_NAME, 1024, deviceName, NULL);
	cl_event event = NULL;

	long sizeSource, sizeHeader;
	cl_int err;

	char* header = readKernelFile(CL_INCLUDE_FILE, &sizeHeader);
	char* source = readKernelFile(CL_KERNEL_FILE, &sizeSource);
	long size = 2 + sizeHeader + sizeSource;
	char* code = (char*)malloc(size * sizeof(char));
	for (int c = 0; c < size; c++) { code[c] = '\0'; }
	strcat(code, header);
	strcat(code, source);
	const char* constCode = code;
	free(header);
	free(source);
	// Compile the kernel
	cl_program program = clCreateProgramWithSource(context, 1, &constCode, NULL, &err);
	checkError(err, __LINE__);
	clBuildProgram(program, 0, NULL, COMPILER_OPTIONS, NULL, NULL);

	// Retrieve the PTX code from the OpenCL compiler and output it to disk
	size_t binSize;
	err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binSize, NULL);
	checkError(err, __LINE__);
	unsigned char* bin = (unsigned char*)malloc(binSize);
	err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &bin, NULL);
	checkError(err, __LINE__);
	FILE* file = fopen(CL_PTX_FILE, "wb");
	fwrite(bin, sizeof(char), binSize, file);
	fclose(file);
	free(bin);

	// Check for compilation errors
	size_t logSize;
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
	char* messages = (char*)malloc((1 + logSize) * sizeof(char));
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, messages, NULL);
	messages[logSize] = '\0';
	if (logSize > 10) { printf(">>> Compiler message: %s\n", messages); }
	free(messages);

	std::ofstream output(argv[3]);
	if (!output.is_open()) {
		std::cerr << "Cannot open file" << std::endl;
		return 1;
	}

	// Prepare OpenCL memory objects
	cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, m * k * sizeof(float), NULL, NULL);
	cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, k * n * sizeof(float), NULL, NULL);
	cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE, m * n * sizeof(float), NULL, NULL);

	// Copy matrices to the GPU
	clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, m * k * sizeof(float), A, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, k * n * sizeof(float), B, 0, NULL, NULL);
	clEnqueueWriteBuffer(queue, bufC, CL_TRUE, 0, m * n * sizeof(float), C, 0, NULL, NULL);

	auto full_start = std::chrono::high_resolution_clock::now();

	// Configure the myGEMM kernel
	//char kernelname[100];
	//sprintf(kernelname, "myGEMM%d", algorithm+1);
	std::string kernelname = "myGEMM" + std::to_string(algorithm + 1);
	cl_kernel kernel = clCreateKernel(program, kernelname.c_str(), &err);
	checkError(err, __LINE__);

	clSetKernelArg(kernel, 0, sizeof(int), (void*)&m);
	clSetKernelArg(kernel, 1, sizeof(int), (void*)&n);
	clSetKernelArg(kernel, 2, sizeof(int), (void*)&k);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&bufA);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&bufB);
	clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&bufC);

	auto start = std::chrono::high_resolution_clock::now();

	// Run the my kernel
	const size_t local[2] = { TS, TS };
	const size_t global[2] = { m, n };
	clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);

	// Wait for calculations to be finished
	clWaitForEvents(1, &event);

	auto end = std::chrono::high_resolution_clock::now();

	const auto delta = (end - start) / std::chrono::microseconds(1);

	// Copy the output matrix C back to the CPU memory
	clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, m * n * sizeof(*C), C, 0, NULL, NULL);

	auto full_end = std::chrono::high_resolution_clock::now();
	const auto full_delta = (full_end - full_start) / std::chrono::microseconds(1);
	printf("\nTime: %d\t%d \n", delta, full_delta);

	// Free the memory objects
	free(code);
	// Free the OpenCL memory objects
	clReleaseMemObject(bufA);
	clReleaseMemObject(bufB);
	clReleaseMemObject(bufC);

	// Clean-up OpenCL 
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	clReleaseProgram(program);
	clReleaseKernel(kernel);

	output << n << " " << m << std::endl;
	for (size_t i = 0; i < m; ++i) {
		for (size_t j = 0; j < n; ++j) {
			printf("\n%f\n", C[i * n + j]);
			output << C[i * n + j] << " ";
		}
		output << std::endl;
	}

	// Free the host memory objects
	free(A);
	free(B);
	free(C);
	return 0;
}

// Print an error message to screen (only if it occurs)
void checkError(cl_int error, int line) {
	if (error != CL_SUCCESS) {
		switch (error) {
		case CL_DEVICE_NOT_FOUND:                 printf("-- Error at %d:  Device not found.\n", line); break;
		case CL_DEVICE_NOT_AVAILABLE:             printf("-- Error at %d:  Device not available\n", line); break;
		case CL_COMPILER_NOT_AVAILABLE:           printf("-- Error at %d:  Compiler not available\n", line); break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:    printf("-- Error at %d:  Memory object allocation failure\n", line); break;
		case CL_OUT_OF_RESOURCES:                 printf("-- Error at %d:  Out of resources\n", line); break;
		case CL_OUT_OF_HOST_MEMORY:               printf("-- Error at %d:  Out of host memory\n", line); break;
		case CL_PROFILING_INFO_NOT_AVAILABLE:     printf("-- Error at %d:  Profiling information not available\n", line); break;
		case CL_MEM_COPY_OVERLAP:                 printf("-- Error at %d:  Memory copy overlap\n", line); break;
		case CL_IMAGE_FORMAT_MISMATCH:            printf("-- Error at %d:  Image format mismatch\n", line); break;
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:       printf("-- Error at %d:  Image format not supported\n", line); break;
		case CL_BUILD_PROGRAM_FAILURE:            printf("-- Error at %d:  Program build failure\n", line); break;
		case CL_MAP_FAILURE:                      printf("-- Error at %d:  Map failure\n", line); break;
		case CL_INVALID_VALUE:                    printf("-- Error at %d:  Invalid value\n", line); break;
		case CL_INVALID_DEVICE_TYPE:              printf("-- Error at %d:  Invalid device type\n", line); break;
		case CL_INVALID_PLATFORM:                 printf("-- Error at %d:  Invalid platform\n", line); break;
		case CL_INVALID_DEVICE:                   printf("-- Error at %d:  Invalid device\n", line); break;
		case CL_INVALID_CONTEXT:                  printf("-- Error at %d:  Invalid context\n", line); break;
		case CL_INVALID_QUEUE_PROPERTIES:         printf("-- Error at %d:  Invalid queue properties\n", line); break;
		case CL_INVALID_COMMAND_QUEUE:            printf("-- Error at %d:  Invalid command queue\n", line); break;
		case CL_INVALID_HOST_PTR:                 printf("-- Error at %d:  Invalid host pointer\n", line); break;
		case CL_INVALID_MEM_OBJECT:               printf("-- Error at %d:  Invalid memory object\n", line); break;
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  printf("-- Error at %d:  Invalid image format descriptor\n", line); break;
		case CL_INVALID_IMAGE_SIZE:               printf("-- Error at %d:  Invalid image size\n", line); break;
		case CL_INVALID_SAMPLER:                  printf("-- Error at %d:  Invalid sampler\n", line); break;
		case CL_INVALID_BINARY:                   printf("-- Error at %d:  Invalid binary\n", line); break;
		case CL_INVALID_BUILD_OPTIONS:            printf("-- Error at %d:  Invalid build options\n", line); break;
		case CL_INVALID_PROGRAM:                  printf("-- Error at %d:  Invalid program\n", line); break;
		case CL_INVALID_PROGRAM_EXECUTABLE:       printf("-- Error at %d:  Invalid program executable\n", line); break;
		case CL_INVALID_KERNEL_NAME:              printf("-- Error at %d:  Invalid kernel name\n", line); break;
		case CL_INVALID_KERNEL_DEFINITION:        printf("-- Error at %d:  Invalid kernel definition\n", line); break;
		case CL_INVALID_KERNEL:                   printf("-- Error at %d:  Invalid kernel\n", line); break;
		case CL_INVALID_ARG_INDEX:                printf("-- Error at %d:  Invalid argument index\n", line); break;
		case CL_INVALID_ARG_VALUE:                printf("-- Error at %d:  Invalid argument value\n", line); break;
		case CL_INVALID_ARG_SIZE:                 printf("-- Error at %d:  Invalid argument size\n", line); break;
		case CL_INVALID_KERNEL_ARGS:              printf("-- Error at %d:  Invalid kernel arguments\n", line); break;
		case CL_INVALID_WORK_DIMENSION:           printf("-- Error at %d:  Invalid work dimensionsension\n", line); break;
		case CL_INVALID_WORK_GROUP_SIZE:          printf("-- Error at %d:  Invalid work group size\n", line); break;
		case CL_INVALID_WORK_ITEM_SIZE:           printf("-- Error at %d:  Invalid work item size\n", line); break;
		case CL_INVALID_GLOBAL_OFFSET:            printf("-- Error at %d:  Invalid global offset\n", line); break;
		case CL_INVALID_EVENT_WAIT_LIST:          printf("-- Error at %d:  Invalid event wait list\n", line); break;
		case CL_INVALID_EVENT:                    printf("-- Error at %d:  Invalid event\n", line); break;
		case CL_INVALID_OPERATION:                printf("-- Error at %d:  Invalid operation\n", line); break;
		case CL_INVALID_GL_OBJECT:                printf("-- Error at %d:  Invalid OpenGL object\n", line); break;
		case CL_INVALID_BUFFER_SIZE:              printf("-- Error at %d:  Invalid buffer size\n", line); break;
		case CL_INVALID_MIP_LEVEL:                printf("-- Error at %d:  Invalid mip-map level\n", line); break;
		case -1024:                               printf("-- Error at %d:  *clBLAS* Functionality is not implemented\n", line); break;
		case -1023:                               printf("-- Error at %d:  *clBLAS* Library is not initialized yet\n", line); break;
		case -1022:                               printf("-- Error at %d:  *clBLAS* Matrix A is not a valid memory object\n", line); break;
		case -1021:                               printf("-- Error at %d:  *clBLAS* Matrix B is not a valid memory object\n", line); break;
		case -1020:                               printf("-- Error at %d:  *clBLAS* Matrix C is not a valid memory object\n", line); break;
		case -1019:                               printf("-- Error at %d:  *clBLAS* Vector X is not a valid memory object\n", line); break;
		case -1018:                               printf("-- Error at %d:  *clBLAS* Vector Y is not a valid memory object\n", line); break;
		case -1017:                               printf("-- Error at %d:  *clBLAS* An input dimension (M,N,K) is invalid\n", line); break;
		case -1016:                               printf("-- Error at %d:  *clBLAS* Leading dimension A must not be less than the size of the first dimension\n", line); break;
		case -1015:                               printf("-- Error at %d:  *clBLAS* Leading dimension B must not be less than the size of the second dimension\n", line); break;
		case -1014:                               printf("-- Error at %d:  *clBLAS* Leading dimension C must not be less than the size of the third dimension\n", line); break;
		case -1013:                               printf("-- Error at %d:  *clBLAS* The increment for a vector X must not be 0\n", line); break;
		case -1012:                               printf("-- Error at %d:  *clBLAS* The increment for a vector Y must not be 0\n", line); break;
		case -1011:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix A is too small\n", line); break;
		case -1010:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix B is too small\n", line); break;
		case -1009:                               printf("-- Error at %d:  *clBLAS* The memory object for Matrix C is too small\n", line); break;
		case -1008:                               printf("-- Error at %d:  *clBLAS* The memory object for Vector X is too small\n", line); break;
		case -1007:                               printf("-- Error at %d:  *clBLAS* The memory object for Vector Y is too small\n", line); break;
		case -1001:                               printf("-- Error at %d:  Code -1001: no GPU available?\n", line); break;
		default:                                  printf("-- Error at %d:  Unknown with code %d\n", line, error);
		}
		exit(1);
	}
}