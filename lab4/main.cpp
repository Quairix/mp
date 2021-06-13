#include <CL/cl.h>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <vector>

#define OFFSET 256

typedef unsigned int uint;
typedef struct Platform Platform;

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

void error(const char* msg)
{
	printf("%s\n", msg);
	exit(-1);
}

cl_program getProgram(const char* path, const cl_context context)
{
	cl_int err;
	FILE* sourceFile = fopen(path, "r");

	fseek(sourceFile, 0, SEEK_END);
	size_t sourceFileSize = ftell(sourceFile);

	fseek(sourceFile, 0, 0);
	const char* sourceCode = (char*)malloc(sourceFileSize * sizeof(char));
	fread((char*)sourceCode, sizeof(char), sourceFileSize, sourceFile);

	fclose(sourceFile);

	cl_program program = clCreateProgramWithSource(context,
		1,
		&sourceCode,
		&sourceFileSize,
		&err);
	if (err != 0)
		error("Error: init OpenCL");

	return program;
}

void buildProgram(const cl_program program, const cl_device_id deviceID)
{
	cl_int err = clBuildProgram(program, 1, &deviceID, "", NULL, NULL);

	if (err != 0)
		error("Error: init OpenCL");
}

cl_kernel createKernel(const cl_program program, const char* kernelName)
{
	cl_int err;
	cl_kernel kernel = clCreateKernel(program, kernelName, &err);

	if (err != 0)
		error("Error: init OpenCL");
	return kernel;
}

cl_mem createBuffer(const cl_context context, const size_t size)
{
	cl_int err;
	cl_mem buf = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &err);

	if (err != 0)
		error("Error: init OpenCL");

	return buf;
}

void enqueueWriteBuffer(const cl_command_queue queue, const cl_mem buf, const size_t size, const float* vector)
{
	clEnqueueWriteBuffer(queue, buf, CL_FALSE, 0, size, vector, 0, NULL, NULL);
}

double getTime(cl_event event)
{
	cl_ulong start_time, end_time;
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
	double total_time = (end_time - start_time) * 1e-6;
	return total_time;
}

float* readFile(size_t& N, char* inputPath) {
	std::ifstream input(inputPath);
	if (!input.is_open()) {
		error("Cannot open file");
	}

	input >> N;

	float* x = (float*)malloc(N * sizeof(float*));

	for (size_t i = 0; i < N; ++i)
		input >> x[i];

	input.close();

	return x;
}

void writeFile(size_t N, float* result, char* outputPath) {
	std::ofstream output(outputPath);
	if (!output.is_open()) {
		error("Cannot open file");
	}

	for (size_t i = 0; i < N; i++) {
		output << std::fixed << std::setprecision(1) << result[i] << " ";
	}
	output.close();
}

void inclusive_prefix_sum(const cl_kernel kernel, const cl_command_queue queue, const cl_context context, char* inputPath, char* outputPath)
{
	size_t N;
	float* x = readFile(N, inputPath);

	float* result = (float*)malloc(N * sizeof(float*));


	auto full_start = std::chrono::high_resolution_clock::now();
	const cl_mem bufX = createBuffer(context, N * sizeof(float));
	const cl_mem bufResult = createBuffer(context, N * sizeof(float));

	clSetKernelArg(kernel, 0, sizeof(bufX), &bufX);
	clSetKernelArg(kernel, 1, sizeof(bufResult), &bufResult);
	clSetKernelArg(kernel, 2, sizeof(N), &N);

	// arguments
	enqueueWriteBuffer(queue, bufX, N * sizeof(float), x);
	enqueueWriteBuffer(queue, bufResult, N * sizeof(float), result);

	const int TREADS_COUNT = 2;
	size_t globalTreads[TREADS_COUNT] = { OFFSET, OFFSET };
	cl_event event;
	clEnqueueNDRangeKernel(queue, kernel, TREADS_COUNT, 0, globalTreads, NULL, 0, NULL, &event);

	clEnqueueReadBuffer(queue, bufResult, CL_TRUE, 0, N * sizeof(float), result, 0, NULL, NULL);

	writeFile(N, result, outputPath);

	auto full_end = std::chrono::high_resolution_clock::now();
	const auto full_delta = (full_end - full_start) * 1e-6;
	double processTime = getTime(event);
	printf("\nTime: %f\t%f\n", processTime, full_delta);

	clReleaseMemObject(bufX);
	clReleaseMemObject(bufResult);
}

int main(int argc, char* argv[])
{
	if (argc < 4)
	{
		error("Использование: ocl2.exe <номер_девайса> <имя_входного_файла> <имя_выходного_файла>");
	}
	cl_uint platformIDsCount;

	int total_devices = getDevices();

	if (total_devices == 0) {
		error("No devices found. Check OpenCL installation!\n");
	}

	int device_number = atoi(argv[1]);
	if (device_number < 0 || device_number >= total_devices) {
		device_number = 0;
	}
	cl_platform_id platform = 0;
	clGetPlatformIDs(1, &platform, NULL);
	cl_device_id device = all_devices[device_number];

	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
	cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);
	cl_program inclusive_prefix_sum_KernelProgram = getProgram("kernel.cl", context);
	buildProgram(inclusive_prefix_sum_KernelProgram, device);


	cl_kernel inclusive_prefix_sum_Kernel = createKernel(inclusive_prefix_sum_KernelProgram, "inclusive_prefix_sum");

	inclusive_prefix_sum(inclusive_prefix_sum_Kernel, queue, context, argv[2], argv[3]);

	clReleaseKernel(inclusive_prefix_sum_Kernel);
	clReleaseProgram(inclusive_prefix_sum_KernelProgram);
	clReleaseContext(context);

	return 0;
}
