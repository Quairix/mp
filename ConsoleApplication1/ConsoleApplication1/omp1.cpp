#include <iostream>
#include <fstream>
#include <math.h>
#include <omp.h>
#include <chrono>

void SwapRows(float* a, int rowFrom, int rowTo, int n) {
	float c = 0;
	for (int i = 0; i < n; i++) {
		c = a[rowFrom * n + i];
		a[rowFrom * n + i] = a[rowTo * n + i];
		a[rowTo * n + i] = c;
	}
}

long double determinant_linear(float* a, int n) {
	long double det = 1;

	int pivot_index = -1;
	double pivot_value = 0;
	double determinant = 1;
	for (int i = 0; i < n; ++i) {
		int k = i;
		for (int j = i + 1; j < n; ++j)
			if (abs(a[j * n + i]) > abs(a[k * n + i]))
				k = j;
		if (abs(a[k * n + i]) < 0.001) {
			det = 0;
			break;
		}
		SwapRows(a, i, k, n);
		if (i != k)
			det = -det;
		det *= a[i * n + i];
		for (int j = i + 1; j < n; ++j)
			a[i * n + j] /= a[i * n + i];
		for (int j = 0; j < n; ++j)
			if (j != i && abs(a[j * n + i]) > 0.001)
				for (int k = i + 1; k < n; ++k)
					a[j * n + k] -= a[i * n + k] * a[j * n + i];
	}
	return det;
}

long double determinant_parallel(float* a, int n, int num_threads)
{
	long double det = 1;

	for (int i = 0; i < n; ++i) {
		int k = i;
		for (int j = i + 1; j < n; ++j)
			if (abs(a[j * n + i]) > abs(a[k * n + i]))
				k = j;
		if (abs(a[k * n + i]) < 0.001) {
			det = 0;
			break;
		}
		SwapRows(a, i, k, n);
		if (i != k)
			det = -det;
		det *= a[i * n + i];

#pragma omp parallel num_threads(num_threads)
		{
#pragma omp for schedule(guided)
			for (int j = i + 1; j < n; ++j)
				a[i * n + j] /= a[i * n + i];

#pragma omp for schedule(guided)
			for (int j = 0; j < n; ++j)
				if (j != i && abs(a[j * n + i]) > 0.001)
					for (int k = i + 1; k < n; ++k)
						a[j * n + k] -= a[i * n + k] * a[j * n + i];
		}
	}

	return det;
}

int main(int argc, char* argv[]) {
	if (argc > 2) {
		int n;
		std::ifstream in(argv[1]);
		if (!in) {
			printf_s("File not found\n");
			return 0;
		}

		int num_threads = atoi(argv[2]);
		if(num_threads == 0) num_threads = omp_get_max_threads();
		in >> n;

		float* mat = (float*)malloc(n * n * sizeof(float));
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				in >> mat[i * n + j];
			}
		}
		in.close();
		long double det;
		auto start = std::chrono::system_clock::now();
		det = num_threads == -1 ?
			determinant_linear(mat, n) : determinant_parallel(mat, n, num_threads);
		auto end = std::chrono::system_clock::now();
		double delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

		printf_s("Determinant: %f\n", det);
		printf_s("\nTime (%i thread(s)): %f ms\n", num_threads, delta);

		free(mat);
	}
	else
		printf_s("Использование: \n\tConsoleApplication1.exe <имя_входного_файла> <кол-во_потоков>");
	return 0;
}