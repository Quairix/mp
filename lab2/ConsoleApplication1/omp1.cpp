#include <iostream>
#include <fstream>
#include <math.h>
#include <omp.h>
#include <chrono>

/**
 *	@param a Массив цветов пикселей
 *	@param n Количество пикселей
 *	@param colors количество цветов
 **/
void brightness_linear(short* a, int n, int colors) {
	short* count = (short*)malloc(colors * sizeof(short));

	for (int i = 0; i < colors; i++)
		count[i] = 0;

	for (int i = 0; i < n; i++)
		count[a[i]]++;

	int max = 0;
	int min = colors;

	int p = n / (colors + 1);

	int start = 0, k = 0;
	while (start < p)
		start += count[k++];
	int startClr = k;
	int end = 0;
	k = colors - 1;
	while (end < p)
		end += count[k--];
	int endClr = k;

	// init min max
	max = startClr;
	min = endClr;

	// find min max
	for (int i = 0; i < n; i++) {
		if (a[i] > startClr && a[i] < endClr) {
			if (max < a[i])
				max = a[i];
			if (min > a[i])
				min = a[i];
		}
	}

	float mn = max - min;

	for (int i = 0; i < n; i++) {
		a[i] = (a[i] - min) * colors / mn;
	}
	free(count);
}

void brightness_parallel(short* a, int n, int colors, int num_threads)
{
	short* count = (short*)malloc(colors * sizeof(short));

	for (int i = 0; i < colors; i++)
		count[i] = 0;

	for (int i = 0; i < n; i++)
		count[a[i]]++;

	int max = 0;
	int min = colors;

	int p = n / (colors + 1);

	// init min max
	for (int i = 0; i < n; i++) {
		if (count[a[i]] > p) {
			max = a[i];
			min = a[i];
			break;
		}
	}

	// find min max
	for (int i = 0; i < n; i++) {
		if (count[a[i]] > p) {
			if (max < a[i])
				max = a[i];
			if (min > a[i])
				min = a[i];
		}
	}
	float mn = max - min;

#pragma omp parallel num_threads(num_threads)
	{
#pragma omp for schedule(static)
		for (int i = 0; i < n; i++) {
			a[i] = (a[i] - min) * colors / mn;
		}
	}
	free(count);
}

int main(int argc, char* argv[]) {
	if (argc > 3) {
		int n;
		int n2;
		int colors;
		std::ifstream in(argv[1], std::ios::binary);
		if (!in) {
			printf_s("File not found\n");
			return 1;
		}

		int num_threads = atoi(argv[3]);
		if (num_threads == 0) num_threads = omp_get_max_threads();
		std::string format;
		std::getline(in, format);

		in >> n;
		in >> n2;
		in >> colors;

		char ma;
		short* mat4 = (short*)malloc(n * n2 * 3 * sizeof(short));
		char* mat = (char*)malloc(n * n2 * 3 * sizeof(char));
		in.read(mat, n * n2 * 3);
		for (int i = 0; i < n * n2 * 3; i++)
			mat4[i] = (short)mat[i] + 128;
		in.close();

		auto start = std::chrono::high_resolution_clock::now();

		num_threads == -1 ?
			brightness_linear(mat4, n * n2 * 3, colors) : brightness_parallel(mat4, n, colors, num_threads);

		auto end = std::chrono::high_resolution_clock::now();
		const auto delta = (end - start) / std::chrono::milliseconds(1);

		printf_s("\nTime (%i thread(s)): %f ms\n", num_threads, delta);

		std::ofstream out(argv[2], std::ios::binary);
		if (!out) {
			printf_s("File not created\n");
			return 1;
		}
		format += "\n";
		out.write(format.c_str(), format.size());

		std::string temp = std::to_string(n) + " " + std::to_string(n2) + "\n" + std::to_string(colors);
		out.write(temp.c_str(), temp.size());

		char* mat2 = (char*)malloc(n * n2 * 3 * sizeof(char));

		for (int i = 0; i < n * n2 * 3; i++) {
			mat2[i] = (char)(mat[i] - 128);
		}

		out.write(mat, n * n2 * 3 * sizeof(char));
		free(mat);
		out.close();
	}
	else
		printf_s("Использование: \n\tConsoleApplication1.exe <имя_входного_файла> <имя_выходного_файла> <кол-во_потоков>");
	return 0;
}