#include <omp.h>
#include <iostream>

double integrate_omp(double (*f)(double),double a,double b,int n,int tr) {
	double h = (b - a) / n;
	double sum = 0;
	
	#pragma omp parallel num_threads(tr)
	{
		int thread_id = omp_get_thread_num();
		int items = n / tr;
		int left_bound = thread_id * items;
		int right_bound = (thread_id + 1) * items * (thread_id != tr - 1);
		right_bound += (n - 1) * (thread_id == tr - 1);
		double rectangle;

		for (int i = left_bound; i < right_bound; i++) {
			rectangle = f(a + h * (i + 0.5));
			#pragma omp atomic
			sum += rectangle;
		}
	}

	return sum * h;
}

double local_variable(double (*f)(double), double a, double b, int n, int tr) {
	double h = (b - a) / n;
	double sum = 0;

	#pragma omp parallel num_threads(tr)
	{
		double local_sum = 0;
		int thread_id = omp_get_thread_num();
		int items = n / tr;
		int left_bound = thread_id * items;
		int right_bound = (thread_id + 1) * items * (thread_id != tr - 1);
		right_bound += (n - 1) * (thread_id == tr - 1);

		for (int i = left_bound; i < right_bound; i++) {
			local_sum += f(a + h * (i + 0.5));
		}
		#pragma omp atomic
		sum += local_sum;
	}

	return sum * h;
}

double cubic(double x) {
	return x * x * x;
}

int main() {
	int threads[] = { 1,2,4,7,8,16,20,40 };
	int tr;
	int nsteps = 40'000'000;
	double result;
	double start;
	double time;
	double default_time1,default_time;

	for (int i = 0; i < 8; i++) {
		tr = threads[i];

		start = omp_get_wtime();
		result = integrate_omp(cubic, 0, 10, nsteps, tr);
		time = omp_get_wtime() - start;

		if (i == 0) default_time = time;
		std::cout << "atomic access " << result << " " << tr << " threads " << default_time/time << " seconds\n";

		start = omp_get_wtime();
		result = local_variable(cubic, 0, 10, nsteps, tr);
		time = omp_get_wtime() - start;

		if (i == 0) default_time1 = time;
		std::cout << "local variable " << result << " " << tr << " threads " << default_time1/time << " seconds\n\n";
	}
	return 0;
}