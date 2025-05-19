#include <omp.h>
#include <chrono>
#include <iostream>
#include <vector>

#ifdef BIG
#define msize 40000
#else
#define msize 20000
#endif

using namespace std;

int main() {

	cout << msize << std::endl;
	
	vector<vector<double> > A(msize);

	for (int i = 0; i < msize; i++) {
		A[i].resize(msize);
		for (int j = 0; j < msize; j++) {
			A[i][j] = i+j;
		}
	}

	vector<double> b(msize);

	for (int i = 0; i < msize; i++)
		b[i] = 5 * i;
	vector<double> c(msize);

	int  threads[] = {1, 2, 4, 7, 8, 16, 20, 40};

    
	for (int i = 0; i < 8; i++) {
		int thread_amount =  threads[i];
		cout << "code started\n";
		auto begin = chrono::steady_clock::now();

		#pragma omp parallel num_threads(thread_amount)
		{
			int threadid = omp_get_thread_num();
			int items_per_thread = msize / thread_amount;
			int left_bound = threadid * items_per_thread;
			int right_bound;
			if (threadid == thread_amount - 1) right_bound = msize - 1;
			else right_bound = left_bound + items_per_thread;
			for (int i = left_bound; i < right_bound; i++) {
				c[i] = 0.0;
				for (int j = 0; j < msize; j++)
					c[i] += A[i][j] * b[j];
			}
		}

		auto end = chrono::steady_clock::now();
		auto elapsed_ms = chrono::duration_cast<std::chrono::milliseconds>(end - begin);
		cout <<  thread_amount << " threads " << elapsed_ms.count()/1000.0 << " seconds\n";
		
		#pragma omp barrier
		{
			cout << "test element " << c.at(10000) << std::endl;
		}
	}
	
	return 0;
}