#include <stdint.h>

void computeBlock(const int64_t rows, const int64_t cols,
		const int rstart, const int rend,
		const int cstart, const int cend,
		double M[rows][cols])
{
	for (int r = rstart; r <= rend; ++r) {
		for (int c = cstart; c <= cend; ++c) {
			M[r][c] = 0.25*(M[r-1][c] + M[r+1][c] + M[r][c-1] + M[r][c+1]);
		}
	}
}

double computeBlockResidual(const int64_t rows, const int64_t cols,
		const int rstart, const int rend,
		const int cstart, const int cend,
		double M[rows][cols])
{
	double sum = 0.0;
	for (int r = rstart; r <= rend; ++r) {
		for (int c = cstart; c <= cend; ++c) {
			const double value = 0.25*(M[r-1][c] + M[r+1][c] + M[r][c-1] + M[r][c+1]);
			const double diff = value - M[r][c];
			sum += diff*diff;
			M[r][c] = value;
		}
	}
	return sum;
}
