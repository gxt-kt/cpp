/* #include "service/HRS.h" */

#include "utils.h"
#include "common/heat.h"
#include "omp.h"

static inline void send(const double *data, int nelems, int dst, int tag)
{
	MPI_Request request;
	MPI_Isend(data, nelems, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD, &request);
	/* HRS_MPI_Iwait(&request, MPI_STATUS_IGNORE); */
}


static inline void recv(double *data, int nelems, int src, int tag)
{
	MPI_Request request;
	MPI_Irecv(data, nelems, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &request);
	/* HRS_MPI_Iwait(&request, MPI_STATUS_IGNORE); */
}

static inline void gaussSeidelSolver(int64_t rows, int64_t cols, int rbs, int cbs, int nrb, int ncb, double M[rows][cols], char reps[nrb][ncb])
{
	if (rank != 0) {
		for (int C = 1; C < ncb-1; ++C)
			#pragma omp task depend(in:reps[1][C]) priority(10)
			send(&M[1][(C-1)*cbs+1], cbs, rank-1, C);
		for (int C = 1; C < ncb-1; ++C)
			#pragma omp task depend(out:reps[0][C]) priority(10)
			recv(&M[0][(C-1)*cbs+1], cbs, rank-1, C);
    }

	if (rank != nranks-1) {
		for (int C = 1; C < ncb-1; ++C)
			#pragma omp task depend(out:reps[nrb-1][C]) priority(10)
			recv(&M[rows-1][(C-1)*cbs+1], cbs, rank+1, C);
	}

	for (int R = 1; R < nrb-1; ++R) {
		for (int C = 1; C < ncb-1; ++C) {
			#pragma omp task \
					depend(in:reps[R-1][C], \
						  reps[R+1][C], \
						  reps[R][C-1], \
						  reps[R][C+1]) \
					depend(inout:reps[R][C])			
		computeBlock(rows, cols, (R-1)*rbs+1, R*rbs, (C-1)*cbs+1, C*cbs, M);
		}
	}

	if (rank != nranks-1) {
		for (int C = 1; C < ncb-1; ++C)
			#pragma omp task depend(in:reps[nrb-2][C]) priority(10)
			send(&M[rows-2][(C-1)*cbs+1], cbs, rank+1, C);
	}
}

double solve(HeatConfiguration *conf, int64_t rows, int64_t cols, int timesteps, void *extraData)
{
	double (*matrix)[cols] = (double (*)[cols]) conf->matrix;
	const int rbs = conf->rbs;
	const int cbs = conf->cbs;
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	const int nrb = (rows-2)/rbs+2;
	const int ncb = (cols-2)/cbs+2;
	printf("%d",ncb);
	printf("%d",nrb);
	char representatives[nrb][ncb];
	#pragma omp parallel
	#pragma omp master
	{
	for (int t = 0; t < timesteps; ++t) {
		gaussSeidelSolver(rows, cols, rbs, cbs, nrb, ncb, matrix, representatives);
	}
	}
	MPI_Barrier(MPI_COMM_WORLD);
	return IGNORE_RESIDUAL;
}
