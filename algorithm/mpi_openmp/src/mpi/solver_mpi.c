#include <mpi.h>

#include "utils.h"
#include "common/heat.h"


static inline void send(const double *data, int nelems, int dst, int tag)
{
	MPI_Send(data, nelems, MPI_DOUBLE, dst, tag, MPI_COMM_WORLD);
}

static inline void recv(double *data, int nelems, int src, int tag)
{
	MPI_Recv(data, nelems, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

static inline void gaussSeidelSolver(int64_t rows, int64_t cols, int rbs, int cbs, double M[rows][cols])
{
	/* if(rank==0) */
	printf("rank:%d rows:%d, cols:%d, rbs:%d, cbs:%d\n",rank,rows,cols,rbs,cbs);
	if (rank != 0) {
		for (int C = 1; C < cols-1; C += cbs)
			send(&M[1][C], cbs, rank-1, C);
		for (int C = 1; C < cols-1; C += cbs)
			recv(&M[0][C], cbs, rank-1, C);
  }

	if (rank != nranks-1) {
		for (int C = 1; C < cols-1; C += cbs)
			recv(&M[rows-1][C], cbs, rank+1, C);
	}

	for (int R = 1; R < rows-1; R += rbs) {
		for (int C = 1; C < cols-1; C += cbs) {
			computeBlock(rows, cols, R, R+rbs-1, C, C+cbs-1, M);
		}
	}

	if (rank != nranks-1) {
		for (int C = 1; C < cols-1; C += cbs)
			send(&M[rows-2][C], cbs, rank+1, C);
	}
}

double solve(HeatConfiguration *conf, int64_t rows, int64_t cols, int timesteps, void *extraData)
{
	double (*matrix)[cols] = (double (*)[cols]) conf->matrix;

	for (int t = 0; t < timesteps; ++t) {
		gaussSeidelSolver(rows, cols, conf->rbs, conf->cbs, matrix);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	return IGNORE_RESIDUAL;
}
