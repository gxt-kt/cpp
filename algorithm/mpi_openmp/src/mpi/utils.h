#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include <mpi.h>

#include "common/heat.h"

extern int rank;
extern int nranks;


void broadcastConfiguration(HeatConfiguration *configuration);

#endif // MPI_UTILS_H
