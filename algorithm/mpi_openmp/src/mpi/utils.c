#include <mpi.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "common/heat.h"

int rank;
int nranks;

void broadcastConfiguration(HeatConfiguration *conf)
{
	MPI_Bcast(conf, sizeof(HeatConfiguration), MPI_BYTE, 0, MPI_COMM_WORLD);

	const int heatSourcesSize = sizeof(HeatSource)*conf->numHeatSources;
	if (rank > 0) {
		// Received heat sources pointer is not valid
		conf->heatSources = (HeatSource *) malloc(heatSourcesSize);
		assert(conf->heatSources != NULL);
	}
	MPI_Bcast(conf->heatSources, heatSourcesSize, MPI_BYTE, 0, MPI_COMM_WORLD);
}

