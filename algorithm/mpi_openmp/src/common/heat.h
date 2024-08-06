#ifndef HEAT_H
#define HEAT_H

#include <stdbool.h>
#include <stdint.h>

#define IGNORE_RESIDUAL ((double) -1.0)
#define DEFAULT_DELTA ((double) 0.00005)
#define DEFAULT_BS 1024
#define MAX_STRING_SIZE 100

#define ROUND(a, b) ((((a) + (b) - 1) / (b)) * (b))

typedef struct {
	float row;
	float col;
	float range;
	float temperature;
} HeatSource;

typedef struct {
	int timesteps;
	int convergenceTimesteps;
	double delta;
	int64_t rows;
	int64_t cols;
	int rbs;
	int cbs;
	int nwins;
	double *matrix;
	int numHeatSources;
	HeatSource *heatSources;
	char confFileName[MAX_STRING_SIZE];
	char imageFileName[MAX_STRING_SIZE];
	bool generateImage;
	bool warmup;
	bool verbose;
} HeatConfiguration;


int initialize(HeatConfiguration *conf, int64_t rows, int64_t cols, int64_t rowOffset);
int finalize(HeatConfiguration *conf);
int writeImage(const char *fileName, double *matrix, int64_t rows, int64_t cols);
void readConfiguration(int argc, char **argv, HeatConfiguration *conf);
void refineConfiguration(HeatConfiguration *conf, int64_t rowValue, int64_t colValue);
void printConfiguration(const HeatConfiguration *conf);
void initializeMatrix(const HeatConfiguration *conf, double *matrix, int64_t rows, int64_t cols, int64_t rowOffset);
double getTime();

double solve(HeatConfiguration *conf, int64_t rows, int64_t cols, int timesteps, void *extraData);
void computeBlock(const int64_t rows, const int64_t cols, const int rstart, const int rend, const int cstart, const int cend, double M[rows][cols]);
double computeBlockResidual(const int64_t rows, const int64_t cols, const int rstart, const int rend, const int cstart, const int cend, double M[rows][cols]);

#endif // HEAT_H
