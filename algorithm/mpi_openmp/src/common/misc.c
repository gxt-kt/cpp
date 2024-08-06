#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <float.h>
#include <getopt.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "common/heat.h"


// 初始化这一块内存
int initialize(HeatConfiguration *conf, int64_t rows, int64_t cols, int64_t rowOffset)
{
	int pagesize = sysconf(_SC_PAGESIZE);
	assert(pagesize > 0);

	// 内存对齐的初始化，一般是4KB对齐，对齐到内存页的大小
	int err = posix_memalign((void **)&conf->matrix, pagesize, rows*cols*sizeof(double));
	if (err || conf->matrix == NULL) {
		fprintf(stderr, "Error: Memory cannot be allocated!\n");
		exit(1);
	}

	initializeMatrix(conf, conf->matrix, rows, cols, rowOffset);

	return 0;
}

int finalize(HeatConfiguration *conf)
{
	assert(conf->matrix != NULL);
	free(conf->matrix);
	conf->matrix = NULL;

	return 0;
}

int writeImage(const char *imageFileName, double *matrix, int64_t rows, int64_t cols)
{
	// RGB table
	unsigned int red[1024], green[1024], blue[1024];

	// Prepare the RGB table
	int n = 1023;
	for (int i = 0; i < 256; i++) {
		red[n] = 255; green[n] = i; blue[n] = 0;
		n--;
	}

	for (int i = 0; i < 256; i++) {
		red[n] = 255-i; green[n] = 255; blue[n] = 0;
		n--;
	}

	for (int i = 0; i < 256; i++) {
		red[n] = 0; green[n] = 255; blue[n] = i;
		n--;
	}

	for (int i = 0; i < 256; i++) {
		red[n] = 0; green[n] = 255-i; blue[n] = 255;
		n--;
	}

	// Find minimum and maximum
	double min = DBL_MAX;
	double max = -DBL_MAX;
	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; ++c) {
			if (matrix[r*cols+c] > max)
				max = matrix[r*cols+c];
			if (matrix[r*cols+c] < min)
				min = matrix[r*cols+c];
		}
	}

	FILE *file = fopen(imageFileName, "w");
	if (file == NULL) {
		fprintf(stderr, "Error: Unable to create image file %s\n", imageFileName);
		exit(1);
	}

	fprintf(file, "P3\n");
	fprintf(file, "%ld %ld\n", cols, rows);
	fprintf(file, "255\n");

	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; ++c) {
			int k = 0;
			if (max-min != 0) {
				k = (int)(1023.0*(matrix[r*cols+c]-min)/(max-min));
			}
			fprintf(file, "%d %d %d  ", red[k], green[k], blue[k]);
			if (c == cols-1) fprintf(file, "\n");
		}
	}

	fclose(file);

	return 0;
}

static void printUsage(int argc, char **argv)
{
	fprintf(stdout, "Usage: %s <-s size> | <-r rows -c cols> <-t timesteps> [OPTION]...\n", argv[0]);
	fprintf(stdout, "Parameters:\n");
	fprintf(stdout, "  -s, --size=SIZE          use SIZExSIZE matrix as the surface\n");
	fprintf(stdout, "  -r, --rows=ROWS          use ROWS as the number of rows of the surface\n");
	fprintf(stdout, "  -c, --cols=COLS          use COLS as the number of columns of the surface\n");
	fprintf(stdout, "  -t, --timesteps=TS       use TS as the number of timesteps\n");
	fprintf(stdout, "Optional parameters:\n");
	fprintf(stdout, "  -b, --bs=BS              use BS as the number of rows and columns of each block (default: %d)\n", DEFAULT_BS);
	fprintf(stdout, "  -R, --rbs=BS             use BS as the number of rows of each block (overrides -b option)\n");
	fprintf(stdout, "  -C, --cbs=BS             use BS as the number of columns of each block (overrides -b option)\n");
	fprintf(stdout, "  -w, --wins=WINS          use WINS as the number of MPI RMA windows for each halo row\n");
	fprintf(stdout, "  -d, --delta=DELTA        use DELTA as the residual threshold (default: %f)\n", DEFAULT_DELTA);
	fprintf(stdout, "  -f, --sources-file=NAME  get the heat sources from the NAME configuration file (default: heat.conf)\n");
	fprintf(stdout, "  -W, --no-warmup          do not perform warmup timestep (warmup enabled by default)\n");
	fprintf(stdout, "  -o, --output=NAME        save the computed matrix to the PPM file named NAME.ppm and disable warmup (disabled by default)\n");
	fprintf(stdout, "  -v, --verbose            display additional information (disabled by default)\n");
	fprintf(stdout, "  -h, --help               display this help and exit\n\n");
}

static void setDefaultConfiguration(HeatConfiguration *conf)
{
	conf->timesteps = 0;
	conf->convergenceTimesteps = -1;
	conf->delta = DEFAULT_DELTA;
	conf->rows = 0;
	conf->cols = 0;
	conf->rbs = DEFAULT_BS;
	conf->cbs = DEFAULT_BS;
	conf->nwins = 0;
	conf->matrix = NULL;
	conf->numHeatSources = 0;
	conf->heatSources = NULL;
	strcpy(conf->confFileName, "heat.conf");
	strcpy(conf->imageFileName, "heat.ppm");
	conf->generateImage = false;
	conf->warmup = true;
	conf->verbose = false;
}

static void readParameters(int argc, char **argv, HeatConfiguration *conf)
{
	static struct option long_options[] = {
		{"size",         required_argument,  0, 's'},
		{"rows",         required_argument,  0, 'r'},
		{"cols",         required_argument,  0, 'c'},
		{"timesteps",    required_argument,  0, 't'},
		{"bs",           required_argument,  0, 'b'},
		{"rbs",          required_argument,  0, 'R'},
		{"cbs",          required_argument,  0, 'C'},
		{"wins",         required_argument,  0, 'w'},
		{"delta",        required_argument,  0, 'd'},
		{"sources-file", required_argument,  0, 'f'},
		{"output",       required_argument,  0, 'o'},
		{"no-warmup",    no_argument,        0, 'W'},
		{"verbose",      no_argument,        0, 'v'},
		{"help",         no_argument,        0, 'h'},
		{0, 0, 0, 0}
	};

	int c, index;
	int bs = DEFAULT_BS;
	int rbs = 0, cbs = 0;

	while ((c = getopt_long(argc, argv, "ho:f:s:r:c:t:b:R:C:d:w:Wv", long_options, &index)) != -1) {
		switch (c) {
			case 'h':
				printUsage(argc, argv);
				exit(0);
			case 'v':
				conf->verbose = true;
				break;
			case 'f':
				if (strlen(optarg) >= MAX_STRING_SIZE) {
					fprintf(stderr, "Error: Configuration name is too long!\n");
					exit(1);
				}
				strcpy(conf->confFileName, optarg);
				break;
			case 'o':
				conf->generateImage = true;
				conf->warmup = false;
				if (strlen(optarg) >= MAX_STRING_SIZE) {
					fprintf(stderr, "Error: Image name is too long!\n");
					exit(1);
				}
				strcpy(conf->imageFileName, optarg);
				break;
			case 'W':
				conf->warmup = false;
				break;
			case 's':
				conf->rows = atoi(optarg);
				conf->cols = atoi(optarg);
				assert(conf->rows > 0 && conf->cols > 0);
				break;
			case 'r':
				conf->rows = atoi(optarg);
				assert(conf->rows > 0);
				break;
			case 'c':
				conf->cols = atoi(optarg);
				assert(conf->cols > 0);
				break;
			case 't':
				conf->timesteps = atoi(optarg);
				assert(conf->timesteps > 0);
				break;
			case 'b':
				bs = atoi(optarg);
				assert(bs > 0);
				break;
			case 'R':
				rbs = atoi(optarg);
				assert(rbs > 0);
				break;
			case 'C':
				cbs = atoi(optarg);
				assert(cbs > 0);
				break;
			case 'w':
				conf->nwins = atoi(optarg);
				assert(conf->nwins > 0);
				break;
			case 'd':
				conf->delta = atof(optarg);
				assert(conf->delta > 0.0);
				break;
			case '?':
				exit(1);
			default:
				abort();
		}
	}

	conf->rbs = (rbs == 0) ? bs : rbs;
	conf->cbs = (cbs == 0) ? bs : cbs;

	if (!conf->rows || !conf->cols || !conf->timesteps || !conf->rbs || !conf->cbs) {
		printUsage(argc, argv);
		exit(1);
	}

#ifdef MPIRMA
	if (conf->nwins <= 0) {
		fprintf(stdout, "Error: Must set a valid number of MPI windows\n");
		exit(1);
	}
#endif
}

void readConfiguration(int argc, char **argv, HeatConfiguration *conf)
{
	char line[MAX_STRING_SIZE];

	setDefaultConfiguration(conf);

	// Read the execution parameters
	readParameters(argc, argv, conf);

	FILE *file = fopen(conf->confFileName, "r");
	if (file == NULL) {
		fprintf(stderr, "Error: Unable to open configuration file %s!\n", conf->confFileName);
		exit(1);
	}

	if (!fgets(line, MAX_STRING_SIZE, file)) {
		fprintf(stderr, "Error: Configuration file is not correct!\n");
		exit(1);
	}

	int n = sscanf(line, "%d", &(conf->numHeatSources));
	if (n != 1) {
		fprintf(stderr, "Error: Configuration file not correct!\n");
		exit(1);
	}
	if (conf->numHeatSources < 1) {
		fprintf(stderr, "Error: Configuration file must have at least one heat source!\n");
		exit(1);
	}

	conf->heatSources = (HeatSource *) malloc(sizeof(HeatSource)*conf->numHeatSources);
	assert(conf->heatSources != NULL);

	for (int i = 0; i < conf->numHeatSources; i++) {
		if (!fgets(line, MAX_STRING_SIZE, file)) {
			fprintf(stderr, "Error: Configuration file is not correct!\n");
			exit(1);
		}

		n = sscanf(line, "%f %f %f %f",
			&(conf->heatSources[i].row),
			&(conf->heatSources[i].col),
			&(conf->heatSources[i].range),
			&(conf->heatSources[i].temperature));

		if (n != 4) {
			fprintf(stderr, "Error: Configuration file not correct!\n");
			exit(1);
		}
	}

	fclose(file);
}

void refineConfiguration(HeatConfiguration *conf, int64_t rowValue, int64_t colValue)
{
	assert(conf->rows > 0);
	assert(conf->cols > 0);
	assert(conf->timesteps > 0);

	if (conf->rows%rowValue) {
		fprintf(stderr, "Warning: The number of rows (%ld) is not divisible by %ld. Rounding it...\n", conf->rows, rowValue);
		// Make the number of rows divisible by the value
		conf->rows = ROUND(conf->rows, rowValue);
	}
	if (conf->cols%colValue) {
		fprintf(stderr, "Warning: The number of cols (%ld) is not divisible by %ld. Rounding it...\n", conf->cols, colValue);
		// Make the number of cols divisible by the value
		conf->cols = ROUND(conf->cols, colValue);
	}
}

void printConfiguration(const HeatConfiguration *conf)
{
	fprintf(stdout, "Rows x Cols       : %ld x %ld\n", conf->rows, conf->cols);
	fprintf(stdout, "Block size        : %d x %d\n", conf->rbs, conf->cbs);
	fprintf(stdout, "Timesteps         : %d\n", conf->timesteps);
	fprintf(stdout, "Delta             : %f\n", conf->delta);
	fprintf(stdout, "Num. heat sources : %d\n", conf->numHeatSources);

	for (int i = 0; i < conf->numHeatSources; i++) {
		fprintf(stdout, "  %2d: (%2.2f, %2.2f) %2.2f %2.2f\n", i+1,
			conf->heatSources[i].row,
			conf->heatSources[i].col,
			conf->heatSources[i].range,
			conf->heatSources[i].temperature
		);
	}
}

void initializeMatrix(const HeatConfiguration *conf, double *matrix, int64_t rows, int64_t cols, int64_t rowOffset)
{
	const int totalRows = conf->rows+2;

	// Set all elements to zero
	memset(matrix, 0, rows*cols*sizeof(double));

	for (int i = 0; i < conf->numHeatSources; i++) {
		const HeatSource *src = &(conf->heatSources[i]);

		// Initialize top row
		if (rowOffset == 0) {
			for (int c = 0; c < cols; ++c) {
				double dist = sqrt(pow((double)c/(double)cols-src->col, 2) + pow(src->row, 2));
				if (dist <= src->range) {
					matrix[c] += (src->range-dist)/src->range*src->temperature;
				}
			}
		}

		// Initialize bottom row
		if (rowOffset+rows == totalRows) {
			for (int c = 0; c < cols; ++c) {
				double dist = sqrt(pow((double)c/(double)cols-src->col, 2) + pow(1-src->row, 2));
				if (dist <= src->range) {
					matrix[(rows-1)*cols+c] += (src->range-dist)/src->range*src->temperature;
				}
			}
		}

		// Initialize left column
		for (int r = 1; r < rows-1; ++r) {
			double dist = sqrt(pow(src->col, 2) + pow((double)(rowOffset+r)/(double)totalRows-src->row, 2));
			if (dist <= src->range) {
				matrix[r*cols] += (src->range-dist)/src->range*src->temperature;
			}
		}

		// Initialize right column
		for (int r = 1; r < rows-1; ++r) {
			double dist = sqrt(pow(1-src->col, 2) + pow((double)(rowOffset+r)/(double)totalRows-src->row, 2));
			if (dist <= src->range) {
				matrix[r*cols+cols-1] += (src->range-dist)/src->range*src->temperature;
			}
		}
	}
}

double getTime()
{
	struct timespec tv;
	clock_gettime(CLOCK_MONOTONIC, &tv);
	return tv.tv_sec+1e-9*tv.tv_nsec;
}
