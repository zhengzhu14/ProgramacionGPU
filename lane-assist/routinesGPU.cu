#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "routinesGPU.h"

#define BLOCK_SIZE 16

#define SHARED_DIM BLOCK_SIZE + 2



__global__ void noise_red(){
	__shared__ float ims[SHARED_DIM][SHARED_DIM];

	int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
	int col = blockIdx.x*BLOCK_SIZE + threadIdx.y;
	
	


}

void canny(uint8_t *im, uint8_t *image_out,
	float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float level,
	int height, int width)
{


}

void lane_assist_GPU(uint8_t *im, int height, int width,
	int *x1, int *y1, int *x2, int *y2, int *nlines)
{

	/* To do */
}

// void line_asist_GPU(uint8_t *im, int height, int width,
// 	uint8_t *imEdge, float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
// 	float *sin_table, float *cos_table, 
// 	uint32_t *accum, int accu_height, int accu_width,
// 	int *x1, int *x2, int *y1, int *y2, int *nlines)
// {

// 	/* To do */
// }
