#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "routinesGPU.h"

#define BLOCK_SIZE 16

#define NOISE_RED 2
#define GRADIENT 2
#define EDGE_RAD 3

#define NOISE_SHARED BLOCK_SIZE + 2*NOISE_RED




__global__ void noise_red(uint8_t *im, float* oNR, int height, int width){
	__shared__ float ims[NOISE_SHARED][NOISE_SHARED]; //Tamanyo (BLOCK_SIZE + 4)*(BLOCK_SIZE + 4)

	int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
	int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	
	int thread_id = threadIdx.y*BLOCK_SIZE + threadIdx.x;

	int total_shared = NOISE_SHARED*NOISE_SHARED;

	//Cargo los datos
	for (int i = thread_id; i < total_shared; i+=BLOCK_SIZE*BLOCK_SIZE){
		//De esta manera hago que hilos consecutivos lean posiciones consecutivas.
		//Posiciones dentro de la matriz shared
		int srow = i/NOISE_SHARED;
		int scol = i%NOISE_SHARED;


		//Obtengo la posicion absoluta de la memoria global
		int gx = blockIdx.x*BLOCK_SIZE + scol - NOISE_RED;
		int gy = blockIdx.y*BLOCK_SIZE + srow - NOISE_RED;

		if (gx >= 0 && gy >= 0 && gx < width && gy < height){
			ims[srow][scol] = im[gy*width + gx];
		}

	}
	__syncthreads();

	if(row < height - 2 && row >= 2 && col >= 2 && col < width - 2){
		int sx = threadIdx.x + NOISE_RED;
		int sy = threadIdx.y + NOISE_RED;
		float sum = 0.0;

		sum = (2.0*ims[sy - 2][sx-2] +  4.0*ims[sy - 2][sx-1] +  5.0*ims[sy - 2][sx] +  4.0*ims[sy - 2][sx+1] + 2.0*ims[sy - 2][sx+2]
				+ 4.0*ims[sy - 1][sx-2] +  9.0*ims[sy - 1][sx-1] + 12.0*ims[sy - 1][sx] +  9.0*ims[sy - 1][sx+1] + 4.0*ims[sy - 1][sx+2]
				+ 5.0*ims[sy ][sx-2] + 12.0*ims[sy ][sx-1] + 15.0*ims[sy ][sx] + 12.0*ims[sy ][sx+1] + 5.0*ims[sy ][sx+2]
				+ 4.0*ims[sy + 1][sx-2] +  9.0*ims[sy + 1][sx-1] + 12.0*ims[sy + 1][sx] +  9.0*ims[sy + 1][sx+1] + 4.0*ims[sy + 1][sx+2]
				+ 2.0*ims[sy + 2][sx-2] +  4.0*ims[sy + 2][sx-1] +  5.0*ims[sy + 2][sx] +  4.0*ims[sy + 2][sx+1] + 2.0*ims[sy + 2][sx+2])
				/159.0;

		
		oNR[row*width + col] = sum;
	}

}

__global__ void intensityGradient (){

}

__global__ void getEdges(){

}

__global__ void hysteresis_thresholding(){
	
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
