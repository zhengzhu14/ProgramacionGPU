#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "routinesGPU.h"

#define BLOCK_SIZE 16

#define NOISE_RED 2
#define GRADIENT 2
#define EDGE_RAD 3
#define HYSTHRES_RAD 3

#define NOISE_SHARED (BLOCK_SIZE + 2*NOISE_RED)
#define GRADIENT_SHARED (BLOCK_SIZE + 2*GRADIENT)
#define EDGE_SHARED (BLOCK_SIZE + 2)
#define HYSTHRES_SHARED (BLOCK_SIZE + 2)


#define PI 3.141593

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

		//Las posiciones que se salen de rango, de esta manera simplemente
		//copio los valores de los bordes para extenderlo
		gx = max(0, min(gx, width - 1));
		gy = max(0, min(gy, height - 1));

		ims[srow][scol] = im[gy*width + gx];

		/* if (gx >= 0 && gy >= 0 && gx < width && gy < height){
			ims[srow][scol] = im[gy*width + gx];
		} */

	}
	__syncthreads();

	if(row >= NOISE_RED && row < height - NOISE_RED && col >= NOISE_RED && col < width - NOISE_RED){
		int sx = threadIdx.x + NOISE_RED;
		int sy = threadIdx.y + NOISE_RED;
		float sum = 0.0;

		sum = (2.0f*ims[sy - 2][sx-2] +  4.0f*ims[sy - 2][sx-1] +  5.0f*ims[sy - 2][sx] +  4.0f*ims[sy - 2][sx+1] + 2.0f*ims[sy - 2][sx+2]
				+ 4.0f*ims[sy - 1][sx-2] +  9.0f*ims[sy - 1][sx-1] + 12.0f*ims[sy - 1][sx] +  9.0f*ims[sy - 1][sx+1] + 4.0f*ims[sy - 1][sx+2]
				+ 5.0f*ims[sy ][sx-2] + 12.0f*ims[sy ][sx-1] + 15.0f*ims[sy ][sx] + 12.0f*ims[sy ][sx+1] + 5.0f*ims[sy ][sx+2]
				+ 4.0f*ims[sy + 1][sx-2] +  9.0f*ims[sy + 1][sx-1] + 12.0f*ims[sy + 1][sx] +  9.0f*ims[sy + 1][sx+1] + 4.0f*ims[sy + 1][sx+2]
				+ 2.0f*ims[sy + 2][sx-2] +  4.0f*ims[sy + 2][sx-1] +  5.0f*ims[sy + 2][sx] +  4.0f*ims[sy + 2][sx+1] + 2.0f*ims[sy + 2][sx+2])
				/159.0f;

		
		oNR[row*width + col] = sum;
	}

}

__global__ void intensityGradient (float* iNR, float *ophi, float *oG, int height, int width){
	__shared__ float nrs[GRADIENT_SHARED][GRADIENT_SHARED]; //Tamanyo (BLOCK_SIZE + 4)*(BLOCK_SIZE + 4)

	float Gx, Gy;
	float phi;

	int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
	int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;
	
	int thread_id = threadIdx.y*BLOCK_SIZE + threadIdx.x;

	int total_shared = GRADIENT_SHARED*GRADIENT_SHARED;


	for (int i = thread_id; i < total_shared; i+=BLOCK_SIZE*BLOCK_SIZE){
		//De esta manera hago que hilos consecutivos lean posiciones consecutivas.
		//Posiciones dentro de la matriz shared
		int srow = i/GRADIENT_SHARED;
		int scol = i%GRADIENT_SHARED;


		//Obtengo la posicion absoluta de la memoria global
		int gx = blockIdx.x*BLOCK_SIZE + scol - GRADIENT;
		int gy = blockIdx.y*BLOCK_SIZE + srow - GRADIENT;

		//Las posiciones que se salen de rango, de esta manera simplemente
		//copio los valores de los bordes para extenderlo
		gx = max(0, min(gx, width - 1));
		gy = max(0, min(gy, height - 1));

		nrs[srow][scol] = iNR[gy*width + gx];

	}
	__syncthreads();

	if(row >= GRADIENT && row < height - GRADIENT && col >= GRADIENT && col < width - GRADIENT){
		int sx = threadIdx.x + GRADIENT;
		int sy = threadIdx.y + GRADIENT;

		Gx = 
			(1.0f*nrs[sy-2][sx-2] +  2.0f*nrs[sy-2][sx-1] +  (-2.0f)*nrs[sy-2][sx+1] + (-1.0f)*nrs[sy-2][sx+2]
			+ 4.0f*nrs[sy-1][sx-2] +  8.0f*nrs[sy-1][sx-1] +  (-8.0f)*nrs[sy-1][sx+1] + (-4.0f)*nrs[sy-1][sx+2]
			+ 6.0f*nrs[sy][sx-2] + 12.0f*nrs[sy][sx-1] + (-12.0f)*nrs[sy][sx+1] + (-6.0f)*nrs[sy][sx+2]
			+ 4.0f*nrs[sy+1][sx-2] +  8.0f*nrs[sy+1][sx-1] +  (-8.0f)*nrs[sy+1][sx+1] + (-4.0f)*nrs[sy+1][sx+2]
			+ 1.0f*nrs[sy+2][sx-2] +  2.0f*nrs[sy+2][sx-1] +  (-2.0f)*nrs[sy+2][sx+1] + (-1.0f)*nrs[sy+2][sx+2]);

		Gy = 
			((-1.0f)*nrs[sy-2][sx-2] + (-4.0f)*nrs[sy-2][sx-1] +  (-6.0f)*nrs[sy-2][sx] + (-4.0f)*nrs[sy-2][sx+1] + (-1.0f)*nrs[sy-2][sx+2]
			+ (-2.0f)*nrs[sy-1][sx-2] + (-8.0f)*nrs[sy-1][sx-1] + (-12.0f)*nrs[sy-1][sx] + (-8.0f)*nrs[sy-1][sx+1] + (-2.0f)*nrs[sy-1][sx+2]
			+    2.0f*nrs[sy+1][sx-2] +    8.0f*nrs[sy+1][sx-1] +    12.0f*nrs[sy+1][sx] +    8.0f*nrs[sy+1][sx+1] +    2.0f*nrs[sy+1][sx+2]
			+    1.0f*nrs[sy+2][sx-2] +    4.0f*nrs[sy+2][sx-1] +     6.0f*nrs[sy+2][sx] +    4.0f*nrs[sy+2][sx+1] +    1.0f*nrs[sy+2][sx+2]);


		oG[row*width + col] = sqrtf((Gx*Gx)+(Gy*Gy));
		phi = atan2f(fabs(Gy),fabs(Gx));
		float val = fabs(phi);

		//Como los valores de angulos pueden ser
		/*
		Si phi <= PI/8, en result no se suma nada
		Si phi > PI/8, (val > PI/8)*45.0 = 45 por lo que se suma 45
		Idem para cada caso.
		*/
		float result = (val > PI/8)*45.0
					 + (val > 3*(PI/8))*45.0
					 + (val > 5*(PI/8))*45.0;

		//Para el ultimo caso si sube de 7*(PI/8) result vale 0.
		result*= (val <= 7*(PI/8));

		ophi[row*width + col] = result;
	}

}

__global__ void getEdges(float* iG, float* iphi, uint8_t* opedge, int height, int width){
	__shared__ float igs[EDGE_SHARED][EDGE_SHARED];
	//__shared__ float iphis[EDGE_SHARED][EDGE_SHARED];

	int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
	int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int thread_id = threadIdx.y*BLOCK_SIZE + threadIdx.x;
	int total_shared = EDGE_SHARED*EDGE_SHARED;

	//Cargo los datos
	for (int i = thread_id; i < total_shared; i+=BLOCK_SIZE*BLOCK_SIZE){
		//De esta manera hago que hilos consecutivos lean posiciones consecutivas.
		//Posiciones dentro de la matriz shared
		int srow = i/EDGE_SHARED;
		int scol = i%EDGE_SHARED;


		//Obtengo la posicion absoluta de la memoria global
		int gx = blockIdx.x*BLOCK_SIZE + scol - 1;
		int gy = blockIdx.y*BLOCK_SIZE + srow - 1;

		//Las posiciones que se salen de rango, de esta manera simplemente
		//copio los valores de los bordes para extenderlo
		gx = max(0, min(gx, width - 1));
		gy = max(0, min(gy, height - 1));

		igs[srow][scol] = iG[gy*width + gx];
		//iphis[srow][scol] = iphi[gy*width + gx];
	}
	__syncthreads();
	
	if(row >= EDGE_RAD && row < height - EDGE_RAD && col >= EDGE_RAD && col < width - EDGE_RAD){
		int sx = threadIdx.x + 1;
		int sy = threadIdx.y + 1;

		float gij = igs[sy][sx];
		//float phi_ij = iphis[sy][sx];
		float phi_ij = iphi[row*width + col];
		float pedge = 0;

		pedge = (phi_ij == 0 && gij > igs[sy][sx + 1] && gij > igs[sy][sx - 1])
				|| (phi_ij == 45 && gij > igs[sy + 1][sx + 1] && gij > igs[sy - 1][sx - 1])
				|| (phi_ij == 90 && gij > igs[sy + 1][sx] && gij > igs[sy - 1][sx])
				|| (phi_ij == 135 && gij > igs[sy + 1][sx - 1] && gij > igs[sy - 1][sx + 1]);

		opedge[row*width + col] = pedge;
	}

}

__global__ void hysteresis_thresholding(float* iG, uint8_t* ipedge, uint8_t* oimage_out, int height, int width, 
	float lowthres, float hithres){
	__shared__ float igs[HYSTHRES_SHARED][HYSTHRES_SHARED];

	int row = blockIdx.y*BLOCK_SIZE + threadIdx.y;
	int col = blockIdx.x*BLOCK_SIZE + threadIdx.x;

	int thread_id = threadIdx.y*BLOCK_SIZE + threadIdx.x;
	int total_shared = HYSTHRES_SHARED*HYSTHRES_SHARED;

	for (int i = thread_id; i < total_shared; i+=BLOCK_SIZE*BLOCK_SIZE){
		//De esta manera hago que hilos consecutivos lean posiciones consecutivas.
		//Posiciones dentro de la matriz shared
		int srow = i/HYSTHRES_SHARED;
		int scol = i%HYSTHRES_SHARED;
		//Obtengo la posicion absoluta de la memoria global
		int gx = blockIdx.x*BLOCK_SIZE + scol - 1;
		int gy = blockIdx.y*BLOCK_SIZE + srow - 1;

		//Las posiciones que se salen de rango, de esta manera simplemente
		//copio los valores de los bordes para extenderlo
		gx = max(0, min(gx, width - 1));
		gy = max(0, min(gy, height - 1));

		igs[srow][scol] = iG[gy*width + gx];
	}
	__syncthreads();

	if (row >= HYSTHRES_RAD && row < height - HYSTHRES_RAD && col >= HYSTHRES_RAD && col < width - HYSTHRES_RAD){
		int sx = threadIdx.x + 1;
		int sy = threadIdx.y + 1;

		
		uint8_t imo = 0;
		float g = igs[sy][sx];
		float pedge = ipedge[row*width + col];

		imo = 255*(g > hithres && pedge);

		imo+= 255*(!(g > hithres && pedge) && 
				(pedge && g >=lowthres && g<hithres) &&
				((igs[sy-1][sx-1] > hithres) || (igs[sy-1][sx]>hithres) || (igs[sy-1][sx+1]>hithres)
				|| (igs[sy][sx-1]> hithres) || (igs[sy][sx]> hithres) || (igs[sy][sx+1]> hithres)
				|| (igs[sy + 1][sx-1]> hithres) || (igs[sy+1][sx]> hithres) || (igs[sy+1][sx+1]> hithres)));

		oimage_out[row*width + col] = imo;

	}

}

void canny(uint8_t *im, uint8_t *image_out,float level,
	int height, int width, float* G, uint8_t* pedge, float* NR, float *phi )
{	

	int i, j;
	int ii, jj;
	//float PI = 3.141593;

	float lowthres, hithres;

	//Reduccion de ruido
	//Matriz de salida
	float* oNR;
	cudaMalloc(&oNR, sizeof(float)* width * height);


	uint8_t* imd;
	cudaMalloc(&imd, sizeof(uint8_t) * width * height);
	cudaMemcpy(imd, im, sizeof(uint8_t) * width * height, cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(
    	(width  + BLOCK_SIZE - 1) / BLOCK_SIZE,
    	(height + BLOCK_SIZE - 1) / BLOCK_SIZE
	);

	noise_red<<<dimGrid, dimBlock>>>(imd, oNR, height, width);

	cudaDeviceSynchronize(); //Sincronizo


	//Intensidad de Gradiente
	float *ophi;
	float *oG;
	cudaMalloc(&ophi, sizeof(float) * width * height);
	cudaMalloc(&oG, sizeof(float) * width * height);


	intensityGradient<<<dimGrid, dimBlock>>>(oNR, ophi, oG, height, width);

	cudaDeviceSynchronize(); //Sincronizo

	cudaFree(oNR); //Libero este espacio de memoria que no se va a usar mas

	uint8_t *opedge;
	cudaMalloc(&opedge, sizeof(uint8_t) * width * height);

	getEdges<<<dimGrid, dimBlock>>>(oG, ophi, opedge, height, width);

	cudaDeviceSynchronize();
	
	cudaFree(ophi); //Libero memoria no usada

	// Hysteresis Thresholding
	lowthres = level/2;
	hithres  = 2*(level);

	uint8_t *oimage_out;
	cudaMalloc(&oimage_out, sizeof(uint8_t) * width * height);

	hysteresis_thresholding<<<dimGrid, dimBlock>>>(oG, opedge, oimage_out, height, width,
												lowthres, hithres);

	cudaDeviceSynchronize();

	cudaMemcpy(image_out, oimage_out, sizeof(uint8_t) * width * height, cudaMemcpyDeviceToHost);


	cudaFree(oG);
	cudaFree(opedge);
	cudaFree(oimage_out);


}

		// for(i=3; i<height-3; i++)
		// for(j=3; j<width-3; j++)
		// {
		// 	image_out[i*width+j] = 0;
		// 	if(G[i*width+j]>hithres && pedge[i*width+j])
		// 		image_out[i*width+j] = 255;
		// 	else if(pedge[i*width+j] && G[i*width+j]>=lowthres && G[i*width+j]<hithres)
		// 		// check neighbours 3x3
		// 		for (ii=-1;ii<=1; ii++)
		// 			for (jj=-1;jj<=1; jj++)
		// 				if (G[(i+ii)*width+j+jj]>hithres)
		// 					image_out[i*width+j] = 255;
		// }



void houghtransform(uint8_t *im, int width, int height, uint32_t *accumulators, int accu_width, int accu_height, 
	float *sin_table, float *cos_table)
{
	int i, j, theta;

	float hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);

	for(i=0; i<accu_width*accu_height; i++)
		accumulators[i]=0;	

	float center_x = width/2.0; 
	float center_y = height/2.0;
	for(i=0;i<height;i++)  
	{  
		for(j=0;j<width;j++)  
		{  
			if( im[ (i*width) + j] > 250 ) // Pixel is edge  
			{  
				for(theta=0;theta<180;theta++)  
				{  
					float rho = ( ((float)j - center_x) * cos_table[theta]) + (((float)i - center_y) * sin_table[theta]);
					accumulators[ (int)((round(rho + hough_h) * 180.0)) + theta]++;

				} 
			} 
		} 
	}
}


void getlines(int threshold, uint32_t *accumulators, int accu_width, int accu_height, int width, int height, 
	float *sin_table, float *cos_table,
	int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines)
{
	int rho, theta, ii, jj;
	uint32_t max;

	for(rho=0;rho<accu_height;rho++)
	{
		for(theta=0;theta<accu_width;theta++)  
		{  

			if(accumulators[(rho*accu_width) + theta] >= threshold)  
			{  
				//Is this point a local maxima (9x9)  
				max = accumulators[(rho*accu_width) + theta]; 
				for(int ii=-4;ii<=4;ii++)  
				{  
					for(int jj=-4;jj<=4;jj++)  
					{  
						if( (ii+rho>=0 && ii+rho<accu_height) && (jj+theta>=0 && jj+theta<accu_width) )  
						{  
							if( accumulators[((rho+ii)*accu_width) + (theta+jj)] > max )  
							{
								max = accumulators[((rho+ii)*accu_width) + (theta+jj)];
							}  
						}  
					}  
				}  

				if(max == accumulators[(rho*accu_width) + theta]) //local maxima
				{
					int x1, y1, x2, y2;  
					x1 = y1 = x2 = y2 = 0;  

					if(theta >= 45 && theta <= 135)  
					{
						if (theta>90) {
							//y = (r - x cos(t)) / sin(t)  
							x1 = width/2;  
							y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width;  
							y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);  
						} else {
							//y = (r - x cos(t)) / sin(t)  
							x1 = 0;  
							y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width*2/5;  
							y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2); 
						}
					} else {
						//x = (r - y sin(t)) / cos(t);  
						y1 = 0;  
						x1 = ((float)(rho-(accu_height/2)) - ((y1 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
						y2 = height;  
						x2 = ((float)(rho-(accu_height/2)) - ((y2 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
					}
					x1_lines[*lines] = x1;
					y1_lines[*lines] = y1;
					x2_lines[*lines] = x2;
					y2_lines[*lines] = y2;
					(*lines)++;
				}
			}
		}
	}
}


void lane_assist_GPU(uint8_t *im, int height, int width,
	uint8_t *imEdge, float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float *sin_table, float *cos_table, 
	uint32_t *accum, int accu_height, int accu_width,
	int *x1, int *y1, int *x2, int *y2, int *nlines)
{

	int threshold;

	/* Canny */
	canny(im, imEdge, 1000.0f, //level
		height, width , G, pedge, NR, phi);


	
	houghtransform(imEdge, width, height, accum, accu_width, accu_height, sin_table, cos_table);

	if (width>height) threshold = width/6;
	else threshold = height/6;

	getlines(threshold, accum, accu_width, accu_height, width, height, 
		sin_table, cos_table,
		x1, y1, x2, y2, nlines);

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
