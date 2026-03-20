#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "routinesCPU.h"
#include "routinesGPU.h"
#include "png_io.h"

/* Time */
#include <sys/time.h>
#include <sys/resource.h>
#include <chrono>

#define EJECUCIONES 20

const char* IMG_EXTENSION = ".png";
const char* CPU_EXTENSION = "_CPU";
const char* GPU_EXTENSION = "_GPU";

const char* FOLDER_EXTENSION = "imagenes/";

static struct timeval tv0;
double get_time()
{
	double t;
	gettimeofday(&tv0, (struct timezone*)0);
	t = ((tv0.tv_usec) + (tv0.tv_sec)*1000000);

	return (t);
}

double get_time_ms() 
{
    // Using steady_clock for consistent measurement
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    
    // Cast to milliseconds
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    
    return static_cast<double>(millis.count());
}

 

int main(int argc, char **argv)
{
	

	uint8_t *imtmp, *im;
	int width, height;

	float sin_table[180], cos_table[180];
	int nlines=0; 
	int x1[10], x2[10], y1[10], y2[10];

	int l;
	double t0, t1;

	int warmup = 4;
	double media = 0.0;


	/* Only accept a concrete number of arguments */
	if(argc != 3)
	{
		printf("./exec image.png [c/g]\n");
		exit(-1);
	}

	
	size_t tam = strlen(argv[1]) + strlen(IMG_EXTENSION) + 1; 
	char* imEntrada = (char*)malloc(tam);

	snprintf(imEntrada, tam, "%s%s", argv[1], IMG_EXTENSION);

	/* Read images */
	imtmp = read_png_fileRGB(imEntrada, &width, &height);
	im    = image_RGB2BW(imtmp, height, width);

	printf("Tamanyo de la imagen es:\n-Altura = %d.\n-Anchura = %d\n\n", height, width);

	init_cos_sin_table(sin_table, cos_table, 180);	

	// Create temporal buffers 
	uint8_t *imEdge = (uint8_t *)malloc(sizeof(uint8_t) * width * height);
	float *NR = (float *)malloc(sizeof(float) * width * height);
	float *G = (float *)malloc(sizeof(float) * width * height);
	float *phi = (float *)malloc(sizeof(float) * width * height);
	float *Gx = (float *)malloc(sizeof(float) * width * height);
	float *Gy = (float *)malloc(sizeof(float) * width * height);
	uint8_t *pedge = (uint8_t *)malloc(sizeof(uint8_t) * width * height);

	//Create the accumulators
	float hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);
	int accu_height = hough_h * 2.0; // -rho -> +rho
	int accu_width  = 180;
	uint32_t *accum = (uint32_t*)malloc(accu_width*accu_height*sizeof(uint32_t));



	char* imSalida;
	tam += strlen(FOLDER_EXTENSION);
	switch (argv[2][0]) {
		case 'c':
			tam+= strlen(CPU_EXTENSION);
			imSalida = (char*)malloc(tam);
			snprintf(imSalida, tam, "%s%s%s%s", FOLDER_EXTENSION, argv[1], CPU_EXTENSION, IMG_EXTENSION);
			for (int i = 0; i < EJECUCIONES; i++){
				nlines = 0;
				t0 = get_time_ms();
				lane_assist_CPU(im, height, width, 
					imEdge, NR, G, phi, Gx, Gy, pedge,
					sin_table, cos_table,
					accum, accu_height, accu_width,
					x1, y1, x2, y2, &nlines);
				t1 = get_time_ms();
				media+=(t1 -t0);
				
			}
			printf("CPU Exection time %f ms.\n", media/EJECUCIONES);
			break;
		case 'g':
			tam+= strlen(GPU_EXTENSION);
			imSalida = (char*)malloc(tam);
			snprintf(imSalida, tam, "%s%s%s%s", FOLDER_EXTENSION, argv[1], GPU_EXTENSION, IMG_EXTENSION);
			//warmup
			printf("EJECUCIONES DE WARM-UP\n");
			for (int i = 0; i < warmup; ++i){
				lane_assist_GPU(im, height, width, 
				imEdge,
				sin_table, cos_table,
				accum, accu_height, accu_width,
				x1, y1, x2, y2, &nlines);
				nlines = 0;
			}
			
			printf("\n");
			printf("EJECUCIONES DE REALES\n");

			for (int i = 0; i < EJECUCIONES; i++){
				nlines = 0;
				t0 = get_time_ms();
				lane_assist_GPU(im, height, width, 
					imEdge,
					sin_table, cos_table,
					accum, accu_height, accu_width,
					x1, y1, x2, y2, &nlines);
				t1 = get_time_ms();
				media += (t1 - t0);
			}
			
			printf("GPU Exection time %f ms.\n", media/EJECUCIONES);
			
			break;
		default:
			printf("Not Implemented yet!!\n");
	}

	for (int l=0; l<nlines; l++)
		printf("(x1,y1)=(%d,%d) (x2,y2)=(%d,%d)\n", x1[l], y1[l], x2[l], y2[l]);

	

	draw_lines(imtmp, width, height, x1, y1, x2, y2, nlines);

	write_png_fileRGB(imSalida, imtmp, width, height);

	free(imEntrada);
	free(imSalida);
}
