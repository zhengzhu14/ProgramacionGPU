
#include <sycl/sycl.hpp>

#include <math.h>

using  namespace  sycl;

#define MAX_WINDOW_SIZE 5*5
#define GROUP_WIDTH 16
#define GROUP_HEIGHT 16

void remove_noise_SYCL(sycl::queue Q, float *im, float *image_out, 
	float thredshold, int window_size,
	int height, int width)
{
	/* To Do */
	int ws2 = (window_size-1)>>1; 

	Q.submit([&](handler & h){
		range global = range <2> (height, width);
		range local = range <2> (GROUP_HEIGHT, GROUP_WIDTH);

		int tile_width = GROUP_WIDTH + 2*ws2;
		int tile_height = GROUP_HEIGHT + 2*ws2;

		local_accessor<float, 2> tile (range <2>(tile_height, tile_width), h); //Creo la memoria compartida

		h.parallel_for(nd_range <2> (global, local), [=](nd_item <2> item){
			int i = item.get_global_id()[0];
			int j = item.get_global_id()[1];

			int li = item.get_local_id()[0];
			int lj = item.get_local_id()[1];

			int group_i = item.get_group(0);
			int group_j = item.get_group(1);

			int thread_id = li*GROUP_HEIGHT + lj; //Calculo el índide absoluto del hilo, como si estuvieramos en un grid de 1 dimension, parecido en CUDA
			
			int tile_size = tile_height*tile_width;
			
			//Cargo los datos al igual que en cuda
			for (int i = thread_id; i < tile_size; i+=GROUP_WIDTH*GROUP_HEIGHT){
				//De esta manera hago que hilos consecutivos lean posiciones consecutivas.
				//Posiciones dentro de la matriz shared
				int srow = i/tile_height;
				int scol = i%tile_width;


				//Obtengo la posicion absoluta de la memoria global
				int gx = group_j*GROUP_WIDTH + scol - ws2;
				int gy = group_i*GROUP_HEIGHT + srow - ws2;

				//Las posiciones que se salen de rango, de esta manera simplemente
				//copio los valores de los bordes para extenderlo
				gx = max(0, min(gx, width - 1));
				gy = max(0, min(gy, height - 1));

				tile[srow][scol] = im[gy*width + gx];

			}

			item.barrier();
			//Tengo los datos cargados

			if (i < ws2 || i >= height - ws2 || j < ws2 || j >= width - ws2) 
			return;

			float window[MAX_WINDOW_SIZE];
			int idx = 0;

			//Cargo en la ventana
			for (int ii = -ws2; ii <= ws2; ++ii){
				for (int jj = -ws2; jj <= ws2; ++jj){
					window[(ii+ws2)*window_size + jj+ws2] = tile[li + ii + ws2][lj + jj + ws2];
				}
			}


			//Buble sort
			int ai, aj;
			float tmp;
			int size = window_size*window_size;

			for (ai=1; ai<size; ai++)
				for (aj=0 ; aj<size - ai; aj++)
					if (window[aj] > window[aj+1]){
						tmp = window[aj];
						window[aj] = window[aj+1];
						window[aj+1] = tmp;
					}

			int median = window[(window_size*window_size-1)>>1];

			if (fabsf((median-tile[(li + ws2)][(lj + ws2)])/median) <=thredshold)
				image_out[i*width + j] = tile[(li + ws2)][(lj + ws2)];
			else
				image_out[i*width + j] = median;

		});

	}).wait();

}
