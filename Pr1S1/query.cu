//Ejercicio 1 parte C

# include <cstdio>
# include <cmath>
# include <cuda_runtime.h>


int main (int argc, char**argv){
    int ndev = 0;
    cudaGetDeviceCount(&ndev);
    for(int d = 0; d < ndev; ++d){
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);

        printf("Device %d : %s\n", d, p.name);
        printf("Compute capability: %d.%d\n", p.major, p.minor);

        printf("Numero de SMs: %d\n", p.multiProcessorCount);
        printf("El tamanyo del warp es: %d\n", p.warpSize);
        printf("Memoria global total: %ld\n", p.totalGlobalMem);
        printf("Memoria compartida por bloque: %ld\n", p.sharedMemPerBlock);
        printf("Registros por bloque: %d\n", p.regsPerBlock);
        printf("Maximo de hilos por bloque: %d\n", p.maxThreadsPerBlock);
        printf("Maximo de hilos por SM: %d\n", p.maxThreadsPerMultiProcessor);

        
    } 


    return 0;
}


