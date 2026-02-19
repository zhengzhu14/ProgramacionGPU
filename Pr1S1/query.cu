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
        
    }


    return 0;
}


