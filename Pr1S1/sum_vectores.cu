# include <cstdio>
# include <cmath>
# include <cuda_runtime.h>

/* Time */
#include <sys/time.h>
#include <sys/resource.h>

const size_t N[] = {1 <<10, 1<<12, 1<<14, 1<<16, 1<<18, 1<<20, 1<<22, 1<<24, 1<<26};

static struct timeval tv0;
double getMicroSeconds()
{
    double t;
    gettimeofday(&tv0, (struct timezone*)0);
    t = ((tv0.tv_usec) + (tv0.tv_sec) * 1000000.0);
    return t;
}

void inicializarHost(float* v, size_t n){
    for(size_t i = 0; i < n; ++i){
        v[i] = 1.0;
    }
}

void sumar_vectores_host(float * C, float *A, float* B, size_t n){
    for(size_t i = 0; i < n; ++i){
        C[i] = A[i] + B[i];
    }
}


int check(const float *GPU, const float *CPU, int n)
{
    for (int i = 0; i < n; i++)
        if (GPU[i] != CPU[i]) return 1;
    return 0;
}


__global__ void sumar_vectores_device(const float * A, const float *B, float * C, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n){
        C[i] = A[i] + B[i];
    }

}

int main(){
    float *hC, *hA, *hB;

    float* dC, *dA, *dB;
    float* sdC;
    
    float msHD, msKernel, msDH = 0;
    float msTotal = 0;

    cudaEvent_t start; 
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (size_t n: N){
        int numElementos = n/sizeof(float);
        printf("Tamanyo de N es: %ld\n", n);

        hC = (float*) malloc(n);
        hA = (float*) malloc(n);
        hB = (float*) malloc(n);
        sdC = (float*) malloc(n);

        inicializarHost(hA, numElementos);
        inicializarHost(hB, numElementos);

        double t0 = getMicroSeconds();
        sumar_vectores_host(hC, hA, hB, numElementos);
        double t1 = getMicroSeconds();
        double secCPU = (t1 - t0) / 1e6;
        printf("-Tiempo CPU: %f\n", secCPU);

        

        cudaMalloc(&dC, n);
        cudaMalloc(&dA, n);
        cudaMalloc(&dB, n);

        //CudaMemCpy() host to device
        cudaEventRecord(start);
        
        cudaMemcpy(dA, hA, n, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, n, cudaMemcpyHostToDevice);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&msHD, start, stop);
        printf("-Tiempo GPU Host->Device: %f\n", msHD);
        //--------------------------

        //Ejecucion del Kernel
        int threads = 256;
        int blocks = (n + threads - 1)/threads;

        cudaEventRecord(start);

        sumar_vectores_device <<<blocks, threads>>>(dA, dB, dC, numElementos);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&msKernel, start, stop);
        printf("-Tiempo GPU Kernel: %f\n", msKernel);
        //-------------------

        //CudaMemCpy() Device to Host
        cudaEventRecord(start);
        
        cudaMemcpy(sdC, dC, n, cudaMemcpyDeviceToHost);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&msDH, start, stop);
        printf("-Tiempo GPU Device->Host: %f\n", msDH);
        //--------------------------

        msTotal = msHD + msKernel + msDH;
        printf("-Tiempo GPU total: %f\n", msTotal);

        if (check(sdC, hC, numElementos))
            printf("Transpose CPU-GPU differs!!\n");
        else
            printf("Check OK\n");

        free(hC);
        free(hA);
        free(hB);
        free(sdC);
        cudaFree(dC);
        cudaFree(dA);
        cudaFree(dB);
    }
    


    return 0;
}