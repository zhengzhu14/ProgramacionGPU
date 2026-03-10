# include <cstdio>
# include <cmath>
# include <cuda_runtime.h>

const int SIZE = 1024; //Valor auxiliar para calentar
const size_t sizes [] = {1<<10, 4<<10, 16<<10, 1 << 20,
                        16 << 20, 64 << 20, 256 << 20};

const int REPS = 10;

void warmup(float *hb1, float *db1){

    cudaMemcpy(db1, hb1, SIZE, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void transferHD(cudaEvent_t start, cudaEvent_t stop, size_t size, float *hb, float *db){
    float ms = 0;
    //Medicion Host to Device

    for (int i = 0; i < REPS; ++i){
        cudaEventRecord(start);
        cudaMemcpy(db, hb, size, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        printf("%f\n", ms);
    }
    
    //Fin de medicion Host to Device
}

void transferDH(cudaEvent_t start, cudaEvent_t stop, size_t size, float *hb, float *db){
    float ms = 0;
    //Medicion  Device to Host

    for (int i = 0; i < REPS; ++i){
        cudaEventRecord(start);
        cudaMemcpy(hb, db , size, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        printf("%f\n", ms);
    }

    //Fin de medicion  Device to Host
}


void transferDD(cudaEvent_t start, cudaEvent_t stop, size_t size, float *db1, float *db2){
    float ms = 0;
    for (int i = 0; i < REPS; ++i){
        cudaEventRecord(start);
        cudaMemcpy(db1, db2, size, cudaMemcpyDeviceToDevice);
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        printf("%f\n", ms);
    }
}


int main (){

    //Calentamiento
    float *hb = (float*)malloc(SIZE);
    float *db = nullptr;
    cudaMalloc(&db, SIZE);

    warmup(hb, db);

    free(hb); cudaFree(db);


    float *hb1;
    float *hb2;
    float* db1 = nullptr;
    float* db2 = nullptr;

    cudaEvent_t start; cudaEventCreate(&start);
    cudaEvent_t stop; cudaEventCreate(&stop);

    for(size_t sz: sizes){

        printf("Tamaño actual %ld\n", sz);
        //Prueba con pageable

        printf("Prueba con pageable\n");
        hb1 = (float*)malloc(sz);
        hb2 = (float*)malloc(sz);
        cudaMalloc(&db1, sz);
        cudaMalloc(&db2, sz);
        printf("Host to Device\n");
        transferHD(start, stop, sz, hb1, db1);
        printf("Device to Host\n");
        transferDH(start, stop, sz, hb2, db2);

        free(hb1);free(hb2); 
        //Fin de pageable

        printf("\n");

        //Prueba con pinned
        printf("Prueba con pinned.\n");
        cudaMallocHost(&hb1, sz);
        cudaMallocHost(&hb2, sz);
        printf("Host to Device\n");
        transferHD(start, stop, sz, hb1, db1);
        printf("Device to Host\n");
        transferDH(start, stop, sz, hb2, db2);

        cudaFreeHost(hb1);
        cudaFreeHost(hb2);
        //Fin de pinned


        printf("\n");

        //Device to device
        printf("Device to Device\n");
        transferDD(start, stop, sz, db1, db2);

        //Fin Device to device

        cudaFree(db1); cudaFree(db2);
        printf("\n");
    }

    return 0;
}