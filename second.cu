#include <stdio.h>

__global__ void Kernel(float *Pd, int Width) {

  // Calculate the column index of the Pd element, denote by x
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  // Calculate the row index of the Pd element, denote by y
  int y = threadIdx.y + blockIdx.y * blockDim.y; 

  float Pvalue = 69;
  // each thread computes one element of the output matrix Pd.      


  // write back to the global memory
  Pd[y*Width + x] = Pvalue;
}

main(void){

    void MatrixMultiplication(float *, int);
    
    const int Width = 1024;

    int size = Width * Width * sizeof(float);
    float *P;

    // allocate memory on the CPU

    P = (float*)malloc(size);



    MatrixMultiplication( P, Width);

    // free the memory allocated on the CPU

    free( P );

    return 0;
}
void MatrixMultiplication(float *P, int Width) {

    int size = Width * Width * sizeof(float);
    float *Pd;
    
    // capture start time
    cudaEvent_t     start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    // allocate memory on the GPU

    cudaMalloc((void**)&Pd, size);

    // transfer M and N to device memory

    // kernel invocation code
    dim3 dimBlock(32, 32);
    dim3 dimGrid(Width/32, Width/32);
    Kernel<<<dimGrid, dimBlock>>>(Pd, Width);

    // transfer P from device     
    cudaMemcpy(P,Pd,size,cudaMemcpyDeviceToHost);

    // get stop time, and display the timing results
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float   elapsedTime;
    cudaEventElapsedTime( &elapsedTime,
                                        start, stop );
    printf( "Time to generate:  %3.1f ms\n", elapsedTime );

    // free the memory allocated on the GPU

    cudaFree(Pd);

    for(int i = 0; i < 10; i++){
        for(int j = 0;j < 10; j++){
            printf("%f   ", P[i*Width + j]);
        }
    }

    // destroy events to free memory
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
}