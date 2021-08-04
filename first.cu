#include <stdio.h>

__global__ void create_fractal2(float *imgd, int width){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y; 

	imgd[y*width + x] = 69.0;


}

main(void){


	int width = 1024;

	int size = width*width*sizeof(float);

	float *img;

	img = (float*)malloc(size);

	float *imgd;

	cudaMalloc((void**)&imgd, size);


	


	dim3 dimBlock(32, 32);
	dim3 dimGrid(width/32, width/32);

	create_fractal2<<<dimGrid, dimBlock>>>(imgd, width);

	cudaMemcpy(img,imgd,size,cudaMemcpyDeviceToHost);

	float hldr = *img;

	printf("%f\n",*img);
	cudaFree(imgd);
	free(img);

}