#include <stdio.h>
//#include <complex.h>

struct ncomplex{
	float x;
	float y;
};


__global__ void create_fractal2(float *imgd, int width){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y; 

	

	ncomplex z0;
	z0.x = 1.0;
	z0.y = 0.5;

	imgd[y*width + x] = z0.x;


}


ncomplex multCC(struct complex z1, struct complex z2);

main(void){
	
	ncomplex z1;
	z1.x = 1.0;
	z1.y = 2.0;
	
	int width = 10;

	int size = width*width*sizeof(float);

	float *img;

	img = (float*)malloc(size);

	float *imgd;

	cudaMalloc((void**)&imgd, size);


	


	dim3 dimBlock(32, 32);
	dim3 dimGrid(1, 1);

	create_fractal2<<<dimGrid, dimBlock>>>(imgd, width);

	cudaMemcpy(img,imgd,size,cudaMemcpyDeviceToHost);

	float hldr = *img;

    for(int i = 0; i < 10; i++){
        for(int j = 0;j < 10; j++){
            printf("%f   ", img[i*width + j]);
        }
    }
	cudaFree(imgd);
	free(img);

}

ncomplex multCC(struct ncomplex z1, struct ncomplex z2){
	float realPart = z1.x*z2.x - z1.y*z2.y;
	float imPart = z1.x*z2.y + z1.y*z2.x;
	ncomplex results;
	results.x = realPart;
	results.y = imPart;

	return results;
}