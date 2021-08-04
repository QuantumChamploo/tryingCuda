#include <stdio.h>
//#include <complex.h>

struct ncomplex{
	float x;
	float y;
};

__device__ ncomplex multCC(struct ncomplex z1, struct ncomplex z2);


__global__ void create_fractal2(float *imgd, int width,int min_x, int max_x, int min_y, int max_y, int iters){

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y; 

	double pixel_size_x = (max_x - min_x)/double(width);
	double pixel_size_y = (max_y - min_y)/double(width);

	ncomplex z0, z1;
	z0.x = min_x + x*pixel_size_x;
	z0.y = min_y + y*pixel_size_y;

	z1.x = 0.0;
	z1.y = 0.0;

	int result = -1;

	for(int k = 0; k < iters; k++){
		ncomplex z2 = multCC(z1,z1);
		double real_part = z2.x + z0.x;
		double imag_part = z2.y + z0.y;
		double ccSquared = real_part*real_part + imag_part*imag_part;
		z1.x = real_part;
		z1.y = imag_part;
		if(ccSquared >= 4.0){
			result = k;
			break;
		}
	}

	if(result == -1){
	 result = iters;
	}



	imgd[y*width + x] = result;


}




main(void){
	
	ncomplex z1;
	z1.x = 1.0;
	z1.y = 2.0;
	
	int width = 64;

	int size = width*width*sizeof(float);

	float *img;

	img = (float*)malloc(size);

	float *imgd;

	cudaMalloc((void**)&imgd, size);

	double min_x, max_x, min_y, max_y;
	int iters;

	min_x = -2.0;
	max_x = 1.0;
	min_y = -1.0;
	max_y = 1.0;
	iters = 20;


	


	dim3 dimBlock(32, 32);
	dim3 dimGrid(width/32, width/32);

	create_fractal2<<<dimGrid, dimBlock>>>(imgd, width, min_x, max_x, min_y, max_y, iters);

	cudaMemcpy(img,imgd,size,cudaMemcpyDeviceToHost);

	float hldr = *img;

    for(int i = 0; i < width; i++){
        for(int j = 0;j < width; j++){
            printf("%.00f   ", img[i*width + j]);
        }
        printf("\n");
    }
	cudaFree(imgd);
	free(img);

}

__device__ ncomplex multCC(struct ncomplex z1, struct ncomplex z2){
	float realPart = z1.x*z2.x - z1.y*z2.y;
	float imPart = z1.x*z2.y + z1.y*z2.x;
	ncomplex results;
	results.x = realPart;
	results.y = imPart;

	return results;
}