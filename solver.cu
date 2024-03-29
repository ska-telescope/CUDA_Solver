
// Copyright 2019 Adam Campbell, Seth Hall, Andrew Ensor
// Copyright 2019 High Performance Computing Research Laboratory, Auckland University of Technology (AUT)

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <math_constants.h>
#include <device_launch_parameters.h>
#include <numeric>

#include "solver.h"

// Flags early termination of deconvolution GPU kernels
__device__ bool d_exit_early = false;
// Tracks number of distinct extracted sources (compressed)
__device__ int d_source_counter = 0;

void init_config(Config *config)
{
	config->grid_size = 18000;
	
	config->right_ascension = true;

	config->force_zero_w_term = false;
	
	config->cell_size = 6.39708380288950e-6;
	
	config->frequency_hz = 100e6;

	// Specify the number of kernels used in w projection
	config->num_wproj_kernels = 339;

	config->max_w = 7083.386050;

	config->w_scale = pow(config->num_wproj_kernels - 1, 2.0) / config->max_w;

	// Kernel oversampling factor
	config->oversampling = 4; // Oxford configuration
	
	// Used to convert visibility uvw coordinates into grid coordinates
	config->uv_scale = config->grid_size * config->cell_size;
	
	// Number of visibilities to process
	config->num_visibilities = 100;

	config->output_dirty_image = "../data/dirty_image.csv";

	// File location to load pre-calculated w-projection kernel
	config->kernel_real_source_file = "../data/w-proj_kernels_real.csv";

	config->kernel_imag_source_file = "../data/w-proj_kernels_imag.csv";

	// Specify file which holds the supports for all kernels
	config->kernel_support_file = "../data/w-proj_supports.csv";

	// File location to load visibility uvw coordinates  
	config->visibility_source_file = "../data/el82-70.txt";   

	// Number of CUDA threads per block - this is GPU specific
	config->gpu_max_threads_per_block = 1024;

	//Number of CUDA threads per block dimension in x and y - this is GPU specific and used for FFT and CC
	config->gpu_max_threads_per_block_dimension = 32;

	// Enable/disable CUDA timing of gridding kernel 
	config->time_gridding = true; 

	// Enable/disable CUDA timing of deconvolution kernel 
	config->time_deconvolution = true; 	

	//Enable/disable the iFFT and Convolution Correction part of the pipeline.
	config->perform_iFFT_CC = true;

	// Enable/disable the execution of Hogbom CLEAN on gridded data
	config->perform_deconvolution = true;

	config->psf_size = 18000;

	config->psf_source_file = "../data/el8270_psf_9000k_real.csv";

	config->output_residual_image = "../data/residual_image.csv";

	config->number_minor_cycles = 60;

	config->loop_gain = 0.1; // 0.1 is typical

	config->output_model_sources_file = "../data/model_sources.csv";

	config->weak_source_percent = 0.01; // ex: 0.01 = 1%

	// Used to determine if we are extracting noise, based on the assumption
	// that located source < noise_detection_factor * running_average
	config->noise_detection_factor = 2.0;
}

void execute_gridding(Config *config, Visibility *vis_uvw, 
	Complex *vis_intensities, int num_visibilities, Complex *kernel,
	int2 *kernel_supports, int num_kernel_samples, PRECISION2 *d_input_grid)
{
	cudaEvent_t start, stop;
	// Handles for GPU memory
	PRECISION2 *d_kernel;
	PRECISION3 *d_vis_uvw;
	PRECISION2 *d_vis;
	int2 *d_supports;

	//PRECISION2 *d_input_grid2;

	// int grid_size_square = config->grid_size * config->grid_size;
	// CUDA_CHECK_RETURN(cudaMalloc(&(*d_input_grid), sizeof(PRECISION2) * grid_size_square));
	// cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_kernel, sizeof(PRECISION2) * num_kernel_samples));
	CUDA_CHECK_RETURN(cudaMemcpy(d_kernel, kernel, sizeof(PRECISION2) * num_kernel_samples,
		cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_supports, sizeof(int2) * config->num_wproj_kernels));
	CUDA_CHECK_RETURN(cudaMemcpy(d_supports, kernel_supports, sizeof(int2) * config->num_wproj_kernels,
		cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	// Allocate and copy visibility uvw to device
	CUDA_CHECK_RETURN(cudaMalloc(&d_vis_uvw, sizeof(PRECISION3) * num_visibilities));
	CUDA_CHECK_RETURN(cudaMemcpy(d_vis_uvw, vis_uvw, sizeof(PRECISION3) * num_visibilities,
		cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	// Allocate memory on device for storing extracted complex visibilities
	CUDA_CHECK_RETURN(cudaMalloc(&d_vis, sizeof(PRECISION2) * num_visibilities));
	CUDA_CHECK_RETURN(cudaMemcpy(d_vis, vis_intensities, sizeof(PRECISION2) * num_visibilities,
		cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	int max_threads_per_block = min(config->gpu_max_threads_per_block, num_visibilities);
	int num_blocks = (int) ceil((double) num_visibilities / max_threads_per_block);
	dim3 kernel_blocks(num_blocks, 1, 1);
	dim3 kernel_threads(max_threads_per_block, 1, 1);

	printf(">>> INFO: Using %d blocks, %d threads, for %d visibilities...\n",
		num_blocks, max_threads_per_block, num_visibilities);

	// Optional timing functionality
	if(config->time_gridding)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
	}

	// Execute gridding kernel
	gridding<<<kernel_blocks, kernel_threads>>>(d_input_grid, d_kernel, d_supports, 
		d_vis_uvw, d_vis, num_visibilities, config->oversampling,
		config->grid_size, config->uv_scale, config->w_scale);
	cudaDeviceSynchronize();

	// Optional report on timing
	if(config->time_gridding)
	{
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf(">>> UPDATE: GPU accelerated gridding completed in %f milliseconds...\n", milliseconds);
	}
	//free gridding memory 
	CUDA_CHECK_RETURN(cudaFree(d_kernel));
	CUDA_CHECK_RETURN(cudaFree(d_vis_uvw));
	CUDA_CHECK_RETURN(cudaFree(d_vis));
	CUDA_CHECK_RETURN(cudaFree(d_supports));
}

void execute_CC(Config *config, PRECISION *prolate, PRECISION *d_output_image)
{
	PRECISION *d_prolate;
	int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, config->grid_size);
	int num_blocks_per_dimension = (int) ceil((double) config->grid_size / max_threads_per_block_dimension);
	dim3 cc_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
	dim3 cc_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);

	printf("UPDATE >>> PERFORMING CONVOLUTION CORRECTION... \n");
	CUDA_CHECK_RETURN(cudaMalloc(&d_prolate, sizeof(PRECISION) * config->grid_size/2));
	CUDA_CHECK_RETURN(cudaMemcpy(d_prolate, prolate, sizeof(PRECISION) * config->grid_size/2, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	execute_convolution_correction<<<cc_blocks, cc_threads>>>(d_output_image, d_prolate, config->grid_size);
	cudaDeviceSynchronize();
	printf("UPDATE >>> CONVOLUTION CORRECTION DONE \n...");
	CUDA_CHECK_RETURN(cudaFree(d_prolate));	

}


void execute_CUDA_iFFT(Config *config, PRECISION2 *d_input_grid, PRECISION *d_output_image)
{
	int grid_size = config->grid_size;

	int max_threads_per_block_dimension = min(config->gpu_max_threads_per_block_dimension, grid_size);
	int num_blocks_per_dimension = (int) ceil((double) grid_size / max_threads_per_block_dimension);
	dim3 shift_blocks(num_blocks_per_dimension, num_blocks_per_dimension, 1);
	dim3 shift_threads(max_threads_per_block_dimension, max_threads_per_block_dimension, 1);

	printf("Shifting grid data for 2D FFT...\n");
	// Perform 2D FFT shift
	fftshift_2D_complex<<<shift_blocks, shift_threads>>>(d_input_grid, grid_size);
	cudaDeviceSynchronize();

	printf("Performing 2D FFT...\n");
	// Perform 2D FFT
	cufftHandle fft_plan;
	CUFFT_SAFE_CALL(cufftPlan2d(&fft_plan, grid_size, grid_size, CUFFT_C2C_PLAN));
	CUFFT_SAFE_CALL(CUFFT_EXECUTE_C2C(fft_plan, d_input_grid, d_input_grid, CUFFT_INVERSE));
	cudaDeviceSynchronize();

	printf("Shifting grid data back into place...\n");
	// Perform 2D FFT shift back
	fftshift_2D_complex_to_real<<<shift_blocks, shift_threads>>>(d_input_grid, d_output_image, grid_size);

	cudaDeviceSynchronize();

	printf("FFT COMPLETE ... ");
	//CUDA_CHECK_RETURN(cudaFree(d_input_grid));	
	printf("Freeing input grid ... ");
}

__global__ void execute_convolution_correction(PRECISION *grid, const PRECISION *prolate, const int grid_size)
{
	const int row_index = threadIdx.y + blockDim.y * blockIdx.y;
    const int col_index = threadIdx.x + blockDim.x * blockIdx.x;

    if(row_index >= grid_size || col_index >= grid_size)
    	return;

    const int grid_index = row_index * grid_size + col_index;
    const int half_grid_size = grid_size / 2;

    const PRECISION taper = prolate[abs(col_index - half_grid_size)] * prolate[abs(row_index - half_grid_size)];


    grid[grid_index] = (ABS(taper) > (1E-10)) ? grid[grid_index] / taper  : 0.0;
}

void create_1D_half_prolate(PRECISION *prolate, int grid_size)
{
	int grid_half_size = grid_size / 2;
	double nu = 0.0;

	for(int index = 0; index < grid_half_size; ++index)
	{
		nu = ((double)index / (double)grid_half_size);
		prolate[index] = (PRECISION)calc_spheroidal_sample(nu);
	}
}


// Calculates a sample on across a prolate spheroidal
// Note: this is the Fred Schwabb approximation technique
double calc_spheroidal_sample(double nu)
{
    static double p[] = {0.08203343, -0.3644705, 0.627866, -0.5335581, 0.2312756,
        0.004028559, -0.03697768, 0.1021332, -0.1201436, 0.06412774};
    static double q[] = {1.0, 0.8212018, 0.2078043,
        1.0, 0.9599102, 0.2918724};

    int part = 0;
    int sp = 0;
    int sq = 0;
    double nuend = 0.0;
    double delta = 0.0;
    double top = 0.0;
    double bottom = 0.0;

    if(nu >= 0.0 && nu < 0.75)
    {
        part = 0;
        nuend = 0.75;
    }
    else if(nu >= 0.75 && nu < 1.0)
    {
        part = 1;
        nuend = 1.0;
    }
    else
        return 0.0;

    delta = nu * nu - nuend * nuend;
    sp = part * 5;
    sq = part * 3;
    top = p[sp];
    bottom = q[sq];

    for(int i = 1; i < 5; i++)
        top += p[sp+i] * pow(delta, i);
    for(int i = 1; i < 3; i++)
        bottom += q[sq+i] * pow(delta, i);
    return (bottom == 0.0) ? 0.0 : top/bottom;
}

__global__ void fftshift_2D_complex(PRECISION2 *grid, const int width)
{
    int row_index = threadIdx.y + blockDim.y * blockIdx.y;
    int col_index = threadIdx.x + blockDim.x * blockIdx.x;
 
    if(row_index >= width || col_index >= width)
        return;
 
    int a = 1 - 2 * ((row_index + col_index) & 1);
    grid[row_index * width + col_index].x *= a;
    grid[row_index * width + col_index].y *= a;
}


__global__ void fftshift_2D_complex_to_real(PRECISION2 *inputImage, PRECISION *output_image, const int width)
{
    int row_index = threadIdx.y + blockDim.y * blockIdx.y;
    int col_index = threadIdx.x + blockDim.x * blockIdx.x;
 
    if(row_index >= width || col_index >= width)
        return;
 	int index = row_index * width + col_index;

    int a = 1 - 2 * ((row_index + col_index) & 1);
    output_image[index] =  inputImage[index].x * a;
}


__global__ void fftshift_2D_real(PRECISION *output_image, const int width)
{
    int row_index = threadIdx.y + blockDim.y * blockIdx.y;
    int col_index = threadIdx.x + blockDim.x * blockIdx.x;
 
    if(row_index >= width || col_index >= width)
        return;
 	int index = row_index * width + col_index;

    int a = 1 - 2 * ((row_index + col_index) & 1);
    output_image[index] *=  a;
}


int read_kernel_supports(Config *config, int2 *kernel_supports)
{
	int total_kernel_samples_needed = 0;

	FILE *kernel_support_file = fopen(config->kernel_support_file,"r");

	if(kernel_support_file == NULL)
	{
		return -1;
	}

	for(int plane_num = 0; plane_num < config->num_wproj_kernels; ++plane_num)
	{
		fscanf(kernel_support_file,"%d\n",&(kernel_supports[plane_num].x));
		kernel_supports[plane_num].y = total_kernel_samples_needed;
		total_kernel_samples_needed += (int)pow((kernel_supports[plane_num].x + 1) * config->oversampling, 2.0);
	}

	fclose(kernel_support_file);
	return total_kernel_samples_needed;
}

__device__ PRECISION2 complex_mult(const PRECISION2 z1, const PRECISION2 z2)
{
	return MAKE_PRECISION2(z1.x * z2.x - z1.y * z2.y, z1.y * z2.x + z1.x * z2.y);
}

__global__ void gridding(PRECISION2 *grid, const PRECISION2 *kernel, const int2 *supports,
	const PRECISION3 *vis_uvw, const PRECISION2 *vis, const int num_vis, const int oversampling,
	const int grid_size, const double uv_scale, const double w_scale)
{
	const unsigned int vis_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(vis_index >= num_vis)
		return;

	// Represents index of w-projection kernel in supports array
	const int plane_index = (int) ROUND(SQRT(ABS(vis_uvw[vis_index].z * w_scale)));

	// Scale visibility uvw into grid coordinate space
	const PRECISION2 grid_coord = MAKE_PRECISION2(
		vis_uvw[vis_index].x * uv_scale,
		vis_uvw[vis_index].y * uv_scale
	);

	const int half_grid_size = grid_size / 2;
	const int half_support = supports[plane_index].x;

	PRECISION conjugate = (vis_uvw[vis_index].z < 0.0) ? -1.0 : 1.0;

	const PRECISION2 snapped_grid_coord = MAKE_PRECISION2(
		ROUND(grid_coord.x * oversampling) / oversampling,
		ROUND(grid_coord.y * oversampling) / oversampling
	);

	const PRECISION2 min_grid_point = MAKE_PRECISION2(
		CEIL(snapped_grid_coord.x - half_support),
		CEIL(snapped_grid_coord.y - half_support)
	);

	const PRECISION2 max_grid_point = MAKE_PRECISION2(
		FLOOR(snapped_grid_coord.x + half_support),
		FLOOR(snapped_grid_coord.y + half_support)
	);

	PRECISION2 grid_point = MAKE_PRECISION2(0.0, 0.0);
	PRECISION2 convolved = MAKE_PRECISION2(0.0, 0.0);
	PRECISION2 kernel_sample = MAKE_PRECISION2(0.0, 0.0);
	int2 kernel_uv_index = make_int2(0, 0);

	int grid_index = 0;
	int kernel_index = 0;
	int w_kernel_offset = supports[plane_index].y;

	for(int grid_v = min_grid_point.y; grid_v <= max_grid_point.y; ++grid_v)
	{	
		kernel_uv_index.y = abs((int)ROUND((grid_v - snapped_grid_coord.y) * oversampling));
		
		for(int grid_u = min_grid_point.x; grid_u <= max_grid_point.x; ++grid_u)
		{
			kernel_uv_index.x = abs((int)ROUND((grid_u - snapped_grid_coord.x) * oversampling));

			kernel_index = w_kernel_offset + kernel_uv_index.y * (half_support + 1)
				* oversampling + kernel_uv_index.x;
			kernel_sample = MAKE_PRECISION2(kernel[kernel_index].x, kernel[kernel_index].y  * conjugate);

			grid_index = (grid_v + half_grid_size) * grid_size + (grid_u + half_grid_size);

			convolved = complex_mult(vis[vis_index], kernel_sample);
			atomicAdd(&(grid[grid_index].x), convolved.x);
			atomicAdd(&(grid[grid_index].y), convolved.y);
		}
	}
}
 
void copy_image_from_gpu(Config *config, PRECISION *d_image, PRECISION *output_image)
{
	printf("UPDATE >>> COPYING GRID BACK TO CPU.... \n");

	CUDA_CHECK_RETURN(cudaMemcpy(output_image, d_image, config->grid_size * config->grid_size * sizeof(PRECISION),
 		cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
}


void copy_complex_image_from_gpu(Config *config, PRECISION2 *d_image, Complex *output_image)
{
	printf("UPDATE >>> COPYING GRID BACK TO CPU.... \n");

	CUDA_CHECK_RETURN(cudaMemcpy(output_image, d_image, config->grid_size * config->grid_size * sizeof(PRECISION2),
 		cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
}


void save_image_to_file(Config *config, PRECISION *grid, int startX, int rangeX, int startY, int rangeY, char *file_path)
{
    FILE *file_real = fopen(file_path, "w");
  
    if(!file_real)
	{	
		printf(">>> ERROR: Unable to create grid files, check file structure exists...\n");
		return;
	}

    for(int row = startY; row < startY+rangeY; ++row)
    {
    	for(int col = startX; col < startX+rangeX; ++col)
        {
            PRECISION grid_point = grid[row * config->grid_size + col];

            #if SINGLE_PRECISION
            	fprintf(file_real, "%f ", grid_point);
            
            #else
           		fprintf(file_real, "%.12f ", grid_point);
            #endif
        }

        fprintf(file_real, "\n");
    }

    fclose(file_real);
}

bool load_image_from_file(PRECISION *image, unsigned int size, char *input_file)
{
	FILE *file = fopen(input_file, "r");

	if(file == NULL)
	{
		printf(">>> ERROR: Unable to load image from file...\n\n");
		return false;
	}

	for(int row = 0; row < size; ++row)
	{
		for(int col = 0; col < size; ++col)
		{
			int image_index = row * size + col;

			#if SINGLE_PRECISION
				fscanf(file, "%f ", &(image[image_index]));
			#else
				fscanf(file, "%lf ", &(image[image_index]));
			#endif
		}
	}

	fclose(file);
	return true;
}

void save_complex_image_to_file(Config *config, Complex *grid, int startX, int rangeX, int startY, int rangeY)
{
    FILE *file_real = fopen(config->output_dirty_image, "w");
    //FILE *file_imag = fopen(config->grid_imag_dest_file, "w");

    if(!file_real )
	{	
		//if(file_real) fclose(file_real);
		//if(file_imag) fclose(file_imag);
		printf(">>> ERROR: Unable to create grid files, check file structure exists...\n");
		return;
	}

    for(int row = startY; row < startY+rangeY; ++row)
    {
    	for(int col = startX; col < startX+rangeX; ++col)
        {
            Complex grid_point = grid[row * config->grid_size + col];

           #if SINGLE_PRECISION
            
            	fprintf(file_real, "%f ", grid_point.real);
            	//fprintf(file_imag, "%f ", grid_point.imag);
            
            #else
            
            	fprintf(file_real, "%.12f ", grid_point.real);
            	//fprintf(file_imag, "%lf ", grid_point.imag);
            #endif
        }

        fprintf(file_real, "\n");
        //fprintf(file_imag, "\n");
    }

    fclose(file_real);
    //fclose(file_imag);
}



bool load_kernel(Config *config, Complex *kernel, int2 *kernel_supports)
{
	FILE *kernel_real_file = fopen(config->kernel_real_source_file, "r");
	FILE *kernel_imag_file = fopen(config->kernel_imag_source_file, "r");
	
	if(!kernel_real_file || !kernel_imag_file)
	{
		if(kernel_real_file) fclose(kernel_real_file);
		if(kernel_imag_file) fclose(kernel_imag_file);
		return false; // unsuccessfully loaded data
	}
	
	int kernel_index = 0;

	for(int plane_num = 0; plane_num < config->num_wproj_kernels; ++plane_num)
	{
		int number_samples_in_kernel = (int) pow((kernel_supports[plane_num].x + 1) * config->oversampling, 2.0);

		for(int sample_number = 0; sample_number < number_samples_in_kernel; ++sample_number)
		{	
			PRECISION real = 0.0;
			PRECISION imag = 0.0; 

            #if SINGLE_PRECISION
				fscanf(kernel_real_file, "%f ", &real);
				fscanf(kernel_imag_file, "%f ", &imag);
            #else
            
				fscanf(kernel_real_file, "%lf ", &real);
				fscanf(kernel_imag_file, "%lf ", &imag);
            #endif
			kernel[kernel_index] = (Complex) {.real = real, .imag = imag};
			kernel_index++;
		}

	}

	fclose(kernel_real_file);
	fclose(kernel_imag_file);
	return true;
}

bool load_visibilities(Config *config, Visibility **vis_uvw, Complex **vis_intensities)
{
	// Attempt to open visibility source file
	FILE *vis_file = fopen(config->visibility_source_file, "r");
	if(vis_file == NULL)
	{
		printf("Unable to open visibility file...\n");
		return false; // unsuccessfully loaded data
	}
	
	// Configure number of visibilities from file
	int num_vis = 0;
	fscanf(vis_file, "%d", &num_vis);
	config->num_visibilities = num_vis;

	// Allocate memory for incoming visibilities
	*vis_uvw = (Visibility*) calloc(num_vis, sizeof(Visibility));
	*vis_intensities = (Complex*) calloc(num_vis, sizeof(Complex));
	if(*vis_uvw == NULL || *vis_intensities == NULL)
	{
		printf("Unable to allocate memory...\n");
		fclose(vis_file);
		return false;
	}
	
	// Load visibility uvw coordinates into memory
	PRECISION vis_u = 0.0;
	PRECISION vis_v = 0.0;
	PRECISION vis_w = 0.0;
	PRECISION vis_real = 0.0;
	PRECISION vis_imag = 0.0;
	PRECISION vis_weight = 0.0;
	PRECISION meters_to_wavelengths = config->frequency_hz / C;

	for(int vis_index = 0; vis_index < num_vis; ++vis_index)
	{
		#if SINGLE_PRECISION
			fscanf(vis_file, "%f %f %f %f %f %f\n", &vis_u, &vis_v,
				&vis_w, &vis_real, &vis_imag, &vis_weight);
		#else
			fscanf(vis_file, "%lf %lf %lf %lf %lf %lf\n", &vis_u, &vis_v,
				&vis_w, &vis_real, &vis_imag, &vis_weight);
		#endif

		(*vis_uvw)[vis_index] = (Visibility) {
			.u = vis_u * meters_to_wavelengths,
			.v = vis_v * meters_to_wavelengths,
			.w = (config->force_zero_w_term) ? (PRECISION)0.0 : vis_w * meters_to_wavelengths 
		};

		if(config->right_ascension)  
		{
			(*vis_uvw)[vis_index].u *= -1.0;
			(*vis_uvw)[vis_index].w *= -1.0;
		}

		(*vis_intensities)[vis_index] = (Complex) {
			.real = vis_real * (PRECISION)1.0,//vis_weight,
			.imag = vis_imag * (PRECISION)1.0//vis_weight
		};
	}

	// Clean up
	fclose(vis_file);
	return true;
}

void allocate_resources(PRECISION **dirty_image, Source **model, PRECISION **psf,
	unsigned int image_size, unsigned int psf_size, unsigned int num_minor_cycles)
{
	*psf = (PRECISION*) calloc(psf_size * psf_size, sizeof(PRECISION));
	*model = (Source*) calloc(num_minor_cycles, sizeof(Source));
}

int performing_deconvolution(Config *config, PRECISION *d_output_image, PRECISION3 *d_sources, PRECISION *psf)
{
	PRECISION3 *d_max_locals;
	PRECISION *d_psf;

	//copy the psf over to GPU
	int psf_size_square = config->psf_size * config->psf_size;
	CUDA_CHECK_RETURN(cudaMalloc(&d_psf, sizeof(PRECISION) * psf_size_square));
	CUDA_CHECK_RETURN(cudaMemcpy(d_psf, psf, sizeof(PRECISION) * psf_size_square, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMalloc(&d_max_locals, sizeof(PRECISION3) * config->grid_size));	

	// row reduction configuration
	int max_threads_per_block = min(config->gpu_max_threads_per_block, config->grid_size);
	int num_blocks = (int) ceil((double) config->grid_size / max_threads_per_block);
	dim3 reduction_blocks(num_blocks, 1, 1);
	dim3 reduction_threads(config->gpu_max_threads_per_block, 1, 1);

	// PSF subtraction configuration
	int max_psf_threads_per_block_dim = min(config->gpu_max_threads_per_block_dimension, config->psf_size);
	int num_blocks_psf = (int) ceil((double) config->psf_size / max_psf_threads_per_block_dim);
	dim3 psf_blocks(num_blocks_psf, num_blocks_psf, 1);
	dim3 psf_threads(max_psf_threads_per_block_dim, max_psf_threads_per_block_dim, 1);

	int cycle_number = 0;
	bool exit_early = false;

	// optional timing start
	cudaEvent_t start, stop;
	if(config->time_deconvolution)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);
	}

	while(cycle_number < config->number_minor_cycles)
	{
		// Find local row maximum via reduction
		find_max_source_row_reduction<<<reduction_blocks, reduction_threads>>>
			(d_output_image, d_max_locals, config->grid_size);
		cudaDeviceSynchronize();

		// Find final image maximum via column reduction (local maximums array)
		find_max_source_col_reduction<<<1, 1>>>
			(d_sources, d_max_locals, cycle_number, config->grid_size, config->loop_gain, 
			 config->weak_source_percent, config->noise_detection_factor);
		cudaDeviceSynchronize();

		subtract_psf_from_residual<<<psf_blocks, psf_threads>>>
				(d_output_image, d_sources, d_psf, cycle_number, config->grid_size, config->psf_size, config->loop_gain);
		cudaDeviceSynchronize();

		compress_sources<<<1, 1>>>(d_sources);
		cudaDeviceSynchronize();

		cudaMemcpyFromSymbol(&exit_early, d_exit_early, sizeof(bool), 0, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		if(exit_early)
		{
			printf(">>> UPDATE: Terminating minor cycles as now just cleaning noise...\n\n");
			break;
		}

		++cycle_number;
	}

	// optional timing end
	if(config->time_deconvolution)
	{
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf(">>> GPU Hogbom completed in %f ms for total cycles %d (average %f ms per cycle)...\n\n", 
			milliseconds, cycle_number, milliseconds / cycle_number);
	}

	// Clean up unneccesary gpu memory
	CUDA_CHECK_RETURN(cudaFree(d_max_locals));
	CUDA_CHECK_RETURN(cudaFree(d_psf));
	cudaDeviceSynchronize();

	int number_of_sources_found = 0;
	cudaMemcpyFromSymbol(&number_of_sources_found, d_source_counter, sizeof(int), 0, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	return number_of_sources_found;
}

__global__ void find_max_source_row_reduction(const PRECISION *residual, PRECISION3 *local_max, const int image_size)
{
	unsigned int row_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(row_index >= image_size)
		return;

	// l, m, intensity 
	// just going to borrow the "m" or y coordinate and use to find the average in this row.
	//PRECISION3 max = MAKE_PRECISION3(0.0, (double) row_index, residual[row_index * image_size]);
	PRECISION3 max = MAKE_PRECISION3(0.0, ABS(residual[row_index * image_size]), residual[row_index * image_size]);
	PRECISION current;

	for(int col_index = 1; col_index < image_size; ++col_index)
	{
		current = residual[row_index * image_size + col_index];
		max.y += ABS(current);
		if(ABS(current) > ABS(max.z))
		{
			// update m and intensity
			max.x = (double) col_index;
			max.z = current;
		}
	}
	
	local_max[row_index] = max;
}

__global__ void find_max_source_col_reduction(PRECISION3 *sources, const PRECISION3 *local_max, const int cycle_number,
	const int image_size, const PRECISION loop_gain, const double weak_source_percent,
	const double noise_detection_factor)
{
	unsigned int col_index = blockIdx.x * blockDim.x + threadIdx.x;

	if(col_index >= 1) // only single threaded
		return;

	//obtain max from row and col and clear the y (row) coordinate.
	PRECISION3 max = local_max[0];
	PRECISION running_avg = local_max[0].y;
	max.y = 0.0;

	PRECISION3 current;
	
	for(int index = 1; index < image_size; ++index)
	{
		current = local_max[index];
		running_avg += current.y;		
		current.y = index;

		if(ABS(current.z) > ABS(max.z))
			max = current;
	}

	running_avg /= (image_size * image_size);
	max.z *= loop_gain;
	
	// determine whether we drop out and ignore this source
	bool extracting_noise = max.z < noise_detection_factor * running_avg * loop_gain;
	bool weak_source = max.z < sources[0].z * weak_source_percent;
	d_exit_early = extracting_noise || weak_source;

	if(d_exit_early)
		return;

	// source was reasonable, so we keep it
	sources[d_source_counter] = max;
	++d_source_counter;
}

__global__ void compress_sources(PRECISION3 *sources)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index >= 1) // only single threaded
		return;

	PRECISION3 last_source = sources[d_source_counter - 1];
	for(int i = d_source_counter - 2; i >= 0; --i)
	{
		if((int)last_source.x == (int)sources[i].x && (int)last_source.y == (int)sources[i].y)
		{
			sources[i].z += last_source.z;
			--d_source_counter;
			break;
		}
	}
}

__global__ void subtract_psf_from_residual(PRECISION *residual, PRECISION3 *sources, const PRECISION *psf, 
	const int cycle_number, const int image_size, const int psf_size, const PRECISION loop_gain)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	// thread out of bounds
	if(idx >= psf_size || idy >= psf_size)
		return;

	const int half_psf_size = psf_size / 2;

	// Determine image coordinates relative to source location
	int2 image_coord = make_int2(
		sources[d_source_counter-1].x - half_psf_size + idx,
		sources[d_source_counter-1].y - half_psf_size + idy
	);
	
	// image coordinates fall out of bounds
	if(image_coord.x < 0 || image_coord.x >= image_size || image_coord.y < 0 || image_coord.y >= image_size)
		return;

	// Get required psf sample for subtraction
	const PRECISION psf_weight = psf[idy * psf_size + idx];

	// Subtract shifted psf sample from residual image
	residual[image_coord.y * image_size + image_coord.x] -= psf_weight  * sources[d_source_counter-1].z;
}

void save_sources_to_file(Source *source, int number_of_sources, char *output_file)
{
	FILE *file = fopen(output_file, "w");

	if(file == NULL)
	{
		printf(">>> ERROR: Unable to save sources to file, moving on...\n\n");
		return;
	}

	fprintf(file, "%d\n", number_of_sources);
	for(int index = 0; index < number_of_sources; ++index)
	{
		#if SINGLE_PRECISION
			fprintf(file, "%f %f %f\n", source[index].l, source[index].m, source[index].intensity);
		#else
			fprintf(file, "%.15f %.15f %.15f\n", source[index].l, source[index].m, source[index].intensity);
		#endif
	}

	fclose(file);
}

void clean_up_gridding_inputs(Complex **grid, Visibility **vis_uvw, Complex **vis_intensities,
	Complex **kernel, int2 **kernel_supports)
{
	if(*grid) 			 free(*grid);
	if(*vis_uvw) 	 	 free(*vis_uvw);
	if(*vis_intensities) free(*vis_intensities);
	if(*kernel) 		 free(*kernel);
	if(*kernel_supports) free(*kernel_supports);
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;

	printf(">>> CUDA ERROR: %s returned %s at %s : %u ",statement, file, cudaGetErrorString(err), line);
	exit(EXIT_FAILURE);
}

void cufft_safe_call(cufftResult err, const char *file, const int line)
{
    if( CUFFT_SUCCESS != err) {
		printf("CUFFT error in file '%s', line %d\nerror %d: %s\nterminating!\n",
			__FILE__, __LINE__, err, cuda_get_error_enum(err));
		cudaDeviceReset();
    }
}

const char* cuda_get_error_enum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}
