
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

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "solver.h"


int main(int argc, char **argv)
{
	Config config;
	init_config(&config);

	printf(">>> UPDATE: Determining memory requirements for convolution kernels...\n");
	int2 *kernel_supports = (int2*) calloc(config.num_wproj_kernels, sizeof(int2));
    if(kernel_supports == NULL)
    {
    	printf(">>> ERROR: unable to allocate memory for kernel supports, terminating...\n");
		clean_up_gridding_inputs(NULL, NULL, NULL, NULL, &kernel_supports);
		return EXIT_FAILURE;
    }

	int total_samples_needed = read_kernel_supports(&config, kernel_supports);
	if(total_samples_needed <= 0)
	{
		printf(">>> ERROR: unable to read kernel samples from file, terminating...\n");
		clean_up_gridding_inputs(NULL, NULL, NULL, NULL, &kernel_supports);
		return EXIT_FAILURE;
	}
	printf(">>> UPDATE: Requirements analysis complete...\n");

	printf(">>> UPDATE: Allocating resources for grid and convolution kernels...\n");
	Complex *grid = (Complex*) calloc(config.grid_size * config.grid_size, sizeof(Complex));
	Complex *kernel = (Complex*) calloc(total_samples_needed, sizeof(Complex));
	if(grid == NULL || kernel == NULL)
	{
		printf(">>> ERROR: unable to allocate memory for grid/kernels, terminating...\n");
		clean_up_gridding_inputs(&grid, NULL, NULL, &kernel, &kernel_supports);
		return EXIT_FAILURE;
	}
	printf(">>> UPDATE: Resource allocation successful...\n");
	
	printf(">>> UPDATE: Loading kernels...\n");
	bool loaded_kernel = load_kernel(&config, kernel, kernel_supports);

	if(!loaded_kernel)
	{
		printf(">>> ERROR: Unable to open kernel source files, terminating...\n");
		clean_up_gridding_inputs(&grid, NULL, NULL, &kernel, &kernel_supports);
		return EXIT_FAILURE;
	}
	printf(">>> UPDATE: Loading kernels complete...\n");

	printf(">>> UPDATE: Loading visibilities...\n");
	Visibility *vis_uvw = NULL;
	Complex *vis_intensities = NULL;
	bool loaded_vis = load_visibilities(&config, &vis_uvw, &vis_intensities);

	if(!loaded_vis || !vis_uvw)
	{	printf(">>> ERROR: Unable to load grid or read visibility files, terminating...\n");
		clean_up_gridding_inputs(&grid, &vis_uvw, &vis_intensities, &kernel, &kernel_supports);
		return EXIT_FAILURE;
	}
	printf(">>> UPDATE: Loading visibilities complete...\n");
	
	printf(">>> UPDATE: LOADING IN CONVOLUTION CORRECTION PROLATE SAMPLES... \n");
	PRECISION *prolate = (PRECISION*) calloc(config.grid_size / 2, sizeof(PRECISION));
	create_1D_half_prolate(prolate, config.grid_size);
	if(!prolate)
	{
		printf("ERROR: Unable to allocate memory for the 1D prolate spheroidal \n");
	    clean_up_gridding_inputs(&grid, &vis_uvw, &vis_intensities, &kernel, &kernel_supports);
	    free(prolate);
	    return EXIT_FAILURE;
	}	


	//This pointers to the output residual image and the output model sources
	//PRECISION *residual_image = NULL;
	Source *model_sources = NULL;

	//need to allocate device 
	PRECISION2 *d_input_grid;

	int grid_size_square = config.grid_size * config.grid_size;
	cudaMalloc((void**)&d_input_grid, sizeof(PRECISION2) * grid_size_square);
	cudaDeviceSynchronize();


	printf(">>> UPDATE: Performing W-Projection based convolutional gridding...\n");

	execute_gridding(&config, vis_uvw, vis_intensities, config.num_visibilities,
		kernel, kernel_supports, total_samples_needed, d_input_grid);

	printf(">>> UPDATE: Gridding complete...\n");

	//clean up the inputs used for gridding - optional as later when doing major cycles we will take this out
	

//	clean_up_gridding_inputs(&grid, &vis_uvw, &vis_intensities, &kernel, &kernel_supports);



	// Complex *residual_image = (Complex*) calloc(config.grid_size*config.grid_size, sizeof(Complex));
	// //final cleanup
	// printf("COPY IMAGE FROM GPU ");
	// copy_complex_image_from_gpu(&config, d_input_grid, residual_image);
	// // printf(">>> UPDATE: Saving grid to file...\n");
	// printf("SAVE IMAGE TO FILE ");
	// save_complex_image_to_file(&config, residual_image, 0,config.grid_size,0,config.grid_size);





	//execute FFT subroutine, now the output is all REAL
	PRECISION *d_output_image;
	cudaMalloc((void**)&d_output_image, sizeof(PRECISION) * grid_size_square);
	cudaDeviceSynchronize();

	execute_CUDA_iFFT(&config, d_input_grid, d_output_image);

	printf("UPDATE >>> Executing Convolution Correction ...\n");
	execute_CC(&config, prolate, d_output_image);


	// Complex *residual_image = (Complex*) calloc(config.grid_size*config.grid_size, sizeof(Complex));
	// //final cleanup
	// printf("COPY IMAGE FROM GPU ");
	// copy_complex_image_from_gpu(&config, d_input_grid, residual_image);
	// // printf(">>> UPDATE: Saving grid to file...\n");
	// printf("SAVE IMAGE TO FILE ");
	// save_complex_image_to_file(&config, residual_image, 0,config.grid_size,0,config.grid_size);

	//return EXIT_SUCCESS;



	PRECISION *residual_image = (PRECISION*) calloc(config.grid_size*config.grid_size, sizeof(PRECISION));
	copy_image_from_gpu(&config, d_output_image, residual_image);
	save_image_to_file(&config, residual_image, 0,config.grid_size,0,config.grid_size);



	// clean_up_gridding_inputs(&grid, &vis_uvw, &vis_intensities, &kernel, &kernel_supports);


	
	//execute the CC subroutine
	//printf("EXECUTING CC");
	//execute_CC(&config, prolate, d_output_image);
	
//	free(prolate);
	//execute the deconvolution routine


	//optional obtain residual image


	//model sources

	
	//final cleanup
	//printf("COPY IMAGE FROM GPU ");
	//copy_image_from_gpu(&config, d_output_image, residual_image);
	// printf(">>> UPDATE: Saving grid to file...\n");
	//printf("SAVE IMAGE TO FILE ");
	//save_image_to_file(&config, residual_image, 0,config.grid_size,0,config.grid_size);

	// printf(">>> UPDATE: Save successful...\n");
	
	// printf(">>> UPDATE: Cleaning up allocated resources...\n");
	// clean_up(&grid, &vis_uvw, &vis_intensities, &kernel, &kernel_supports, &prolate);
	// printf(">>> UPDATE: Cleaning complete, exiting...\n");

	return EXIT_SUCCESS;
}