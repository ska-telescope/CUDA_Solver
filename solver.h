
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

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SOLVER_H_
#define SOLVER_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cufft.h>

	#ifndef SINGLE_PRECISION
		#define SINGLE_PRECISION 1
	#endif

	// Define global precisions
	#ifndef PRECISION
		#if SINGLE_PRECISION
			#define PRECISION float
			#define PRECISION2 float2
			#define PRECISION3 float3
			#define PRECISION4 float4
			#define CUFFT_C2C_PLAN CUFFT_C2C
			#define CUFFT_C2P_PLAN CUFFT_C2R
		#else
			#define PRECISION double
			#define PRECISION2 double2
			#define PRECISION3 double3
			#define PRECISION4 double4
			#define CUFFT_C2C_PLAN CUFFT_Z2Z
			#define CUFFT_C2P_PLAN CUFFT_Z2D
		#endif
	#endif

	// Define function macros
	#if SINGLE_PRECISION
		#define SIN(x) sinf(x)
		#define COS(x) cosf(x)
		#define ABS(x) fabsf(x)
		#define SQRT(x) sqrtf(x)
		#define ROUND(x) roundf(x)
		#define CEIL(x) ceilf(x)
		#define FLOOR(x) floorf(x)
		#define MAKE_PRECISION2(x,y) make_float2(x,y)
		#define MAKE_PRECISION3(x,y,z) make_float3(x,y,z)
		#define MAKE_PRECISION4(x,y,z,w) make_float4(x,y,z,w)
		#define CUFFT_EXECUTE_C2P(a,b,c) cufftExecC2R(a,b,c)
		#define CUFFT_EXECUTE_C2C(a,b,c,d) cufftExecC2C(a,b,c,d)
	#else
		#define SIN(x) sin(x)
		#define COS(x) cos(x)
		#define ABS(x) fabs(x)
		#define SQRT(x) sqrt(x)
		#define ROUND(x) round(x)
		#define FLOOR(x) floor(x)
		#define CEIL(x) ceil(x)
		#define MAKE_PRECISION2(x,y) make_double2(x,y)
		#define MAKE_PRECISION3(x,y,z) make_double3(x,y,z)
		#define MAKE_PRECISION4(x,y,z,w) make_double4(x,y,z,w)
		#define CUFFT_EXECUTE_C2P(a,b,c) cufftExecZ2D(a,b,c)
		#define CUFFT_EXECUTE_C2C(a,b,c,d) cufftExecZ2Z(a,b,c,d)
	#endif

	#define C 299792458.0

	#define CUDA_CHECK_RETURN(value) check_cuda_error_aux(__FILE__,__LINE__, #value, value)

	#define CUFFT_SAFE_CALL(err) cufft_safe_call(err, __FILE__, __LINE__)

	typedef struct Config {
		// full solver configs
		int grid_size;
		double cell_size;
		char *output_dirty_image;
		int gpu_max_threads_per_block;
		int gpu_max_threads_per_block_dimension;
		bool time_gridding;
		bool time_deconvolution;
		bool perform_iFFT_CC;
		bool perform_deconvolution;

		//gridding specific con
		bool right_ascension;
		bool force_zero_w_term;
		double frequency_hz;
		int oversampling;
		double uv_scale;
		int num_visibilities;
		int num_wproj_kernels;
		double max_w;
		double w_scale;
		char *kernel_real_source_file;
		char *kernel_imag_source_file;
		char *kernel_support_file;
		char *visibility_source_file;

		//deconvolution specific
		unsigned int psf_size;
		unsigned int number_minor_cycles;
		double loop_gain;
		char *output_model_sources_file;
		char *psf_source_file;
		char *output_residual_image;
		double weak_source_percent;
		double noise_detection_factor;
	} Config;

	typedef struct Visibility {
		PRECISION u;
		PRECISION v;
		PRECISION w;
	} Visibility;

	typedef struct Complex {
		PRECISION real;
		PRECISION imag;
	} Complex;

	typedef struct Source {
		PRECISION l;
		PRECISION m;
		PRECISION intensity;
	} Source;


	//GLOBAL FUNCTIONS
	void init_config(Config *config);

	void save_image_to_file(Config *config, PRECISION *grid, int startX, int rangeX, int startY, int rangeY, char *file_path);

	bool load_image_from_file(PRECISION *image, unsigned int size, char *input_file);

	void save_complex_image_to_file(Config *config, Complex *grid, int startX, int rangeX,int startY, int rangeY);

	double calc_spheroidal_sample(double nu);

	void save_sources_to_file(Source *source, int number_of_sources, char *output_file);

	void copy_image_from_gpu(Config *config, PRECISION *d_output_image, PRECISION *output_image);

	void copy_complex_image_from_gpu(Config *config, PRECISION2 *d_image, Complex *output_image);

	void check_cuda_error_aux(const char *file, unsigned line, const char *statement, cudaError_t err);

	void cufft_safe_call(cufftResult err, const char *file, const int line);

	const char* cuda_get_error_enum(cufftResult error);


	//GRIDDING SPECIFIC FUNCTIONS
	void clean_up_gridding_inputs(Complex **grid, Visibility **vis_uvw, Complex **vis_intensities,
		Complex **kernel, int2 **kernel_supports);

	void create_1D_half_prolate(PRECISION *prolate, int grid_size);

	int read_kernel_supports(Config *config, int2 *kernel_supports);

	bool load_visibilities(Config *config, Visibility **vis_uvw, Complex **vis_intensities);

	bool load_kernel(Config *config, Complex *kernel, int2 *kernel_supports);

	void execute_gridding(Config *config, Visibility *vis_uvw, 
		Complex *vis_intensities, int num_visibilities, Complex *kernel,
		int2 *kernel_supports, int num_kernel_samples, PRECISION2 *d_input_grid);

	__global__ void gridding(PRECISION2 *grid, const PRECISION2 *kernel, const int2 *supports,
		const PRECISION3 *vis_uvw, const PRECISION2 *vis, const int num_vis, const int oversampling,
		const int grid_size, const double uv_scale, const double w_scale);


	//iFFT and CC FUNCTIONS
	void execute_CC(Config *config, PRECISION *prolate, PRECISION *d_output_image);

	void execute_CUDA_iFFT(Config *config, PRECISION2 *d_input_grid, PRECISION *d_output_image);

	__global__ void fftshift_2D_complex(PRECISION2 *grid, const int width);

	__global__ void fftshift_2D_complex_to_real(PRECISION2 *inputImage, PRECISION *output_image, const int width);

	__global__ void fftshift_2D_real(PRECISION *output_image, const int width);

	__global__ void execute_convolution_correction(PRECISION *grid, const PRECISION *prolate, const int grid_size);

	__device__ PRECISION2 complex_mult(const PRECISION2 z1, const PRECISION2 z2);


	//DECONVOLUTION FUNCTIONS
	int performing_deconvolution(Config *config, PRECISION *d_output_image, PRECISION3 *d_sources, PRECISION *psf);

	__global__ void find_max_source_row_reduction(const PRECISION *residual, PRECISION3 *local_max, const int image_size);

	__global__ void find_max_source_col_reduction(PRECISION3 *sources, const PRECISION3 *local_max, const int cycle_number,
		const int image_size, const PRECISION loop_gain, const double weak_source_percent,
		const double noise_detection_factor);

	__global__ void subtract_psf_from_residual(PRECISION *residual, PRECISION3 *sources, const PRECISION *psf, 
		const int cycle_number, const int image_size, const int psf_size, const PRECISION loop_gain);

	__global__ void compress_sources(PRECISION3 *sources);


#endif /* SOLVER_H */

#ifdef __cplusplus
}
#endif