// Compile with (on gpu session on cluster): nvcc -O3 -arch=sm_86 cuda.cu -o cuda
// Run with: ./cuda <input_file> <output_file> <block_size_x> <block_size_y> <num_steps>

#include "matrix.hpp"
#include "helpers.c"
#include <time.h>
#include <stdio.h>

#include <cuda_runtime.h>

/**
 * Macro to check if a CUDA call has an error, and if it does, report it and
 * exit the program.
 */
#define CHECK(call)                                                       \
{                                                                         \
   const cudaError_t error = call;                                        \
   if (error != cudaSuccess)                                              \
   {                                                                      \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
      printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));  \
      exit(1);                                                            \
   }                                                                      \
}

__global__ void update_cell_kernel(size_t rows, size_t cols, double* input, double* output) {
    size_t row = blockIdx.x*blockDim.x + threadIdx.x;
    size_t col = blockIdx.y*blockDim.y + threadIdx.y;

    // check if we are out of bounds
    if (row >= rows || col >= cols) {
        return;
    }

    // checking if we are at an edge or corner
    bool is_top_boundary = row == 0;
    bool is_bottom_boundary = row == rows-1;
    bool is_left_boundary = col == 0;
    bool is_right_boundary = col == cols-1;

    // get the number of neighbors of the cell at (row, col) while making sure we don't go out of bounds
    double num_of_neighbors = (is_top_boundary ? 0 : input[(row-1) * cols + col]) +  /* top */               \
        (is_bottom_boundary ? 0 : input[(row+1) * cols + col]) + /* bottom */                                \
        (is_left_boundary ? 0 : input[row * cols + (col-1)]) + /* left */                                    \
        (is_right_boundary ? 0 : input[row * cols + (col+1)]) + /* right */                                  \
        ((is_top_boundary || is_left_boundary) ? 0 : input[(row-1) * cols + (col-1)]) + /* top left */         \
        ((is_top_boundary || is_right_boundary) ? 0 : input[(row-1) * cols + (col+1)]) + /* top right */       \
        ((is_bottom_boundary || is_left_boundary) ? 0 : input[(row+1) * cols + (col-1)]) + /* bottom left */   \
        ((is_bottom_boundary || is_right_boundary) ? 0 : input[(row+1) * cols + (col+1)]); /* bottom right */  \

    // get the value of the cell at (row, col)
    double value = input[row * cols + col];

    // conway's game of life rules
    if (value == 1) {
        if (num_of_neighbors <= 1 || num_of_neighbors >= 4) {
            output[row * cols + col] = 0;
        } else {
           output[row * cols + col] = 1;
        }
    } else {
        if (num_of_neighbors == 3) {
            output[row * cols + col] = 1;
        } else {
            output[row * cols + col] = 0;
        }
    }
}

int main(int argc, const char* argv[]) {
    if (argc != 6) {
        printf("Usage: %s <input_file> <output_file> <block_size_x> <block_size_y> <num_steps>\n", argv[0]);
        return -1;
    }
    // parse arguments
    Matrix<double> input = Matrix<double>::from_csv(argv[1]);
    const char* output_file = argv[2];
    size_t rows = input.rows;
    size_t cols = input.cols;
    size_t block_size_x = atoi(argv[3]);
    size_t block_size_y = atoi(argv[4]);
    size_t num_steps = atoi(argv[5]);
    
    // start the timer
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    Matrix<double> output = Matrix<double>(rows, cols).fill_zeros();

    // Allocate GPU memory
    double* d_input;
    double* d_output;
    CHECK(cudaMalloc(&d_input, rows*cols*sizeof(double)));
    CHECK(cudaMalloc(&d_output, rows*cols*sizeof(double)));

    // Copy input matrix to GPU
    CHECK(cudaMemcpy(d_input, input.data, rows*cols*sizeof(double), cudaMemcpyHostToDevice));

    // Calculate grid and block sizes
    dim3 block_size(block_size_x, block_size_y);
    dim3 grid_size((rows+block_size.x-1)/block_size.x, (cols+block_size.y-1)/block_size.y);

    // Launch kernel
    for (size_t t = 0; t < num_steps; t++) {

        // updates all cells in the matrix
        update_cell_kernel<<<grid_size, block_size>>>(rows, cols, d_input, d_output);
    
        // Check for any errors launching the kernel
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error launching update_cell_kernel: %s\n", cudaGetErrorString(err));
            return -1;
        }

        // Wait for GPU to finish before accessing on host
        cudaDeviceSynchronize();

        // Swap input and output matrices
        d_input = d_output;
    }

    // Copy output matrix from GPU
    // Note: d_input now points to the output matrix
    CHECK(cudaMemcpy(output.data, d_input, rows*cols*sizeof(double), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CHECK(cudaFree(d_input));
    if (d_output != d_input) {
        CHECK(cudaFree(d_output));
    }

    // get the elapsed time
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time = end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)/1000000000.0;
    printf("Time: %g secs\n", time);

    print_matrix(output);

    // save the output matrix
    output.to_csv(output_file);

    // reset the device
    CHECK(cudaDeviceReset());

    return 0;
}