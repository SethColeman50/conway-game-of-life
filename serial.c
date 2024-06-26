// Compile with (laptop): g++-13 -Wall serial.c -o serial -lm
// Compile with (mucluster): g++ -Wall serial.c -o serial -lm
// Run with: ./serial <input_file> <output_file> <num_steps>

#include "matrix.hpp"
#include "helpers.c"
#include <time.h>
#include <stdio.h>

void update_cell(size_t row, size_t col, Matrix<double>& input, Matrix<double>& output) {
    // checking if we are at an edge or corner
    bool is_top_boundary = row == 0;
    bool is_bottom_boundary = row == input.rows-1;
    bool is_left_boundary = col == 0;
    bool is_right_boundary = col == input.cols-1;

    // get the number of neighbors of the cell at (row, col) while making sure we don't go out of bounds
    double num_of_neighbors = (is_top_boundary ? 0 : input(row-1, col)) +  /* top */               \
        (is_bottom_boundary ? 0 : input(row+1, col)) + /* bottom */                                \
        (is_left_boundary ? 0 : input(row, col-1)) + /* left */                                    \
        (is_right_boundary ? 0 : input(row, col+1)) + /* right */                                  \
        ((is_top_boundary || is_left_boundary) ? 0 : input(row-1, col-1)) + /* top left */         \
        ((is_top_boundary || is_right_boundary) ? 0 : input(row-1, col+1)) + /* top right */       \
        ((is_bottom_boundary || is_left_boundary) ? 0 : input(row+1, col-1)) + /* bottom left */   \
        ((is_bottom_boundary || is_right_boundary) ? 0 : input(row+1, col+1)); /* bottom right */  \

    // get the value of the cell at (row, col)
    double value = input(row, col);

    // conway's game of life rules
    if (value == 1) {
        if (num_of_neighbors <= 1 || num_of_neighbors >= 4) {
            output(row,col) = 0;
        } else {
            output(row,col) = 1;
        }
    } else {
        if (num_of_neighbors == 3) {
            output(row,col) = 1;
        } else {
            output(row,col) = 0;
        }
    }
}



int main(int argc, const char* argv[]) {
    // parse arguments
    Matrix<double> input = Matrix<double>::from_csv(argv[1]);
    const char* output_file = argv[2];
    size_t rows = input.rows;
    size_t cols = input.cols;
    size_t num_steps = atoi(argv[3]);
    
    // start the timer
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    Matrix<double> output;

    for (size_t t = 0; t < num_steps; t++) {
        // resets output matrix from last iteration
        output = Matrix<double>(rows, cols).fill_zeros();

        // updates all cells in the matrix
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                update_cell(i, j, input, output);
            }
        }
        
        // sets the input matrix to the output matrix for next iteration
        input = output;
    }

    // get the elapsed time
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time = end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)/1000000000.0;
    printf("Time: %g secs\n", time);


    print_matrix(output);

    // save the output matrix
    output.to_csv(output_file);

    return 0;
}