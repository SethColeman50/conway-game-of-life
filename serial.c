// Compile with: g++ -Wall serial.c matrix.c -o serial -lm

#include "matrix.hpp"
#include <time.h>
#include <stdio.h>

void update_cell(size_t row, size_t col, Matrix<double>& input, Matrix<double>& output) {
    size_t rows = input.rows;
    size_t cols = input.cols;

    bool is_top_boundary = row == 0;
    bool is_bottom_boundary = row == rows-1;
    bool is_left_boundary = col == 0;
    bool is_right_boundary = col == cols-1;

    // Get the value of the cell at (row, col)
    double value = input(row, col);

    // Get the value of the cell at (row-1, col)
    double value_up = is_top_boundary ? 0 : input(row-1, col);

    double value_up_left = (is_top_boundary || is_left_boundary) ? 0 : input(row-1, col-1);

    double value_up_right = (is_top_boundary || is_right_boundary) ? 0 : input(row-1, col+1);

    // Get the value of the cell at (row+1, col)
    double value_down = (is_bottom_boundary) ? 0 : input(row+1, col);

    double value_down_left = (is_bottom_boundary || is_left_boundary) ? 0 : input (row+1, col-1);

    double value_down_right = (is_bottom_boundary || is_right_boundary) ? 0 : input(row+1, col+1);

    // Get the value of the cell at (row, col-1)
    double value_left = (is_left_boundary) ? 0 : input(row, col-1);

    // Get the value of the cell at (row, col+1)
    double value_right = (is_right_boundary) ? 0 : input(row, col+1);

    // Update the value of the cell at (row, col)'
    double num_of_neighbors = value_up + value_down + value_left + value_right + value_up_left + value_up_right + value_down_left + value_down_right;

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

void print_matrix(Matrix<double>& matrix) {
    for (size_t i = 0; i < matrix.rows; i++) {
        for (size_t j = 0; j < matrix.cols; j++) {
            printf("%g ", matrix(i, j));
        }
        printf("\n");
    }
}


int main(int argc, const char* argv[]) {
    // parse arguments
    Matrix<double> input = Matrix<double>::from_csv(argv[1]);
    size_t rows = input.rows;
    size_t cols = input.cols;
    size_t num_steps = atoi(argv[2]);
    
    // start the timer
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    Matrix<double> output;

    for (size_t t = 1; t < num_steps; t++) {
        output = Matrix<double>(rows, cols).fill_zeros();
        print_matrix(input);

        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                // Update the value of the cell at (i, j)
                // update_cell(i, j, input, output);
                printf("Updating cell (%zu, %zu) at time %zu\n", i, j, t); 
            }
        }
        input = output;
    }

    // get the elapsed time
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time = end.tv_sec-start.tv_sec+(end.tv_nsec-start.tv_nsec)/1000000000.0;
    printf("Time: %g secs\n", time);

    // print_matrix(output);

    // save the output matrix
    output.to_csv("output.csv");

    return 0;
}