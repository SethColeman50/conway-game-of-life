// helper file for the conway's game of life project

#include "matrix.hpp"
#include <stdio.h>

void print_matrix(Matrix<double>& matrix) {
    if (matrix.rows > 20 || matrix.cols > 20) {
        printf("Matrix too large to print\n");
        return;
    }
    for (size_t i = 0 ; i < matrix.cols+2; i++) {
        printf("\u001b[48;5;10m \u001b[48;5;10m \u001b[0m");
    }
    printf("\n");
    for (size_t i = 0; i < matrix.rows; i++) {
        printf("\u001b[48;5;10m \u001b[48;5;10m \u001b[0m");
        for (size_t j = 0; j < matrix.cols; j++) {
            if (matrix(i,j)) {
                printf("\u001b[48;5;240m \u001b[48;5;240m \u001b[0m");
            } else {
                // printf("\u001b[47;1m  \u001b[0m");
                printf("\u001b[48;5;231m \u001b[48;5;231m \u001b[0m");
            }
        }
        printf("\u001b[48;5;10m \u001b[48;5;10m \u001b[0m");
        printf("\n");
    }
    for (size_t i = 0 ; i < matrix.cols+2; i++) {
        printf("\u001b[48;5;10m  \u001b[0m");
    }
    printf("\n");
}