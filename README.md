ASSIGNMENT 1 

#include <mpi.h>
#include <iostream>
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For seeding random number generator

#define MIN_POSITION 0   // Minimum position in the domain
#define MAX_POSITION 100 // Maximum position in the domain
#define NUM_WALKERS 5    // Number of walkers per process
#define MAX_STEPS 20     // Maximum number of steps each walker takes
#define TAG 0            // MPI message tag

using namespace std;

// Structure to store walker data
struct Walker {
    int position;
    int steps_left;
};

int main(int argc, char** argv) {
    int rank, size;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes

    // Seed random generator differently for each process
    srand(time(0) + rank);

    // Define the range each process handles
    int local_min = rank * (MAX_POSITION / size);
    int local_max = (rank + 1) * (MAX_POSITION / size) - 1;

    // Initialize walkers for each process
    Walker walkers[NUM_WALKERS];
    for (int i = 0; i < NUM_WALKERS; i++) {
        walkers[i].position = local_min; // Start from the beginning of local range
        walkers[i].steps_left = (rand() % MAX_STEPS) + 1; // Random step count
    }

    // Process Walkers
    for (int i = 0; i < NUM_WALKERS; i++) {
        while (walkers[i].steps_left > 0) {
            walkers[i].position++;

            // Check if the walker exceeds the local range
            if (walkers[i].position > local_max) {
                if (rank < size - 1) {
                    // Send walker to the next process
                    MPI_Send(&walkers[i], 2, MPI_INT, rank + 1, TAG, MPI_COMM_WORLD);
                    break; // Stop processing this walker
                } else {
                    // Wrap-around to MIN_POSITION if at the last process
                    walkers[i].position = MIN_POSITION;
                }
            }

            walkers[i].steps_left--;
        }
    }

    // Receiving Walkers
    MPI_Status status;
    Walker incoming_walker;
    while (MPI_Recv(&incoming_walker, 2, MPI_INT, MPI_ANY_SOURCE, TAG, MPI_COMM_WORLD, &status) == MPI_SUCCESS) {
        // Continue processing the received walker
        while (incoming_walker.steps_left > 0) {
            incoming_walker.position++;

            if (incoming_walker.position > local_max) {
                if (rank < size - 1) {
                    // Send to the next process
                    MPI_Send(&incoming_walker, 2, MPI_INT, rank + 1, TAG, MPI_COMM_WORLD);
                    break;
                } else {
                    // Wrap-around at last process
                    incoming_walker.position = MIN_POSITION;
                }
            }

            incoming_walker.steps_left--;
        }
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}



ASSIGNMENT 2  

// Estimate the value of Pi using the Monte Carlo method and demonstrate basic MPI functions.
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[]) {
    int rank, size, i;
    long long int num_points = 1000000;
    long long int local_num_points, local_count = 0, global_count = 0;
    double x, y;
    double pi;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_num_points = num_points / size;
    srand(time(NULL) + rank);

    for (i = 0; i < local_num_points; i++) {
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;
        if ((x * x + y * y) <= 1) {
            local_count++;
        }
    }

    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        pi = (4.0 * global_count) / num_points;
        printf("Estimated Pi value: %f\n", pi);
    }

    MPI_Finalize();
    return 0;
} 


// Matrix Multiplication using MPI. Consider 70X70 matrix compute using serial sequential order and compare the time. For computing the time use double start_time, run_time; run_time = omp_get_wtime() - start_time; Time in seconds 
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 70

void matrix_multiply(int rank, int size, double A[SIZE][SIZE], double B[SIZE][SIZE], double *C_partial) {
    int start = rank * (SIZE / size);
    int end = (rank == size - 1) ? SIZE : (rank + 1) * (SIZE / size);

    for (int i = start; i < end; i++) {
        for (int j = 0; j < SIZE; j++) {
            C_partial[(i - start) * SIZE + j] = 0;
            for (int k = 0; k < SIZE; k++) {
                C_partial[(i - start) * SIZE + j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE * SIZE];
    double start_time, run_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure SIZE is greater than number of processes
    if (size > SIZE) {
        if (rank == 0) {
            fprintf(stderr, "Error: Too many MPI processes. Use up to %d processes.\n", SIZE);
        }
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    // Initialize matrices in root process
    if (rank == 0) {
        srand(0);
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                A[i][j] = rand() % 10;
                B[i][j] = rand() % 10;
            }
        }
    }

    // Broadcast matrices
    MPI_Bcast(A, SIZE * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, SIZE * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);  // Synchronize all processes
    start_time = omp_get_wtime(); // Start timing

    // Allocate buffer for partial matrix results
    int local_rows = SIZE / size + (rank == size - 1 ? SIZE % size : 0);
    double *C_partial = (double *)malloc(local_rows * SIZE * sizeof(double));
    if (C_partial == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Perform matrix multiplication
    matrix_multiply(rank, size, A, B, C_partial);

    // Gather results properly
    MPI_Gather(C_partial, local_rows * SIZE, MPI_DOUBLE,
               C, local_rows * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    run_time = omp_get_wtime() - start_time; // Stop timing

    if (rank == 0) {
        printf("Matrix multiplication completed in %lf seconds.\n", run_time);
    }

    free(C_partial); // Free allocated memory
    MPI_Finalize();
    return 0;
} 

// 3. Parallel Sorting using MPI (Odd-Even Sort)
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 16  // Change this for a larger array

// Helper function to swap two elements
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Local Bubble Sort
void bubble_sort(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
            }
        }
    }
}

// Odd-Even Sort using MPI
void parallel_odd_even_sort(int *local_data, int local_n, int rank, int size) {
    int phase;
    int partner;
    for (phase = 0; phase < size; phase++) {
        if (phase % 2 == 0) {
            // Even phase: Even-ranked processes exchange with the next rank
            if (rank % 2 == 0) {
                partner = rank + 1;
            } else {
                partner = rank - 1;
            }
        } else {
            // Odd phase: Odd-ranked processes exchange with the next rank
            if (rank % 2 == 0) {
                partner = rank - 1;
            } else {
                partner = rank + 1;
            }
        }

        if (partner >= 0 && partner < size) {
            int recv_data[local_n];
            MPI_Sendrecv(local_data, local_n, MPI_INT, partner, 0,
                         recv_data, local_n, MPI_INT, partner, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Merge local_data with received data
            int merged[2 * local_n];
            for (int i = 0; i < local_n; i++) merged[i] = local_data[i];
            for (int i = 0; i < local_n; i++) merged[local_n + i] = recv_data[i];

            bubble_sort(merged, 2 * local_n);

            // Keep the first half or second half based on rank order
            if (rank < partner) {
                for (int i = 0; i < local_n; i++) local_data[i] = merged[i];
            } else {
                for (int i = 0; i < local_n; i++) local_data[i] = merged[local_n + i];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int *global_data = NULL;
    int local_n;
    int *local_data;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_n = SIZE / size;  // Number of elements per process

    // Root initializes the array
    if (rank == 0) {
        global_data = (int *)malloc(SIZE * sizeof(int));
        srand(0);
        for (int i = 0; i < SIZE; i++) {
            global_data[i] = rand() % 100;
        }
        printf("Unsorted Array: ");
        for (int i = 0; i < SIZE; i++) {
            printf("%d ", global_data[i]);
        }
        printf("\n");
    }

    // Allocate local array
    local_data = (int *)malloc(local_n * sizeof(int));

    // Scatter data from root to all processes
    MPI_Scatter(global_data, local_n, MPI_INT, local_data, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Sort locally
    bubble_sort(local_data, local_n);

    // Perform Parallel Odd-Even Sort
    parallel_odd_even_sort(local_data, local_n, rank, size);

    // Gather sorted data
    MPI_Gather(local_data, local_n, MPI_INT, global_data, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    // Root prints final sorted array
    if (rank == 0) {
        printf("Sorted Array: ");
        for (int i = 0; i < SIZE; i++) {
            printf("%d ", global_data[i]);
        }
        printf("\n");
        free(global_data);
    }

    free(local_data);
    MPI_Finalize();
    return 0;
} 

// 4. Heat Distribution Simulation using MPI
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 20         // Grid size (N x N)
#define MAX_ITER 500 // Maximum iterations
#define EPSILON 0.01 // Convergence criteria

// Initialize grid with boundary conditions
void initialize_grid(double grid[N][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            grid[i][j] = 0.0; // Default temperature
        }
    }
    // Set boundary conditions
    for (int i = 0; i < N; i++)
    {
        grid[0][i] = 100.0;     // Top boundary (hot)
        grid[N - 1][i] = 100.0; // Bottom boundary (hot)
    }
}

// Perform one iteration of heat distribution
void compute_heat_distribution(double local_grid[][N], double new_local_grid[][N], int start, int end)
{
    for (int i = start; i < end; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            new_local_grid[i][j] = 0.25 * (local_grid[i - 1][j] + local_grid[i + 1][j] +
                                           local_grid[i][j - 1] + local_grid[i][j + 1]);
        }
    }
}

int main(int argc, char *argv[])
{
    int rank, size;
    double grid[N][N], new_grid[N][N];
    int local_start, local_end;
    double max_diff, global_diff;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = N / size; // Number of rows each process handles

    // Define local computation range (each process gets a section of rows)
    local_start = rank * rows_per_proc;
    local_end = (rank + 1) * rows_per_proc;

    if (rank == 0)
    {
        initialize_grid(grid); // Only rank 0 initializes the grid
    }

    // Broadcast the initialized grid to all processes
    MPI_Bcast(grid, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int iter = 0;
    do
    {
        max_diff = 0.0;

        // Exchange boundary rows with neighboring processes
        if (rank > 0)
        { // Send to above rank
            MPI_Sendrecv(&grid[local_start][0], N, MPI_DOUBLE, rank - 1, 0,
                         &grid[local_start - 1][0], N, MPI_DOUBLE, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1)
        { // Send to below rank
            MPI_Sendrecv(&grid[local_end - 1][0], N, MPI_DOUBLE, rank + 1, 0,
                         &grid[local_end][0], N, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Compute new temperature values
        compute_heat_distribution(grid, new_grid, local_start, local_end);

        // Compute the max temperature difference for convergence check
        for (int i = local_start; i < local_end; i++)
        {
            for (int j = 1; j < N - 1; j++)
            {
                double diff = fabs(new_grid[i][j] - grid[i][j]);
                if (diff > max_diff)
                {
                    max_diff = diff;
                }
                grid[i][j] = new_grid[i][j]; // Update grid
            }
        }

        // Find the maximum difference across all processes
        MPI_Allreduce(&max_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        iter++;
    } while (global_diff > EPSILON && iter < MAX_ITER);

    // Gather final grid at root process
    // Allocate a separate receive buffer in root process
    double final_grid[N][N];

    MPI_Gather(&grid[local_start][0], rows_per_proc * N, MPI_DOUBLE,
               &final_grid[0][0], rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Use final_grid in root process
    if (rank == 0)
    {
        printf("Final Heat Distribution:\n");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                printf("%6.2f ", final_grid[i][j]);
            }
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
} 

// 5. Parallel Reduction using MPI
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 100  // Size of the array

int main(int argc, char *argv[]) {
    int rank, size;
    double local_sum = 0.0, global_sum = 0.0;
    double data[ARRAY_SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int elements_per_proc = ARRAY_SIZE / size;  // Divide array among processes

    // Root process initializes the array
    if (rank == 0) {
        srand(0);
        for (int i = 0; i < ARRAY_SIZE; i++) {
            data[i] = rand() % 10;  // Random numbers between 0-9
        }
    }

    // Scatter the array to all processes
    double local_data[elements_per_proc];
    MPI_Scatter(data, elements_per_proc, MPI_DOUBLE,
                local_data, elements_per_proc, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Each process computes its local sum
    for (int i = 0; i < elements_per_proc; i++) {
        local_sum += local_data[i];
    }

    // Perform reduction to sum up all local sums into global_sum at root
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root process prints the final result
    if (rank == 0) {
        printf("Total sum of array elements: %lf\n", global_sum);
    }

    MPI_Finalize();
    return 0;
} 

// 6. Parallel Dot Product using MPI
// dot_product=A0*B0+A1*B1+...+An*Bn​

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define VECTOR_SIZE 100  // Size of vectors

int main(int argc, char *argv[]) {
    int rank, size;
    double local_dot = 0.0, global_dot = 0.0;
    double A[VECTOR_SIZE], B[VECTOR_SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int elements_per_proc = VECTOR_SIZE / size;  // Divide work among processes

    // Root process initializes vectors
    if (rank == 0) {
        srand(0);
        for (int i = 0; i < VECTOR_SIZE; i++) {
            A[i] = rand() % 10;
            B[i] = rand() % 10;
        }
    }

    // Scatter vectors to all processes
    double local_A[elements_per_proc], local_B[elements_per_proc];
    MPI_Scatter(A, elements_per_proc, MPI_DOUBLE, local_A, elements_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, elements_per_proc, MPI_DOUBLE, local_B, elements_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Each process computes its local dot product
    for (int i = 0; i < elements_per_proc; i++) {
        local_dot += local_A[i] * local_B[i];
    }

    // Reduce all local dot products to get the final result at rank 0
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 prints the final dot product
    if (rank == 0) {
        printf("Dot Product: %lf\n", global_dot);
    }

    MPI_Finalize();
    return 0;
} 

// Parallel Prefix Sum (Scan) using MPI
// prefix_sum[i]=A[0]+A[1]+...+A[i]
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 8  // Number of elements

int main(int argc, char *argv[]) {
    int rank, size;
    int local_value, prefix_sum;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize local values (each process gets one value)
    local_value = rank + 1; // Example: 1, 2, 3, ..., size

    // Perform parallel prefix sum using MPI_Scan
    MPI_Scan(&local_value, &prefix_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    printf("Process %d: Local Value = %d, Prefix Sum = %d\n", rank, local_value, prefix_sum);

    MPI_Finalize();
    return 0;
} 

// Parallel Matrix Transposition using MPI
// B[j][i]=A[i][j]
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 6  // Define matrix size (change as needed)

void print_matrix(double matrix[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            printf("%5.1f ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int rank, size;
    double A[SIZE][SIZE], B[SIZE][SIZE];
    double local_A[SIZE][SIZE], local_B[SIZE][SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = SIZE / size; // Ensure SIZE is divisible by size

    // Root process initializes the matrix
    if (rank == 0) {
        printf("Original Matrix:\n");
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                A[i][j] = i * SIZE + j + 1;  // Example initialization
            }
        }
        print_matrix(A);
    }

    // Scatter rows to all processes
    MPI_Scatter(A, rows_per_proc * SIZE, MPI_DOUBLE, local_A, rows_per_proc * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform local transposition
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < SIZE; j++) {
            local_B[j][i + rank * rows_per_proc] = local_A[i][j]; // Transpose local part
        }
    }

    // Gather transposed parts
    MPI_Gather(local_B, rows_per_proc * SIZE, MPI_DOUBLE, B, rows_per_proc * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Root prints the transposed matrix
    if (rank == 0) {
        printf("Transposed Matrix:\n");
        print_matrix(B);
    }

    MPI_Finalize();
    return 0;
} 

ASSIGNMENT 3 

/*
Q1. DAXPY Loop Using MPI
Formula: 𝑋[𝑖]=𝑎×𝑋[𝑖]+𝑌[𝑖]
X and 𝑌 are vectors of size 2^16
a is a scalar.
Compare MPI implementation speedup with a serial version.
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N (1 << 16)  // 2^16 elements

void daxpy_serial(double a, double *X, double *Y) {
    for (int i = 0; i < N; i++) {
        X[i] = a * X[i] + Y[i];
    }
}

void daxpy_parallel(double a, double *X, double *Y, int rank, int size) {
    int local_n = N / size;  // Divide elements among processes
    int start = rank * local_n;
    int end = start + local_n;

    for (int i = start; i < end; i++) {
        X[i] = a * X[i] + Y[i];
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    double *X, *Y;
    double a = 2.5;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate vectors
    X = (double *)malloc(N * sizeof(double));
    Y = (double *)malloc(N * sizeof(double));

    // Initialize X and Y in root process
    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            X[i] = rand() % 10;
            Y[i] = rand() % 10;
        }
    }

    // Broadcast X and Y to all processes
    MPI_Bcast(X, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(Y, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    start_time = MPI_Wtime();
    daxpy_parallel(a, X, Y, rank, size);
    end_time = MPI_Wtime();

    // Gather results at rank 0
    MPI_Gather(X + rank * (N / size), N / size, MPI_DOUBLE, X, N / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("MPI DAXPY Time: %lf seconds\n", end_time - start_time);
    }

    free(X);
    free(Y);
    MPI_Finalize();
    return 0;
} 

/*
Q2. Approximation of π using MPI_Bcast & MPI_Reduce
Broadcast num_steps to all processes.
Each process computes a partial sum.
MPI_Reduce aggregates results.
*/
#include <mpi.h>
#include <stdio.h>

#define NUM_STEPS 100000

int main(int argc, char *argv[]) {
    int rank, size, i;
    double step, local_sum = 0.0, pi = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    step = 1.0 / (double)NUM_STEPS;
    MPI_Bcast(&step, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (i = rank; i < NUM_STEPS; i += size) {
        double x = (i + 0.5) * step;
        local_sum += 4.0 / (1.0 + x * x);
    }

    MPI_Reduce(&local_sum, &pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        pi = step * pi;
        printf("Approximated π: %lf\n", pi);
    }

    MPI_Finalize();
    return 0;
} 

/*
Q3. Parallel Prime Number Finding Using MPI_Send & MPI_Recv
Master sends numbers to test.
Workers check for primality and return results.
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_NUM 100  // Find primes up to this number

int is_prime(int n) {
    if (n < 2) return 0;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int rank, size, num, flag;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {  // Master
        for (num = 2; num <= MAX_NUM; num++) {
            int worker;
            MPI_Recv(&worker, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&num, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
        }
        
        // Send termination signal (-1)
        for (int i = 1; i < size; i++) {
            MPI_Recv(&num, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&num, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        printf("Prime finding complete.\n");

    } else {  // Workers
        while (1) {
            MPI_Send(&rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Recv(&num, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (num == -1) break;

            flag = is_prime(num) ? num : -num;
            MPI_Send(&flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
