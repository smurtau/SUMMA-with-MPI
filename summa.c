/******************************************************************************************
*
*	Filename:	summa.c
*	Purpose:	A paritally implemented program for MSCS6060 HW. Students will complete 
*			the program by adding SUMMA implementation for matrix multiplication C = A * B.  
*	Assumptions:    A, B, and C are square matrices n by n; 
*			the total number of processors (np) is a square number (q^2).
*	To compile, use 
*	    mpicc -o summa summa.c
*       To run, use
*	    mpiexec -n $(NPROCS) ./summa
*********************************************************************************************/

#include <stdio.h>
#include <time.h>	
#include <stdlib.h>	
#include <math.h>	
#include "mpi.h"

#define min(a, b) ((a < b) ? a : b)
#define SZ 4000		//Each matrix of entire A, B, and C is SZ by SZ. Set a small value for testing, and set a large value for collecting experimental data.



void matMulAdd(double **c, double **a, double **b, int block_sz){
	int ii;
	int myrank;
	int p;

	MPI_Status status;
 	int tag;
	int i,j,k;
        int n;


        n = block_sz; 

	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	/* Data distribution */ 

	if( myrank != 0 ) {
		MPI_Recv( &a[0][0], n*n/p, MPI_INT, 0, tag, MPI_COMM_WORLD, &status );
		MPI_Recv( &b[0][0], n*n, MPI_INT, 0, tag, MPI_COMM_WORLD, &status );
	} else {
		for( i=1; i<p; i++ ) {
			for (ii = 0; ii < n/p; ii++)
				for(j = 0; j < n; j++)
				{
					a[ii][j] = i * n/p + ii + j + 1;
				}
//                      printf("sending a to %d\n", i);
			MPI_Send( &a[0][0], n*n/p, MPI_INT, i, tag, MPI_COMM_WORLD );
			MPI_Send( &b[0][0], n*n, MPI_INT, i, tag, MPI_COMM_WORLD );
		}
	}

//	printf("populate root's a\n");
	if (myrank == 0){
		for (ii = 0; ii < n/p; ii++)
			for(j = 0; j < n; j++)
			{
				a[ii][j] = ii + j + 1;
			}
	}

	/* Computation */ 

        printf("calculating...\n");
	for ( i=0; i<n/p; i++) 
		for (j=0; j<n; j++) {
			c[i][j]=0;
			for (k=0; k<n; k++)
				c[i][j] += a[i][k] * b[k][j];
 		}  

	/* Result gathering */ 

	if (myrank != 0)
		MPI_Send( &c[0][0], n*n/p, MPI_INT, 0, tag, MPI_COMM_WORLD);
	else
  		for (i=1; i<p; i++){
			MPI_Recv( &c[0][0], n*n/p, MPI_INT, i, tag, MPI_COMM_WORLD, &status);

        	}
	MPI_Finalize();
}




/**
*   Allocate space for a two-dimensional array
*/
double **alloc_2d_double(int n_rows, int n_cols) {
	int i;
	double **array;
	array = (double **)malloc(n_rows * sizeof (double *));
        array[0] = (double *) malloc(n_rows * n_cols * sizeof(double));
        for (i=1; i<n_rows; i++){
                array[i] = array[0] + i * n_cols;
        }
        return array;
}

/**
*	Initialize arrays A and B with random numbers, and array C with zeros. 
*	Each array is setup as a square block of blck_sz.
**/
void initialize(double **lA, double **lB, double **lC, int blck_sz){
	int i, j;
	double value;
	// Set random values...technically it is already random and this is redundant
	for (i=0; i<blck_sz; i++){
		for (j=0; j<blck_sz; j++){
			lA[i][j] = (double)rand() / (double)RAND_MAX;
			lB[i][j] = (double)rand() / (double)RAND_MAX;
			lC[i][j] = 0.0;
		}
	}
}


/**
*	Perform the SUMMA matrix multiplication. 
*       Follow the pseudo code in lecture slides.
*/
void matmul(int my_rank, int proc_grid_sz, int block_sz, double **my_A,
						double **my_B, double **my_C){

	//Add your implementation of SUMMA algorithm
	double **buffA, **buffB;
	buffA = alloc_2d_double(block_sz,block_sz);
	buffB = alloc_2d_double(block_sz,block_sz);

	MPI_Comm grid_comm,row_comm,col_comm;
	int grid_rank, q;
	int dimsizes[2];
	int wraparound[2];
	int coordinates[2];
	int free_coords[2];
	int reorder = 1;

	q = (int)sqrt((double)block_sz);
	dimsizes[0] = dimsizes[1] = q;
	wraparound[0] = wraparound[1] = 1;

	MPI_Cart_create(MPI_COMM_WORLD, 2, dimsizes, wraparound, reorder, &grid_comm);
	MPI_Comm_rank(grid_comm, &my_rank);
	MPI_Cart_coords(grid_comm, my_rank, 2, coordinates);
	MPI_Cart_rank(grid_comm, coordinates, &grid_rank);
	
	free_coords[0] = 0;
	free_coords[1] = 1;
	MPI_Cart_sub(grid_comm,free_coords, &row_comm);

	free_coords[0] = 1;
	free_coords[1] = 0;
	MPI_Cart_sub(grid_comm, free_coords, &col_comm);

	for (int k = 0; k < proc_grid_sz; k++){
		if (coordinates[1] == k) {
			for(int i = 0; i < block_sz; i++){
				for (int j = 0; j < block_sz; j++){
					buffA[i][j] = my_A[i][j];
				}
			}
		}
		MPI_Bcast(*buffA, block_sz*block_sz,MPI_DOUBLE,k,row_comm);

		if (coordinates[0] == k) {
			for(int i = 0; i < block_sz; i++){
				for (int j = 0; j < block_sz; j++){
					buffB[i][j] = my_B[i][j];
				}
			}
		}
		MPI_Bcast(*buffB, block_sz*block_sz,MPI_DOUBLE,k,col_comm);

		if(coordinates[0] == k && coordinates[1] == k){
			matmulAdd(my_C,my_A,my_B,block_sz);
		}
		else if (coordinates[0] == k) {
			matmulAdd(my_C,buffA,my_B,block_sz);
		}
		else if (coordinates[1] == k){
			matmulAdd(my_C,my_A,buffB,block_sz);
		} 
		else
			matmulAdd(my_C,buffA,buffB,block_sz);
	}
}


int main(int argc, char *argv[]) {
	int rank, num_proc;							//process rank and total number of processes
	double start_time, end_time, total_time;	// for timing
	int block_sz;								// Block size length for each processor to handle
	int proc_grid_sz;							// 'q' from the slides


	
	srand(time(NULL));							// Seed random numbers

/* insert MPI functions to 1) start process, 2) get total number of processors and 3) process rank*/

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&num_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


/* assign values to 1) proc_grid_sz and 2) block_sz*/
	
	//...

	if (SZ % proc_grid_sz != 0){
		printf("Matrix size cannot be evenly split amongst resources!\n");
		printf("Quitting....\n");
		exit(-1);
	}

	// Create the local matrices on each process

	double **A, **B, **C;
	A = alloc_2d_double(block_sz, block_sz);
	B = alloc_2d_double(block_sz, block_sz);
	C = alloc_2d_double(block_sz, block_sz);

	
	initialize(A, B, C, block_sz);

	// Use MPI_Wtime to get the starting time
	start_time = MPI_Wtime();


	// Use SUMMA algorithm to calculate product C
	matmul(rank, proc_grid_sz, block_sz, A, B, C);


	// Use MPI_Wtime to get the finishing time
	end_time = MPI_Wtime();


	// Obtain the elapsed time and assign it to total_time
	total_time = end_time - start_time;

	// Insert statements for testing
	//...


	if (rank == 0){
		// Print in pseudo csv format for easier results compilation
		printf("squareMatrixSideLength,%d,numMPICopies,%d,walltime,%lf\n",
			SZ, num_proc, total_time);
	}

	// Destroy MPI processes

	MPI_Finalize();

	return 0;
}
