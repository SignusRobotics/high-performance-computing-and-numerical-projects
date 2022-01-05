#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

#include "functions.h"

void allocate_image(Image *u, int m, int n);

void convert_jpeg_to_image(unsigned char *image_chars, Image *u);
void convert_image_to_jpeg(Image *u, unsigned char *image_chars);

void deallocate_image(Image *u);
int coordinate_Down(int my_rank, int blockparameter, int num_procs);
int coordinate_Up(int my_rank, int blockparameter, int num_procs);
int coordinate_Left(int my_rank, int blockparameter, int num_procs);
int coordinate_Right(int my_rank, int blockparameter, int num_procs);

void transpose_matrix(Image *u, Image *transpose_u);

void iso_diffusion_denoising_parallel(Image *u, Image *u_bar, float kappa, int process, float *ghost_vector_upper_float, float *ghost_vector_lower_float, int num_procs); //, MPI_Datatype one_row_float, MPI_Status status)
void iso_diffusion_denoising_parallel_block(Image *u, Image *u_bar, float kappa, int process, float *ghost_vector_upper_float, float *ghost_vector_lower_float, float *ghost_vector_left_float, float *ghost_vector_right_float, int num_procs, int blockparameter);

void allocate_image(Image *u, int m, int n)
{
    float **arr = malloc(m * sizeof *arr);
    arr[0] = malloc(m * n * sizeof *arr[0]);

    for (int i = 1; i < m; i++)
    {
        arr[i] = &(arr[0][i * n]);
    }

    Image im = {arr, m, n};
    *u = im;
}

void convert_jpeg_to_image(unsigned char *image_chars, Image *u)
{
    //int teller = 0;
    for (int i = 0; i < u->m; i++)
    {
        for (int j = 0; j < u->n; j++)
        {
            // u->image_data[i][j] = (float)image_chars[teller * u->n + j];
            u->image_data[i][j] = (float)image_chars[i * u->n + j];
        }
        //teller++;
    }
}

void convert_image_to_jpeg(Image *u, unsigned char *image_chars)
{
    //int teller = 0;
    for (int i = 0; i < u->m; i++)
    {
        for (int j = 0; j < u->n; j++)
        {
            //image_chars[teller * u->n + j] = (unsigned char)u->image_data[i][j];
            image_chars[i * u->n + j] = (unsigned char)u->image_data[i][j];
        }
        //teller++;
    }
}

void deallocate_image(Image *u)
{
    free(u->image_data[0]); // Must be deallocated first to avoid memory leaks.
    free(u->image_data);
}

void iso_diffusion_denoising_parallel(Image *u, Image *u_bar, float kappa, int process, float *ghost_vector_upper_float, float *ghost_vector_lower_float, int num_procs)
{

    // Initialize u_bar:
    for (int row = 0; row < u->m; row++)
    {
        for (int col = 0; col < u->n; col++)
        {
            if (row == 0 || col == 0 || row == u->m - 1 || col == u->n)
            {
                u_bar->image_data[row][col] = u->image_data[row][col];
            }
            /*else
            {
                u_bar->image_data[row][col] = 0.0;
            }*/
        }
    }

    /*
    for (int row = 0; row < u->m; row++)
    {
        for (int col = 0; col < u->n; col++)
        {
            u_bar->image_data[row][col] = u->image_data[row][col];
        }
    }*/

    if (process > 0)
    {
        // regner ut den første linja for prosesser > 0:
        for (int col = 1; col < u->n - 1; col++)
        {
            u_bar->image_data[0][col] =
                u->image_data[0][col] +
                kappa *
                    (ghost_vector_upper_float[col] +
                     u->image_data[0][col - 1] -
                     4 *
                         u->image_data[0][col] +
                     u->image_data[0][col + 1] +
                     u->image_data[1][col]);
        }
    }
    // The middel is the same:
    for (int row = 1; row < u->m - 1; row++)
    {
        for (int col = 1; col < u->n - 1; col++)
        {
            u_bar->image_data[row][col] =
                u->image_data[row][col] +
                kappa *
                    (u->image_data[row - 1][col] +
                     u->image_data[row][col - 1] -
                     4 *
                         u->image_data[row][col] +
                     u->image_data[row][col + 1] +
                     u->image_data[row + 1][col]);
        }
    }

    if (process < num_procs - 1)
    {
        // printf("my_rank = %d, m = %d \n", process, u->m);

        // the last line is calculated if the process < number of processes -1 :
        for (int col = 1; col < u->n - 1; col++)
        {
            u_bar->image_data[u->m - 1][col] =
                u->image_data[u->m - 1][col] +
                kappa *
                    (u->image_data[u->m - 2][col] +
                     u->image_data[u->m - 1][col - 1] -
                     4 *
                         u->image_data[u->m - 1][col] +
                     u->image_data[u->m - 1][col + 1] +
                     ghost_vector_lower_float[col]);
        }
    }
}

int coordinate_Up(int my_rank, int blockparameter, int num_procs)
{
    int rank_up;

    // if (my_rank == 0)
    if (my_rank < blockparameter) // num_procs - blockparameter)
    {
        rank_up = MPI_PROC_NULL;
        return rank_up;
    }
    else
    {
        rank_up = my_rank - blockparameter;
        return rank_up;
    }
}

int coordinate_Down(int my_rank, int blockparameter, int num_procs)
{
    int rank_down;

    //    if (my_rank == num_procs - 1 || my_rank > num_procs - blockparameter)
    if (my_rank >= num_procs - blockparameter)

    {
        rank_down = MPI_PROC_NULL;
        return rank_down;
    }
    else
    {
        rank_down = my_rank + blockparameter;
        return rank_down;
    }
}

int coordinate_Left(int my_rank, int blockparameter, int num_procs)
{
    int rank_Left;

    //    if (my_rank == num_procs - 1 || my_rank > num_procs - blockparameter)
    //  if (my_rank >= num_procs - blockparameter)

    if ((my_rank + 1) % blockparameter == 0)
    {
        rank_Left = MPI_PROC_NULL;
        return rank_Left;
    }
    else
    {
        rank_Left = my_rank + 1;

        return rank_Left;
    }
}

int coordinate_Right(int my_rank, int blockparameter, int num_procs)
{
    int rank_Right;

    //    if (my_rank == num_procs - 1 || my_rank > num_procs - blockparameter)
    //  if (my_rank >= num_procs - blockparameter)
    if ((my_rank) % blockparameter == 0)
    {
        rank_Right = MPI_PROC_NULL;
        return rank_Right;
    }
    else
    {
        rank_Right = my_rank - 1;
        return rank_Right;
    }
}

void transpose_matrix(Image *u, Image *transpose_u)
{
    for (int row = 0; row < transpose_u->m; row++)
    {
        for (int col = 0; col < transpose_u->n; col++)
        {
            transpose_u->image_data[row][col] = u->image_data[col][row];
            //printf("isotest \n");
        }
    }
}

void iso_diffusion_denoising_parallel_block(Image *u, Image *u_bar, float kappa, int process, float *ghost_vector_upper_float, float *ghost_vector_lower_float, float *ghost_vector_left_float, float *ghost_vector_right_float, int num_procs, int blockparameter)
{
    // Initialize u_bar:
    for (int row = 0; row < u->m; row++)
    {
        for (int col = 0; col < u->n; col++)
        {
            u_bar->image_data[row][col] = u->image_data[row][col];
        }
    }

    /*   for (int row = 0; row < u->m; row++)
    {
        for (int col = 0; col < u->n; col++)
        {
            if (row == 0 || col == 0 || row == u->m - 1 || col == u->n)
            {
                u_bar->image_data[row][col] = u->image_data[row][col];
            }
            /*else
            {
                u_bar->image_data[row][col] = 0.0;
            }*/
    //      }
    //   }

    if ((process) % blockparameter != 0)
    {

        // Use left ghost vector,and find first column
        // column zero
        for (int row = 1; row < u->m - 1; row++)
        {
            u_bar->image_data[row][0] =
                u->image_data[row][0] +
                kappa *
                    (u->image_data[row - 1][0] +
                     ghost_vector_left_float[row] +
                     -4 *
                         u->image_data[row][0] +
                     u->image_data[row][1] +
                     u->image_data[row + 1][0]);
        }
    }

    if (process < blockparameter)
    {

        // regner ut den første linja for prosesser > blockparameter:
        for (int col = 1; col < u->n - 1; col++)
        {
            u_bar->image_data[0][col] =
                u->image_data[0][col] +
                kappa *
                    (ghost_vector_upper_float[col] + //ghost_u->image_data[0][col] +
                     u->image_data[0][col - 1] -
                     4 *
                         u->image_data[0][col] +
                     u->image_data[0][col + 1] +
                     u->image_data[1][col]);
        }
    }
    // The middel is the same:
    for (int row = 1; row < u->m - 1; row++)
    {
        for (int col = 1; col < u->n - 1; col++)
        {
            u_bar->image_data[row][col] =
                u->image_data[row][col] +
                kappa *
                    (u->image_data[row - 1][col] +
                     u->image_data[row][col - 1] -
                     4 *
                         u->image_data[row][col] +
                     u->image_data[row][col + 1] +
                     u->image_data[row + 1][col]);
        }
    }

    if ((process + 1) % blockparameter != 0)
    {
        // Calculate the last column with ghost vector right and find last column. from prosess my_rank + 1:
        for (int row = 1; row < u->m - 1; row++)
        {
            u_bar->image_data[row][u->n - 1] =
                u->image_data[row][u->n - 1] +
                kappa *
                    (u->image_data[row - 1][u->n - 1] +
                     u->image_data[row][u->n - 2] -
                     4 *
                         u->image_data[row][u->n - 1] +
                     ghost_vector_right_float[row] +
                     u->image_data[row + 1][u->n - 1]);
        }
    }

    if (process < num_procs - blockparameter)
    {
        // the last line is calculated if the process < number of processes - blockparameter:
        for (int col = 1; col < u->n - 1; col++)
        {
            u_bar->image_data[u->m - 1][col] =
                u->image_data[u->m - 1][col] +
                kappa *
                    (u->image_data[u->m - 2][col] +
                     u->image_data[u->m - 1][col - 1] -
                     4 *
                         u->image_data[u->m - 1][col] +
                     u->image_data[u->m - 1][col + 1] +
                     ghost_vector_lower_float[col]);
        }
    }
}