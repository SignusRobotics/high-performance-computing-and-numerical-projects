#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h> 
#include <string.h>

#include "functions.h"



void allocate_image(Image *u, int m, int n);

void convert_jpeg_to_image(unsigned char *image_chars, Image *u);
void convert_image_to_jpeg(Image *u, unsigned char *image_chars);

void deallocate_image(Image *u);

void iso_diffusion_denoising(Image *u, Image *u_bar, float kappa, int iters);
//void reset_u(Image *u, Image *u_bar);



void allocate_image(Image *u, int m, int n)
{

    // 1D array
    // Image u = {u, m, n};

    float **arr = malloc(m * sizeof *arr);
    arr[0] = malloc(m * n * sizeof *arr[0]);

    for (int i = 1; i < m; i++)
    {
        arr[i] = &(arr[0][i * n]);
    }

    Image im = {arr, m, n};
    *u = im;
    printf("test2");

    //return u;

    /*
    int height, width, comp, i,j,k;
    unsigned char *new_chars;
    
 // creating a horizontally flipped image 
  new_chars = (unsigned char*)malloc(height*width*comp*sizeof(unsigned char));
  for (i=0; i<height; i++)
    for (j=0; j<width; j++)
      for (k=0; k<comp; k++)
	new_chars[i*(width*comp)+j*comp+k]=image_chars[i*(width*comp)+(width-j-1)*comp+k];
*/
}

void convert_jpeg_to_image(unsigned char *image_chars, Image *u)
{
    int teller = 0;
    for (int i = 0; i < u->m; i++)
    {
        for (int j = 0; j < u->n; j++)
        {
            u->image_data[i][j] = (float)image_chars[teller * u->n + j];
        }
        teller++;
    }
}

void convert_image_to_jpeg(Image *u, unsigned char *image_chars)
{
    int teller = 0;
    for (int i = 0; i < u->m; i++)
    {
        for (int j = 0; j < u->n; j++)
        {
            //u->image_data[i][j] = (float)image_chars[teller*u->n + j];
            image_chars[teller * u->n + j] = (unsigned char)u->image_data[i][j];
        }
        teller++;
    }
}

void deallocate_image(Image *u)
{
    free(u->image_data[0]); // Must be deallocated first to avoid memory leaks.
    free(u->image_data);
}

void iso_diffusion_denoising(Image *u, Image *u_bar, float kappa, int iters)
{

    //u_bar->image_data = u->image_data;
    Image *tmp_ptr; 
    
    //u_bar->image_data = u->image_data


    for (int row = 0; row < u->m; row++)
    {
        for (int col = 0; col < u->n; col++)
        {
            if (row == 0 || col == 0)
            {
                u_bar->image_data[row][col] = u->image_data[row][col];
            }
            else
            {
                u_bar->image_data[row][col] = 0.0;
            }
        }
    } 

    for (int i = 0; i < iters; i++)
    {
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
        if (i != iters - 1)
        {
            // swap the pointers
            tmp_ptr = u; 
            u = u_bar; 
            u_bar = u; 
            
            //reset_u(u, u_bar);
        }
    }
}