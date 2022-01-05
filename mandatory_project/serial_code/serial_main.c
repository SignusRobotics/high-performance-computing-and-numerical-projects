#include <stdio.h>

#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef __MACH__
#include <stdlib.h>
#else
#include <malloc.h>

#endif

//#include "functions.h"

typedef struct Image Image;

struct Image
{
    float **image_data;
    int m;
    int n;
};

//struct Image
void allocate_image(Image *u, int m, int n);

void convert_jpeg_to_image(unsigned char *image_chars, Image *u);
void convert_image_to_jpeg(Image *u, unsigned char *image_chars);

void deallocate_image(Image *u);

void iso_diffusion_denoising(Image *u, Image *u_bar, float kappa, int iters);
void reset_u(Image *u, Image *u_bar);

/* The purpose of this program is to demonstrate how the functions
   'import_JPEG_file' & 'export_JPEG_file' can be used. */

void import_JPEG_file(const char *filename, unsigned char **image_chars,
                      int *image_height, int *image_width,
                      int *num_components);
void export_JPEG_file(const char *filename, const unsigned char *image_chars,
                      int image_height, int image_width,
                      int num_components, int quality);

int main(int argc, char *argv[])
{
    //int height, width, comp, i, j, k;
    //unsigned char *image_chars, *new_chars;

    /* read from command line: kappa, iters, input_jpeg_filename, output_jpeg_filename */
    /* ... */
    //struct Image image;
    int m, n, c, iters;
    float kappa;
    Image u;
    Image u_bar;
    double start, end;
    unsigned char *image_chars;
    char *input_jpeg_filename, *output_jpeg_filename;

    kappa = 0.2;
    iters = 1000;

    input_jpeg_filename = "mona_lisa_noisy.jpg";
    output_jpeg_filename = "mona_lisa_image.jpg";

    int height, width, comp, i, j, k;
    //unsigned char *new_chars;

    printf("test \n");

    //import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
    import_JPEG_file("mona_lisa_noisy.jpg", &image_chars, &m, &n, &c);

    //printf("%u \n",image_chars[0]);

    allocate_image(&u, m, n);

    allocate_image(&u_bar, m, n);

    convert_jpeg_to_image(image_chars, &u);
    start = clock();
    iso_diffusion_denoising(&u, &u_bar, kappa, iters);
    end = clock();
    printf("time iso seriell %f \n", (double)(end - start) / CLOCKS_PER_SEC);
    convert_image_to_jpeg(&u_bar, image_chars);
    //export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);
    export_JPEG_file("mona_lisa_image.jpg", image_chars, m, n, c, 75);

    deallocate_image(&u);
    deallocate_image(&u_bar);

    return 0;
}

//struct Image
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
    //printf("test2");

    //return u;
}

void convert_jpeg_to_image(unsigned char *image_chars, Image *u)
{
    int teller = 0;
    for (int i = 0; i < u->m; i++)
    {
        for (int j = 0; j < u->n; j++)
        {
            //u->image_data[i][j] = (float)image_chars[teller * u->n + j];
            u->image_data[i][j] = (float)image_chars[i * u->n + j];
        }
        //teller++;
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

void iso_diffusion_denoising(Image *u, Image *u_bar, float kappa, int iters)
{
    printf("start");
    Image *tmp_ptr;

    //tmp_ptr = *u;

    /*
    **(u_bar->image_data) = **(u->image_data); */
    //u_bar = u;

    //u = tmp_ptr;

    //u_bar->image_data = u->image_data

    for (int row = 0; row < u->m; row++)
    {
        for (int col = 0; col < u->n; col++)
        {
            u_bar->image_data[row][col] = u->image_data[row][col];
        }
    }

    /*  for (int row = 0; row < u->m; row++)
    {
        for (int col = 0; col < u->n; col++)
        {
            if (row == 0 || col == 0)
            {
                u_bar->image_data[row][col] = u->image_data[row][col];
            }
            
        }
    } */

    int start = 1; //100;

    for (int i = 0; i < iters; i++)
    {
        for (int row = start; row < u->m - start; row++)
        {
            for (int col = start; col < u->n - start; col++)
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
            u_bar = tmp_ptr;

            //reset_u(u, u_bar);
        }
    }
    printf("iso");
    //deallocate_image(&tmp_ptr);
}

void reset_u(Image *u, Image *u_bar)
{
    for (int row = 0; row < u->m; row++)
    {
        for (int col = 0; col < u->n; col++)
        {
            u->image_data[row][col] = u_bar->image_data[row][col];
        }
    }
}