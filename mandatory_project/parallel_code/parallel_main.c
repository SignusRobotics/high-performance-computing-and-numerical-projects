/* needed header files .... */
/* declarations of functions import_JPEG_file and export_JPEG_file */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#ifdef __MACH__
#include <stdlib.h>
#else
#include <malloc.h>
#endif

// #include "functions.h"

typedef struct Image Image;

struct Image
{
    float **image_data; /* a 2D array of floats */
    int m;              /* # pixels in vertical-direction */
    int n;              /* # pixels in horizontal-direction */
};

//struct Image
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
    int m, c;
    int my_rank, num_procs; //my_m, my_n,
    int my_n, my_m, rest_rows, rest_cols;

    // kappa = 0.2;
    // iters = 100;

    Image u;
    Image u_bar;
    // Image whole_image;
    Image tmp_ptr;
    Image transpose_u;

    int stride, stride_col;
    int rank_up, rank_down;
    int rank_Left, rank_Right;
    int source;
    double start_time, start_time_iso;
    double end_time_iso, time_iso_blokk;

    double end_time, time_iso;

    unsigned char *image_chars,
        *my_image_chars;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    /* read from command line: kappa, iters, input_jpeg_filename, output_jpeg_filename */
    /* ... */

    float kappa;

    // input_jpeg_filename = "mona_lisa_noisy.jpg";
    // output_jpeg_filename = "mona_lisa_image.jpg";
    if (argc != 5)
    {
        if (my_rank == 0)
        {
            printf("4 arguments required\n./ input_jpeg_filename, output_jpeg_file, kappa, iters \nExample usage: mona_lisa_noisy.jpg mona_lisa_denoised.jpg 0.2 100\n");
        }
        return 1;
    }

    int n = atof(argv[3]);
    int iters = atof(argv[4]);
    char txt[255];
    char const *const input_jpeg_filename = argv[1];
    char const *const output_jpeg_filename = argv[2];

    printf("Input filename %s\nOutput filename %s\nKappa %f\niters %d\n", input_jpeg_filename, output_jpeg_filename, kappa, iters);

    double size = num_procs;
    double block_decision = (double)sqrt(size);

    int C = (int)block_decision;
    double B = C - block_decision;

    if (my_rank == 0)
    {
        import_JPEG_file(input_jpeg_filename, &image_chars, &m, &n, &c);
        // allocate_image(&whole_image, m, n);
        printf("m = %d, n = %d \n", m, n);
    }

    // MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&c, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&kappa, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iters, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate displacements and number of rows for each process.
    int *nr_rows = malloc(num_procs * sizeof *nr_rows);
    // antall kolonner i en prosess
    int *nr_cols = malloc(num_procs * sizeof *nr_cols);

    // hvor mange pixler i gitt prosess:
    int *elementsSentMyRank = malloc(num_procs * sizeof *elementsSentMyRank);

    //scatter displacement for rader:
    int *scatterDisplacement = malloc(num_procs * sizeof *scatterDisplacement);

    // for rader:
    int *gatherDisplacement = malloc(num_procs * sizeof *gatherDisplacement);

    // displacement for kolonner:
    int *colDisplacement = malloc(num_procs * sizeof *colDisplacement);

    // antall rader og kolonner pr prosess
    if (num_procs == 1 || B != 0.0)
    {
        my_n = n;
        rest_cols = 0;

        my_m = m / num_procs;
        rest_rows = m % num_procs;
        //printf("rader = %d + rest = %d \n", my_m, rest_rows);
    }
    else
    {
        my_n = n / C;
        rest_cols = n % C;
        // printf("col = %d + rest = %d \n", my_n, rest_cols);

        my_m = m / C;
        rest_rows = m % C;
        //printf("rader = %d + rest = %d \n", my_m, rest_rows);
    }

    // startpunkt for prosess 0.
    // summert foregående elementer fra start fra gitt prosess:
    scatterDisplacement[0] = 0;
    // rader:
    gatherDisplacement[0] = 0;
    // column
    colDisplacement[0] = 0;

    elementsSentMyRank[0] = 0;

    if (num_procs == 1 || B != 0.0)
    //        if (num_procs == 1 || B != 0.0)
    {
        // Definerer antall rader, kolonner, ant elementer i en prosess, start punkt pr prosess.
        for (int procnr = 0; procnr < num_procs - 1; procnr++)
        {
            nr_rows[procnr] = my_m + ((procnr >= (num_procs - rest_rows)) ? 1 : 0);
            nr_cols[procnr] = my_n + ((procnr >= (num_procs - rest_cols)) ? 1 : 0);

            // Displacementstart for row. starter på 0, plusser på antall radene kumulativt.
            gatherDisplacement[procnr + 1] = gatherDisplacement[procnr] + nr_rows[procnr];
        }

        // siste prosess antall rader og kolonner.
        nr_rows[num_procs - 1] = my_m + ((num_procs - 1) >= (num_procs - rest_rows) ? 1 : 0);
        nr_cols[num_procs - 1] = my_n + ((num_procs - 1) >= (num_procs - rest_cols) ? 1 : 0);
    }

    // block_decision **2 = num_procs
    if (B == 0.0)
    {
        int teller_rest = rest_rows;
        int teller_rest_col = rest_cols;
        for (int i = 0; i < C; i++)
        {
            for (int j = 0; j < C; j++)
            {
                // Each row - same number of rows
                if (teller_rest > 0)
                {
                    nr_rows[C * i + j] = my_m + 1;
                }

                else
                {
                    nr_rows[C * i + j] = my_m;
                }

                if (teller_rest_col > 0)
                {
                    nr_cols[C * j + i] = my_n + 1;
                }

                else
                {
                    nr_cols[C * j + i] = my_n;
                }
            }
            teller_rest--;
            teller_rest_col--;
        }
        for (int i = 1; i < C; i++)
        {
            gatherDisplacement[i] = 0;
        }
        for (int procnr = C; procnr < num_procs; procnr++)
        {
            gatherDisplacement[procnr] = gatherDisplacement[procnr - C] + nr_rows[procnr - C];
        }

        // regner ut antall kolonner forflyttet pr prosess: For kvadratiske blokker:
        for (int i = 0; i < C; i++)
        {
            for (int procnr = 0; procnr < C; procnr++)
            {
                if (procnr == 0)
                {
                    colDisplacement[i * C] = 0;
                }

                else
                {
                    colDisplacement[(i * C + procnr)] = colDisplacement[(i * C + procnr) - 1] + nr_cols[(i * C + procnr) - 1];
                }
            }
        }
        int teller = 0;
        // regner ut ny scatterDisplacement for blokkprosess:
        for (int i = 0; i < C; i++)
        {
            for (int j = 0; j < C; j++)
            {
                scatterDisplacement[i * C + j] = teller;
                teller++;
            }
            teller = C * (i + 1);
        }
    }

    /* each process asks process 0 for a partitioned region */
    /* of image_chars and copy the values into u */
    /* ... */

    // Allokerer minne for hver pikselverdier for hver blokk av 1D array.
    if (B == 0)
    {
        my_image_chars = malloc((nr_rows[my_rank] * nr_cols[my_rank]) * sizeof *my_image_chars);
        allocate_image(&u, nr_rows[my_rank], nr_cols[my_rank]);
        allocate_image(&u_bar, nr_rows[my_rank], nr_cols[my_rank]);
    }
    else
    {
        my_image_chars = malloc((nr_rows[my_rank] * nr_cols[my_rank]) * sizeof *my_image_chars);
        allocate_image(&u, nr_rows[my_rank], nr_cols[my_rank]);
        allocate_image(&u_bar, nr_rows[my_rank], nr_cols[my_rank]);
    }

    // venter til alle har fått allokert minne til pikselverdiene og sub-bildene u og u_bar.
    // MPI_Barrier(MPI_COMM_WORLD);

    // forflytning pr rad, viktig ved flere blokker i en vektor.

    stride = 0;

    // Create new datatype for vector:
    // one vector equals one row. and the blocklength is the data that i collected for the given block.
    MPI_Datatype one_row, one_row_block; // one_row_scatter, one_scatter, one_row_recv, one_recv;
    MPI_Type_vector(1,                   // number of blocks
                    nr_cols[my_rank],    // elements in each block
                    stride,              // elements between each block
                    MPI_UNSIGNED_CHAR,   // old datatype
                    &one_row);           // name of new datatype
    MPI_Type_commit(&one_row);

    /*
    // Scattering from m*n array to
    MPI_Type_vector(1,                 // number of blocks
                    nr_cols[my_rank],  // elements in each block
                    0,                 // elements between each block
                    MPI_UNSIGNED_CHAR, // old datatype
                    &one_row_scatter); // name of new datatype

    MPI_Type_create_resized(one_row_scatter, 0, nr_cols[my_rank] * sizeof(unsigned char), &one_scatter); //sizeof(unsigned char)
    MPI_Type_commit(&one_scatter); */

    // Recieve from m*n array to
    /*   MPI_Type_vector(1,                 // number of blocks
                    nr_cols[my_rank],  // elements in each block
                    n,                 // elements between each block
                    MPI_UNSIGNED_CHAR, // old datatype
                    &one_row_recv);    // name of new datatype

    MPI_Type_create_resized(one_row_recv, 0, nr_cols[my_rank], &one_recv);
    MPI_Type_commit(&one_recv); */

    // Finding up and down
    MPI_Datatype one_row_float;
    MPI_Type_vector(1,                // number of blocks
                    nr_cols[my_rank], // elements in each block
                    stride,           // elements between each block
                    MPI_FLOAT,        // old datatype
                    &one_row_float);  // name of new datatype
    MPI_Type_commit(&one_row_float);

    // Finding ghost left and right.
    MPI_Datatype one_col_float;
    MPI_Type_vector(1,                // number of blocks
                    nr_rows[my_rank], // elements in each block
                    0,                // elements between each block
                    MPI_FLOAT,        // old datatype
                    &one_col_float);  // name of new datatype
    MPI_Type_commit(&one_col_float);

    stride = n;

    // Create new datatype for vector:
    // one vector equals nr of rows for given process. and the blocklength is the data that is collected for the given block.
    MPI_Datatype one_block, one_blocks;
    MPI_Type_vector(nr_rows[my_rank],  // number of blocks
                    nr_cols[my_rank],  // elements in each block
                    stride,            // elements between each block
                    MPI_UNSIGNED_CHAR, // old datatype
                    &one_blocks);      // name of new datatype
                                       // MPI_Type_commit(&one_block);

    MPI_Type_create_resized(one_blocks, 0, nr_cols[my_rank], &one_block); //sizeof(unsigned char)
    MPI_Type_commit(&one_block);

    MPI_Datatype block, blocks, test;
    int sizes[2] = {m, n}; //{nr_rows[my_rank], nr_cols[my_rank]};
    int subsizes[2] = {nr_rows[my_rank] - ((C - 1) + rest_rows) / C, nr_cols[my_rank] - ((C - 1) + rest_cols) / C};
    int starts[2] = {gatherDisplacement[0], colDisplacement[0]};

    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_UNSIGNED_CHAR, &block);
    MPI_Type_create_resized(block, 0, nr_cols[my_rank] * sizeof(unsigned char), &blocks);
    MPI_Type_commit(&blocks);

    /* allocating ghost points vector   */
    float *ghost_vector_upper_float = malloc(nr_cols[my_rank] * sizeof *ghost_vector_upper_float);
    float *ghost_vector_lower_float = malloc(nr_cols[my_rank] * sizeof *ghost_vector_lower_float);

    /* allocating ghost points vector   */
    float *ghost_vector_left_float = malloc(nr_rows[my_rank] * sizeof *ghost_vector_left_float);
    float *ghost_vector_right_float = malloc(nr_rows[my_rank] * sizeof *ghost_vector_right_float);

    /* 2D decomposition of the m x n pixels evenly among the MPI processes */
    int *nr_block = malloc(num_procs * sizeof *nr_block);

    for (int i = 0; i < num_procs; i++)
    {
        nr_block[i] = 1;
        elementsSentMyRank[i] = gatherDisplacement[i] * C + colDisplacement[i] / nr_cols[my_rank];
    }
    //MPI_Barrier(MPI_COMM_WORLD);

    if (num_procs == 1 || B != 0.0)
    {
        // fordeler bildets piksler til hver del:
        MPI_Scatterv(image_chars,        // Data scattered
                     nr_rows,            // nr of elements of given datatype sent to buffer
                     gatherDisplacement, // antall rader
                     one_row,            // Datatype
                     my_image_chars,     // subimage recieve buffer
                     nr_rows[my_rank],   // nr of elements of given datatype recieved to buffer in given proces.
                     one_row,            // Datatype
                     0,                  // master process
                     MPI_COMM_WORLD);
    }
    else
    {
        MPI_Scatterv(image_chars,        // Data scattered
                     nr_block,           // nr of elements of given datatype sent to buffer
                     elementsSentMyRank, // Start element
                     blocks,             // Datatype
                     my_image_chars,     // subimage recieve buffer
                     nr_rows[my_rank],   // nr of elements of given datatype recieved to buffer in given proces.
                     one_row,            // Datatype one_row_recv
                     0,                  // master process
                     MPI_COMM_WORLD);
    }

    //MPI_Barrier(MPI_COMM_WORLD);

    // Find ghost points to each process:
    if (num_procs == 1 || B != 0)
    {
        rank_up = coordinate_Up(my_rank, 1, num_procs);
        rank_down = coordinate_Down(my_rank, 1, num_procs);
    }
    convert_jpeg_to_image(my_image_chars, &u);

    int i = 0;
    // MPI_Barrier(MPI_COMM_WORLD);

    // Selve utregning av støyreduksjon:
    if (B != 0 || num_procs == 1)
    {
        start_time = MPI_Wtime();

        while (i < iters)
        {
            // Henter nedre grense ghostvedtor til delbilde over:
            MPI_Sendrecv(u.image_data[0],          // the send array
                         1,                        // antall som skal sendes (1 vektor)
                         one_row_float,            // Datatypen sendt
                         rank_up,                  // hvilken prosess som skal sendes til my_rank + 1, for ned
                         0,                        //tag?
                         ghost_vector_lower_float, // hvor vektoren skal sendes
                         1,                        // antall elementer i en vektor
                         one_row_float,            // Datatypen
                         rank_down,                // source
                         0,                        //recv tag
                         MPI_COMM_WORLD,
                         //&status
                         MPI_STATUS_IGNORE);

            // Henter øvre ghostvedtor til delbilde under:
            MPI_Sendrecv(u.image_data[nr_rows[my_rank] - 1], // the send array
                         1,                                  //ghost_vector_upper_nr[my_rank], //nr_cols[my_rank], //ghost_vector_nr[my_rank], // antall som skal sendes (1 vektor)
                         one_row_float,                      // Datatypen sendt
                         rank_down,                          // hvilken prosess som skal sendes til my_rank + 1, for ned
                         0,                                  //tag?
                         ghost_vector_upper_float,           // ghost_vector_upper_float, // hvor vektoren skal sendes
                         1,                                  //1,                        //ghost_vector_upper_nr[my_rank], // nr_cols[my_rank],          // antall elementer i en vektor
                         one_row_float,                      // Datatypen
                         rank_up,                            //my_rank,                            // source
                         0,                                  //rank_up,                        //recv tag
                         MPI_COMM_WORLD,
                         //&status
                         MPI_STATUS_IGNORE);

            // regner ut for en iterajon
            iso_diffusion_denoising_parallel(&u, &u_bar, kappa, my_rank, ghost_vector_upper_float, ghost_vector_lower_float, num_procs); //, one_row_float, status);

            // swap u og u_bar, slik at u_bar kan regnes på nytt med gamle u_bar.
            if (i < iters - 1)
            {
                tmp_ptr = u;
                u = u_bar;
                u_bar = tmp_ptr;
            }
            i++;
        }
        i++;
    }
    end_time = MPI_Wtime();
    time_iso = end_time - start_time;

    if (my_rank == 0)
        printf("iso_funksjon rad = %f \n", time_iso);

    // for å finne ghostvektor til høyre og venstre
    allocate_image(&transpose_u, nr_cols[my_rank], nr_rows[my_rank]);

    int z = 0;
    if (B == 0)
    {
        rank_up = coordinate_Up(my_rank, C, num_procs);
        rank_down = coordinate_Down(my_rank, C, num_procs);
        rank_Left = coordinate_Left(my_rank, C, num_procs);
        rank_Right = coordinate_Right(my_rank, C, num_procs);
    }
    double start_time_blokk = MPI_Wtime();

    if (B == 0 || num_procs == 1)
    {
        start_time = MPI_Wtime();

        //while (z <= iters)
        for (int t = 0; t < iters; t++)
        {
            // Henter nedre grense ghostvedtor til delbilde over:
            MPI_Sendrecv(u.image_data[0],          // the send array
                         1,                        // antall som skal sendes (1 vektor)
                         one_row_float,            // Datatypen sendt
                         rank_up,                  // hvilken prosess som skal sendes til my_rank + 1, for ned
                         0,                        //tag?
                         ghost_vector_lower_float, // hvor vektoren skal sendes
                         1,                        // antall elementer i en vektor
                         one_row_float,            // Datatypen
                         rank_down,                // source
                         0,                        //recv tag
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE); //&status
                                             // double end_time_iso = MPI_Wtime();

            // Henter øvre ghostvedtor til delbilde under:
            MPI_Sendrecv(u.image_data[nr_rows[u.m - 1]], // the send array
                         1,                              // antall som skal sendes (1 vektor)
                         one_row_float,                  // Datatypen sendt
                         rank_down,                      // hvilken prosess som skal sendes til my_rank + 1, for ned
                         0,                              //tag?
                         ghost_vector_upper_float,       // hvor vektoren skal sendes
                         1,                              // antall elementer i en vektor
                         one_row_float,                  // Datatypen
                         rank_up,                        // source
                         0,                              //recv tag
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE); //&status

            transpose_matrix(&u, &transpose_u);

            // Henter right ghostvedtor til delbilde til left.
            MPI_Sendrecv(transpose_u.image_data[0], // the send array
                         1,                         // antall som skal sendes (1 vektor)
                         one_col_float,             // Datatypen sendt
                         rank_Right,                // hvilken prosess som skal sendes til
                         2,                         //tag?
                         ghost_vector_right_float,  // hvor vektoren skal sendes
                         1,                         // antall elementer i en vektor
                         one_col_float,             // Datatypen
                         rank_Left,                 // source
                         2,                         //recv tag
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE); //&status

            // Henter left ghostvedtor til delbilde til høyre:
            MPI_Sendrecv(transpose_u.image_data[transpose_u.m - 1], // the send array
                         1,                                         // antall som skal sendes (1 vektor)
                         one_col_float,                             // Datatypen sendt
                         rank_Left,                                 // hvilken prosess som skal sendes til my_rank + 1, for ned
                         2,                                         //tag?
                         ghost_vector_left_float,                   // hvor vektoren skal sendes
                         1,                                         // antall elementer i en vektor
                         one_col_float,                             // Datatypen
                         rank_Right,                                //my_rank,  source
                         2,                                         //recv tag
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE); //&status

            // vent tilghostvektor overført.
            // regner ut for en iterajon

            iso_diffusion_denoising_parallel_block(&u, &u_bar, kappa, my_rank, ghost_vector_upper_float, ghost_vector_lower_float, ghost_vector_left_float, ghost_vector_right_float, num_procs, C);

            if (z < iters)
            {
                tmp_ptr = u;
                u = u_bar;
                u_bar = tmp_ptr;
            }
            z++;
        }
        double end_time = MPI_Wtime();

        double time_iso_blokk = end_time - start_time;

        if (my_rank == 0)
            printf("iso_funksjon blokk = %f \n", time_iso_blokk);
    }

    //  double end_time_blokk = MPI_Wtime();

    // double time_iso_blokk = end_time_blokk - start_time_blokk;

    // if (my_rank == 0)
    //     printf("iso_funksjon blokk = %f \n", time_iso_blokk);

    deallocate_image(&transpose_u);
    convert_image_to_jpeg(&u_bar, my_image_chars);

    /* each process sends its resulting content of u_bar to process 0 */
    /* process 0 receives from each process incoming values and */
    /* copy them into the designated region of struct whole_image */
    /* ... */

    if (B != 0 || num_procs == 1)
    {
        MPI_Gatherv(my_image_chars,   //
                    nr_rows[my_rank], //
                    one_row,          //
                    image_chars,
                    nr_rows,
                    gatherDisplacement,
                    one_row, 0,
                    MPI_COMM_WORLD //&status
        );
    }
    else
    {
        MPI_Gatherv(my_image_chars,     // array som skal samles
                    nr_rows[my_rank],   // antall av dataypen som skal samles
                    one_row,            // Datatype til samlingsarray
                    image_chars,        // Mottaker array- samles til.
                    nr_block,           // antall elementer av datatypen
                    elementsSentMyRank, // Hvor i mottaker arrayet dataen skal samles.
                    one_block,          // Datatype samlet
                    0,                  // Root
                    MPI_COMM_WORLD
                    //&status
        );
    }

    printf("gather %d \n", my_rank);

    if (my_rank == 0)
    {
        export_JPEG_file(output_jpeg_filename, image_chars, m, n, c, 75);
        //         deallocate_image(&whole_image);
    }

    deallocate_image(&u);
    deallocate_image(&u_bar);

    free(my_image_chars);
    free(nr_rows);
    free(nr_cols);

    free(ghost_vector_lower_float);
    free(ghost_vector_upper_float);

    free(ghost_vector_right_float);
    free(ghost_vector_left_float);

    free(elementsSentMyRank);
    free(scatterDisplacement);
    free(gatherDisplacement);
    free(colDisplacement);
    MPI_Type_free(&one_row);

    MPI_Type_free(&block);
    MPI_Type_free(&blocks);
    MPI_Type_free(&one_row_float);
    MPI_Type_free(&one_col_float);

    //  MPI_Type_free(&one_row_scatter);
    //  MPI_Type_free(&one_scatter);
    //   MPI_Type_free(&one_row_recv);
    //    MPI_Type_free(&one_recv);

    MPI_Finalize();

    return 0;
}

//struct Image
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