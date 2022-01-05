#ifndef FUNCTIONS_H_
# define FUNCTIONS_H_

typedef struct Image Image;

struct Image
{
    float **image_data; /* a 2D array of floats */
    int m;              /* # pixels in vertical-direction */
    int n;              /* # pixels in horizontal-direction */
};

/*typedef struct
{
float** image_data; 
int m; 
int n; 
}
image;*/

#endif 


