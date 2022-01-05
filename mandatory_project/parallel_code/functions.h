#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

typedef struct Image Image;

struct Image
{
    float **image_data;
    int m;
    int n;
};

#endif