/*
1) Place K points into the space represented by the objects that are being clustered. These points represent initial group centroids.
2) Assign each object to the group that has the closest centroid.
3) When all objects have been assigned, recalculate the positions of the K centroids.
4) Repeat Steps 2 and 3 until the centroids no longer move. This produces a separation of the objects into groups from which the metric to be minimized can be calculated.
*/

/* Takes in an array of pointers, where each pointer points to an 
array of char R, G and B values*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} Pixel;

Pixel *makeCentroid(unsigned char r, unsigned char g, unsigned char b){
    Pixel *pixel = (Pixel *) malloc (sizeof (Pixel));
    pixel->r = r;
    pixel->g = g;
    pixel->b = b;
    return pixel;
}

Pixel *randomizeCentroid(){
    unsigned char r, g, b;
    time_t t;
    int n = 3;
    
    /* Initializes random number generator */
    srand((unsigned) time(&t));
    r = rand() % 255;
    g = rand() % 255;
    b = rand() % 255;

    Pixel *pixel = (Pixel *) malloc (sizeof (Pixel));
    pixel->r = r;
    pixel->g = g;
    pixel->b = b;
    return pixel;
}

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Ptr<IplImage> img = cvLoadImage(argv[1]);
    int LENGTH = img->height*img->width;

    int numc = 4; /* Centroids */

    Pixel centroids[numc];
    for (int i = 0; i<numc; i++){
        centroids[i] = *randomizeCentroid();
    }

    printf("%d\n",centroids[0].r);
    printf("%d\n",centroids[0].g);
    printf("%d\n\n",centroids[0].b);

    int iterations = 0;

    /*
    printf("%d\n",img->height); //720
    printf("%d\n",img->width); //1080
    */

    Pixel rgb_array[LENGTH];

    for (int y = 0; y < img->height; y++) {
        for(int x = 0; x < img->width; x++) {
            uchar blue = ((uchar*)(img->imageData + img->widthStep*y))[x*3];
            uchar green = ((uchar*)(img->imageData + img->widthStep*y))[x*3+1];
            uchar red = ((uchar*)(img->imageData + img->widthStep*y))[x*3+2];
            rgb_array[y*img->height + x] = *makeCentroid(blue,green,red);
        }
    }

    /*
    printf("%d\n",rgb_array[100].r);
    printf("%d\n",rgb_array[100].g);
    printf("%d\n",rgb_array[100].b);
    */

    for (int i = 0; i<LENGTH; i++){
        rgb_array[0]
        for (int j=0; j<numc; j++){

        }

        /*printf("%d,%d,%d\n",rgb_array[i].r,rgb_array[i].g,rgb_array[i].b);*/
    }

    cvShowImage("image", img);

    cvWaitKey(0);
    /*free();*/
    return 0;
}