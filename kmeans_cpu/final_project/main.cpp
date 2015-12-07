#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <iostream>
#include <typeinfo>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define CLUSTER_NUM       2

using namespace cv;
using namespace std;


// generate a random RGB pixel as centroid
Vec3b randomPixel() {
    return Vec3b(rand()%255, rand()%255, rand()%255);
}

// calculate distance bewteen two Vec3b
float pixelDistance(Vec3b p1, Vec3b p2) {
    float a = (float) (p1.val[0] - p2.val[0]);
    float b = (float) (p1.val[1] - p2.val[1]);
    float c = (float) (p1.val[2] - p2.val[2]);
    return sqrt(a*a + b*b + c*c);
}

// map a given pixel to one of the clusters
unsigned char mapToCluster(Vec3b pixel, Vec3b* centroids) {
    float temp;
    float d = FLT_MAX;
    unsigned char cluster;
    for (unsigned char i = 0; i < CLUSTER_NUM; i++) {
        temp = pixelDistance(pixel, centroids[i]);
        if (temp < d) {
            d = temp;
            cluster = i;
        }
    }
    return cluster;
}

Mat labelToImage(unsigned char** label, Vec3b* centroids, Mat image) {
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            printf("%u", label[i][j]);
            image.at<Vec3b>(i,j) = centroids[label[i][j]];
        }
    }
    return image;
}

int main(int argc, char** argv)
{
    srand(time(NULL)); //reset the random seed for this particular run
    Mat image, image_final;
    image = imread("RivenSquare.jpg", IMREAD_COLOR); // read the image
    if (!image.data) {
        printf("No image data \n");
        return -1;
    }
    image_final.create(image.rows, image.cols, CV_8UC1);

    //generate random pixel
    Vec3b* centroids = (Vec3b*) malloc(CLUSTER_NUM * sizeof(Vec3b));
    for (int i = 0; i < CLUSTER_NUM; i++) {
        centroids[i] = randomPixel();
        printf("%u, ", centroids[i].val[0]);
        printf("%u, ", centroids[i].val[1]);
        printf("%u\n", centroids[i].val[2]);
    }

    //generate a 2D array for labelling
    unsigned char ** label = new unsigned char*[image.rows];
    for (int i = 0; i < image.rows; i++) {
      label[i] = new unsigned char;
    }
    Vec3b pix = image.at<Vec3b>(1,1);
    
    //attemp to map all pixels to cluster
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            label[i][j] = mapToCluster(image.at<Vec3b>(i,j), centroids);
        }
    }

    image_final = labelToImage(label, centroids, image);
    /*
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            printf("%u", label[i][j]);
        }
        printf("\n");;
    }*/


    //Vec3b is vector3 of unsigned char: 0-255
    /*for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            printf( "%u, ", image.at<Vec3b>(i, j).val[0]);
            printf( "%u, ", image.at<Vec3b>(i, j).val[1]);
            printf( "%u\n", image.at<Vec3b>(i, j).val[2]);
        }
    }*/
    

    // splits the image into "BGR" channel
    
    //Mat channel[3];
    //split(image, channel);
    //printf("image cols: %i\n", image.cols);
    //printf("image rows: %i\n", image.rows);
    //printf("channel[0] cols: %i\n", channel[0].cols);
    //printf("channel[0] rows: %i\n", channel[0].rows);
    

    //channel[0]=Mat::zeros(image.rows, image.cols, CV_8UC1);//Set B to 0
    //channel[1]=Mat::zeros(image.rows, image.cols, CV_8UC1);//Set G to 0
    //channel[2]=Mat::zeros(image.rows, image.cols, CV_8UC1);//Set R to 0

    
    //Merging channels into image
    //merge(channel,3,image_final);

    //write to a file (create if doesn't exist)
    imwrite("image_final.jpg",image_final);

    //display image
    //namedWindow("Display Image", WINDOW_AUTOSIZE );
    //imshow("Display Image", image_final);
    //waitKey(0);




    return 0;
}