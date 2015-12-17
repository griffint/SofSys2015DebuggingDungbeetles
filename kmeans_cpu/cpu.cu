#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/cudaarithm.hpp"

//#define CLUSTER_NUM       20
//#define ITERATION         10

using namespace cv;
using namespace std;
using namespace cuda;

int ROWS, COLS, CLUSTER_NUM, CLUSTER_MAX, ITERATION;

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

void updateCentroid(unsigned char** label, Vec3b* centroids, Mat original) {

    int index;
    // use Vec3i(vector3 int) to hold a bigger number
    Vec3i sum[CLUSTER_NUM];
    for (int i = 0; i < CLUSTER_NUM; i++) {
        sum[i] = Vec3i(0, 0, 0);
    }

    // dynamically allocating memory because CLUSTER_NUM changes at run time
    int* count = new int[CLUSTER_NUM];
    for (int i = 0; i < CLUSTER_NUM; i++) {
        count[i] = 0;
    }
    // the addition part of averaging
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            index = label[i][j];
            sum[index].val[0] += original.at<Vec3b>(i, j).val[0];
            sum[index].val[1] += original.at<Vec3b>(i, j).val[1];
            sum[index].val[2] += original.at<Vec3b>(i, j).val[2];
            count[index]++;
        }
    }
    // the division part of averaging
    for (int i = 0; i < CLUSTER_NUM; i++) {
        // randomize pixel if no pixels are assigned to the cluster
        if (count[i] == 0) {
            centroids[i] = randomPixel();
            continue;
        }
        sum[i].val[0] /= count[i];
        sum[i].val[1] /= count[i];
        sum[i].val[2] /= count[i];
        centroids[i] = sum[i];
    }
}

Mat labelToImage(unsigned char** label, Vec3b* centroids, Mat image) {
    Mat result = image.clone();
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            result.at<Vec3b>(i,j) = centroids[label[i][j]];
        }
    }
    return result;
}

int main(int argc, char** argv)
{
    // check for correct input format
    if (argc != 4) {
        printf("usage: ./gpu_fix filename(without '.jpg') #clusters #iterations\n");
        exit(-1);
    }

    // read inputs from commandline
    string fn = argv[1];
    CLUSTER_NUM = stoi(argv[2]);
    ITERATION = stoi(argv[3]);

    // check version c++11 or c++98
    printf("c++ version: ");
    if( __cplusplus == 201103L ) std::cout << "C++11\n" ;
    else if( __cplusplus == 199711L ) std::cout << "C++98\n" ;
    else std::cout << "pre-standard C++\n" ;

    // print image information to commandline
    printf("number of clusters: %i\n", CLUSTER_NUM);
    printf("number of iterations: %i\n", ITERATION);
    printf("mode: CPU\n");


    // set random seed to a particular number to compare against GPU
    //srand(1);
    // randomize seed
    srand(time(NULL));
    Mat h_image, h_image_final;
    //string fn = "tree";
    h_image = imread(fn + ".jpg", IMREAD_COLOR); // read the image
    if (!h_image.data) {
        printf("No image data \n");
        return -1;
    }
    ROWS = h_image.rows;
    COLS = h_image.cols;

    printf("file size: %ix%i\n", ROWS, COLS);

    h_image_final.create(ROWS, COLS, CV_8UC1);

    // split original image to RGB channels
    Mat h_image_channel[3];
    split(h_image, h_image_channel);
    Mat h_image_final_channel[3];
    h_image_final_channel[0] = h_image_channel[0].clone();
    h_image_final_channel[1] = h_image_channel[1].clone();
    h_image_final_channel[2] = h_image_channel[2].clone();

    //generate random pixel for centroid

    Vec3b* h_centroids = (Vec3b*) malloc(CLUSTER_NUM * sizeof(Vec3b));
    for (int i = 0; i < CLUSTER_NUM; i++) {
        h_centroids[i] = randomPixel();
    }
    

    //generate a 2D array for labelling
    unsigned char ** h_label = (unsigned char **) malloc(ROWS * sizeof(unsigned char *));
    for (int i = 0; i < ROWS; i++) {
      h_label[i] = (unsigned char *) malloc(COLS * sizeof(unsigned char));
    }

    // loop through different numbers of clusters and produce an image 
    CLUSTER_MAX = CLUSTER_NUM;
    for (int c = 1; c <= CLUSTER_MAX; c++) {
        CLUSTER_NUM = c;
        // randomize centroid pixels
        for (int i = 0; i < c; i++) {
            h_centroids[i] = randomPixel();
        }
        // the real k-means: for a given number of clusters, update the centroid k times and output an image
        for (int k = 0; k < ITERATION; k++) {
            // loop through every pixel in the image
            for (int i = 0; i < ROWS; i++) {
                for (int j = 0; j < COLS; j++) {
                    h_label[i][j] = mapToCluster(h_image.at<Vec3b>(i,j), h_centroids);
                }
            }

            // update the image to the cluster colors
            h_image_final = labelToImage(h_label, h_centroids, h_image);

            // update the centroid locations
            updateCentroid(h_label, h_centroids, h_image);
        }
        // adding the number of iteration to the output image
        putText(h_image_final, to_string(c), Point(20, 20), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(50,50,100), 1, CV_AA);
        // writes to file for each iteration of image
        imwrite(fn + "/" + fn + "_cpu_" + to_string(c) + ".jpg",h_image_final);
    }
    imwrite(fn + "_cpu_time.jpg", h_image_final);
    return 0;
}

//display image
/*
namedWindow("Iteration " + to_string(k), WINDOW_AUTOSIZE );
if(!h_image_final.empty()){
    imshow("Iteration " + to_string(k), h_image_final);
}
waitKey(0);
*/