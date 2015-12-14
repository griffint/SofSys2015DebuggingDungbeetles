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

int ROWS, COLS, CLUSTER_NUM, ITERATION;

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
    Vec3i sum[CLUSTER_NUM];
    for (int i = 0; i < CLUSTER_NUM; i++) {
        sum[i] = Vec3i(0, 0, 0);
    }

     // all elements 0
    int* count = new int[CLUSTER_NUM];
    for (int i = 0; i < CLUSTER_NUM; i++) {
        count[i] = 0;
    }
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            index = label[i][j];
            sum[index].val[0] += original.at<Vec3b>(i, j).val[0];
            sum[index].val[1] += original.at<Vec3b>(i, j).val[1];
            sum[index].val[2] += original.at<Vec3b>(i, j).val[2];
            count[index]++;
        }
    }

    for (int i = 0; i < CLUSTER_NUM; i++) {
        if (count[i] == 0) {
            continue;
        }
        //printf("normalize (sums/count):\n");
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
            //printf("%u", label[i][j]);
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
    string fn = argv[1];
    CLUSTER_NUM = stoi(argv[2]);
    ITERATION = stoi(argv[3]);
    // check version c++11 or c++98
    printf("c++ version: ");
    if( __cplusplus == 201103L ) std::cout << "C++11\n" ;
    else if( __cplusplus == 199711L ) std::cout << "C++98\n" ;
    else std::cout << "pre-standard C++\n" ;

    printf("number of clusters: %i\n", CLUSTER_NUM);
    printf("number of iterations: %i\n", ITERATION);
    printf("mode: CPU\n");



    srand(1);
    //srand(time(NULL)); //reset the random seed for this particular run
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

    // split original image to RGB
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
        //printf("original %i : ", i);
        //printf("%u, ", h_centroids[i].val[0]);
        //printf("%u, ", h_centroids[i].val[1]);
        //printf("%u\n", h_centroids[i].val[2]);
    }
    

    //generate a 2D array for labelling
    unsigned char ** h_label = (unsigned char **) malloc(ROWS * sizeof(unsigned char *));
    for (int i = 0; i < ROWS; i++) {
      h_label[i] = (unsigned char *) malloc(COLS * sizeof(unsigned char));
    }

    string fn_head = "test/test_";

    // the real k-means
    for (int k = 0; k < ITERATION; k++) {
        // attemp to map all pixels to cluster
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                h_label[i][j] = mapToCluster(h_image.at<Vec3b>(i,j), h_centroids);
            }
        }

        // update the image to the cluster colors
        h_image_final = labelToImage(h_label, h_centroids, h_image);


        //display image
        /*
        namedWindow("Iteration " + to_string(k), WINDOW_AUTOSIZE );
        if(!h_image_final.empty()){
            imshow("Iteration " + to_string(k), h_image_final);
        }
        waitKey(0);
        */

        // save this file
        //string fn = fn_head + to_string(k) + ".jpg";
        //cout << fn << endl;
        imwrite(fn + "/" + fn + "_cpu.jpg",h_image_final);

        // update the centroid locations
        updateCentroid(h_label, h_centroids, h_image);
    }
    imwrite(fn + "_cpu.jpg", h_image_final);
    return 0;
}