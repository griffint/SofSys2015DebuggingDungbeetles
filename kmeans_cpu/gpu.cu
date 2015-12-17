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

#define CLUSTER_NUM       40
#define ITERATION         20

using namespace cv;
using namespace std;
using namespace cuda;

int ROWS, COLS;

__global__ void GPU_mapToCluster(unsigned char * d_label, size_t pitch, Vec3b* d_centroids, Vec3i* d_centroids_new, int* d_count, unsigned char* d_image_B, unsigned char* d_image_G, unsigned char* d_image_R, int ROWS, int COLS ) {
    int col = blockIdx.x;
    int row = blockIdx.y;
    int index = COLS * row + col;
    float temp, a, b, c;
    float d = FLT_MAX;
    unsigned char cluster;
    unsigned char p1_b = d_image_B[index];
    unsigned char p1_g = d_image_G[index];
    unsigned char p1_r = d_image_R[index];
    for (unsigned char i = 0; i < CLUSTER_NUM; i++) {
        a = (float) (p1_b - d_centroids[i].val[0]);
        b = (float) (p1_g - d_centroids[i].val[1]);
        c = (float) (p1_r - d_centroids[i].val[2]);
        temp = sqrt(a*a + b*b + c*c);
        if (temp < d) {
            d = temp;
            cluster = i;
        }
    }
    *((unsigned char*)((char*)d_label + row * pitch) + col) = cluster;
    // adding to the new cluster for averaging later
    atomicAdd(&d_centroids_new[cluster].val[0], p1_b);
    atomicAdd(&d_centroids_new[cluster].val[1], p1_g);
    atomicAdd(&d_centroids_new[cluster].val[2], p1_r);
    atomicAdd(&d_count[cluster], 1);
}

__global__ void GPU_labelToImage (unsigned char * d_label, size_t pitch, Vec3b* d_centroids, unsigned char* d_image_final_B, unsigned char* d_image_final_G, unsigned char* d_image_final_R, int COLS, int ROWS) {
    int col = blockIdx.x;
    int row = blockIdx.y;
    int color = threadIdx.x;
    int index_image = row*COLS + col;
    int index_label = *((unsigned char*)((char*)d_label + row * pitch) + col);

    switch(color) {
      case 0 :
        d_image_final_B[index_image] = d_centroids[index_label].val[0];
        break;
      case 1:
        d_image_final_G[index_image] = d_centroids[index_label].val[1];
        break;
      case 2 :
        d_image_final_R[index_image] = d_centroids[index_label].val[2];
        break;
      default :
        printf("wtf there is a bug lol\n" );
   }
}

__global__ void GPU_updateCentroid(Vec3b* d_centroids, Vec3i* d_centroids_new, int* d_count){
    int index = blockIdx.x;
    if (d_count[index] != 0) {
        d_centroids[index].val[0] = d_centroids_new[index].val[0]/d_count[index];
        d_centroids[index].val[1] = d_centroids_new[index].val[1]/d_count[index];
        d_centroids[index].val[2] = d_centroids_new[index].val[2]/d_count[index];
    }
    // resetting the values for next update
    d_count[index] = 0;
    // retardedly setting the bgr value to 0 one by one because cannot reference host methods
    d_centroids_new[index].val[0] = 0;
    d_centroids_new[index].val[1] = 0;
    d_centroids_new[index].val[2] = 0;
}

// generate a random RGB pixel as centroid
Vec3b randomPixel() {
    return Vec3b(rand()%255, rand()%255, rand()%255);
}

int main(int argc, char** argv)
{
    // check version c++11 or c++98
    if( __cplusplus == 201103L ) std::cout << "C++11\n" ;
    else if( __cplusplus == 199711L ) std::cout << "C++98\n" ;
    else std::cout << "pre-standard C++\n" ;

    printf("number of clusters: %i\n", CLUSTER_NUM);
    printf("number of iterations: %i\n", ITERATION);
    printf("mode: GPU\n");
    
    srand(1);
    //srand(time(NULL)); //reset the random seed for this particular run
    Mat h_image, h_image_final;
    string fn = "riven";
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
    /*
    for (int i = 0; i < CLUSTER_NUM; i++) {
        h_centroids[i] = randomPixel();
        printf("original %i : ", i);
        printf("%u, ", h_centroids[i].val[0]);
        printf("%u, ", h_centroids[i].val[1]);
        printf("%u\n", h_centroids[i].val[2]);
    }
    */

    // generate data structure for new centroids and setting them to zero
    // we're still trying to update the centroids in cuda because:
    // 1. If we don't we have to download the centroid back onto CPU, meaning the overhead is inevitable
    // 2. Given 1, with atomic operations, the kernel code performance is at worst the same as the GPU code. Hence there is no reason not to do this in GPU

    Vec3i* h_centroids_new = (Vec3i*) malloc(CLUSTER_NUM * sizeof(Vec3i));
    for (int i = 0; i < CLUSTER_NUM; i++) {
        h_centroids_new[i] = Vec3i(0, 0, 0);
    }

    // generate count array to initialize to 0
    int* h_count = (int*) malloc(CLUSTER_NUM * sizeof(int));
    for (int i = 0; i < CLUSTER_NUM; i++) {
        h_count[i] = 0;
    }

    //generate a 2D array for labelling
    unsigned char ** h_label = (unsigned char **) malloc(ROWS * sizeof(unsigned char *));
    for (int i = 0; i < ROWS; i++) {
      h_label[i] = (unsigned char *) malloc(COLS * sizeof(unsigned char));
    }

    // gpu data structure
    // --original image-- three 1D array of unsigned char for BGR
    unsigned char* d_image_B;
    unsigned char* d_image_G;
    unsigned char* d_image_R;
    cudaMalloc((void**) &d_image_B, ROWS * COLS * sizeof(unsigned char));
    cudaMalloc((void**) &d_image_G, ROWS * COLS * sizeof(unsigned char));
    cudaMalloc((void**) &d_image_R, ROWS * COLS * sizeof(unsigned char));
    cudaMemcpy(d_image_B, h_image_channel[0].data, ROWS * COLS * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_image_G, h_image_channel[1].data, ROWS * COLS * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_image_R, h_image_channel[2].data, ROWS * COLS * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // --new image-- three 1D array of unsigned char of BGR
    unsigned char* d_image_final_B;
    unsigned char* d_image_final_G;
    unsigned char* d_image_final_R;
    cudaMalloc((void**) &d_image_final_B, ROWS * COLS * sizeof(unsigned char));
    cudaMalloc((void**) &d_image_final_G, ROWS * COLS * sizeof(unsigned char));
    cudaMalloc((void**) &d_image_final_R, ROWS * COLS * sizeof(unsigned char));


    // --labels--    2D array unsigned char on GPU
    unsigned char* d_label;
    // variable for data structure alignment when using cudaMallocPitch
    size_t pitch_label;  
    // access method: (data type*)((char*)d_label + Row * pitch) + Column;
    cudaMallocPitch(&d_label, &pitch_label, COLS * sizeof(unsigned char), ROWS);

    // --centroids--   1D array of unsign char on GPU
    Vec3b* d_centroids;
    cudaMalloc((void**) &d_centroids, CLUSTER_NUM * sizeof(Vec3b));

    // copy from host to device 
    cudaMemcpy(d_centroids, h_centroids, CLUSTER_NUM * sizeof(Vec3b), cudaMemcpyHostToDevice);

    // --newcentroids-- 1D array of unsign char on GPU to update centroids
    Vec3i* d_centroids_new;
    cudaMalloc((void**) &d_centroids_new, CLUSTER_NUM * sizeof(Vec3i));
    cudaMemcpy(d_centroids_new, h_centroids_new, CLUSTER_NUM * sizeof(Vec3i), cudaMemcpyHostToDevice);
    // --count--        1D array of int for number of pixels in each cluster
    int* d_count;
    cudaMalloc((void**) &d_count, CLUSTER_NUM * sizeof(int));
    cudaMemcpy(d_count, h_count, CLUSTER_NUM * sizeof(int), cudaMemcpyHostToDevice);

    string fn_head = "test/test_";

    // the real k-means
    for (int k = 0; k < ITERATION; k++) {
        // map all pixels to cluster
        GPU_mapToCluster<<<dim3(COLS, ROWS), dim3(1)>>>(d_label, pitch_label, d_centroids, d_centroids_new, d_count, d_image_B, d_image_G, d_image_R, ROWS, COLS);
        cudaDeviceSynchronize();

        // update the image to the cluster colors
        GPU_labelToImage<<<dim3(COLS, ROWS), dim3(3)>>>(d_label, pitch_label, d_centroids, d_image_final_B, d_image_final_G, d_image_final_R, COLS, ROWS);
        cudaDeviceSynchronize();

        // download the calculated image channels from GPU to CPU
        cudaMemcpy(h_image_final_channel[0].data, d_image_final_B, ROWS * COLS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_image_final_channel[1].data, d_image_final_G, ROWS * COLS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_image_final_channel[2].data, d_image_final_R, ROWS * COLS * sizeof(unsigned char), cudaMemcpyDeviceToHost);

        //Merging the new channels into image
        merge(h_image_final_channel, 3, h_image_final);

        //display image
        /*
        namedWindow("Iteration " + to_string(k), WINDOW_AUTOSIZE );
        if(!h_image_final.empty()){
            imshow("Iteration " + to_string(k), h_image_final);
        }
        waitKey(0);
        */
        GPU_updateCentroid<<<dim3(CLUSTER_NUM), dim3(1)>>>(d_centroids, d_centroids_new, d_count);
        cudaDeviceSynchronize();

        // save this file
        //string fn = fn_head + to_string(k) + ".jpg";
        //cout << fn << endl;
        //imwrite(fn,h_image_final);

        // update the centroid locations
        //updateCentroid(h_label, h_centroids, h_image);
    }
    imwrite(fn + "_gpu.jpg",h_image_final);
    waitKey(0);

    return 0;
}