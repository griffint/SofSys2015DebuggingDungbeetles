/* Cuda Version of k means */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define CLUSTER_NUM       30
#define ITERATION         10

using namespace cv;
using namespace std;

int ROWS, COLS;

//Cuda code starts here
__global__
void cuda_cluster_map(Vec3b * d_out, Vec3b * d_in) {

}
// ends here

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

    int count[CLUSTER_NUM] = { 0 }; // all elements 0
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
    if( __cplusplus == 201103L ) std::cout << "C++11\n" ;
    else if( __cplusplus == 199711L ) std::cout << "C++98\n" ;
    else std::cout << "pre-standard C++\n" ;
    srand(time(NULL)); //reset the random seed for this particular run
    Mat image, image_final;
    image = imread("griffin.jpg", IMREAD_COLOR); // read the image
    if (!image.data) {
        printf("No image data \n");
        return -1;
    }
    ROWS = image.rows;
    COLS = image.cols;
    image_final.create(ROWS, COLS, CV_8UC1);

    //generate random pixel
    Vec3b* centroids = (Vec3b*) malloc(CLUSTER_NUM * sizeof(Vec3b));
    for (int i = 0; i < CLUSTER_NUM; i++) {
        centroids[i] = randomPixel();
        printf("original %i : ", i);
        printf("%u, ", centroids[i].val[0]);
        printf("%u, ", centroids[i].val[1]);
        printf("%u\n", centroids[i].val[2]);
    }

    //generate a 2D array for labelling
    unsigned char ** label = (unsigned char **) malloc(ROWS * sizeof(unsigned char *));
    for (int i = 0; i < ROWS; i++) {
      label[i] = (unsigned char *) malloc(COLS * sizeof(unsigned char));
    }
    //Vec3b pix = image.at<Vec3b>(1,1);

    string fn_head = "griffin/griffin_";

    //CUDA code starts here
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTE = ARRAY_SIZE * sizeof(Vec3b);
    //ends here


    // the real k-means
    for (int k = 0; k < ITERATION; k++) {
        //attemp to map all pixels to cluster
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                label[i][j] = mapToCluster(image.at<Vec3b>(i,j), centroids);
            }
        }
        // update the image to the cluster colors
        image_final = labelToImage(label, centroids, image);
        
        // save this file
        string fn = fn_head + to_string(k) + ".jpg";
        cout << fn << endl;
        imwrite(fn,image_final);

        // update the centroid locations
        updateCentroid(label, centroids, image);
    }

    for (int i = 0; i < CLUSTER_NUM; i++) {
        printf("%i, ", centroids[i].val[0]);
        printf("%i, ", centroids[i].val[1]);
        printf("%i\n", centroids[i].val[2]);
    }

    

    //image_final = Mat(image);


    // update centroid
    //printf("%d iterations\n", ITERATION);

    /*
    Vec3b a = Vec3b(0, 0, 0);
    Vec3i b = Vec3i(300, 2, 3);
    a.val[0] += b.val[0];
    printf("test: %i\n", a.val[0
    */


    /*
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            printf("%u", label[i][j]);
        }
        printf("\n");;
    }*/


    //Vec3b is vector3 of unsigned char: 0-255
    /*for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf( "%u, ", image.at<Vec3b>(i, j).val[0]);
            printf( "%u, ", image.at<Vec3b>(i, j).val[1]);
            printf( "%u\n", image.at<Vec3b>(i, j).val[2]);
        }
    }*/
    

    // splits the image into "BGR" channel
    
    //Mat channel[3];
    //split(image, channel);
    //printf("image cols: %i\n", COLS);
    //printf("image rows: %i\n", ROWS);
    //printf("channel[0] cols: %i\n", channel[0].cols);
    //printf("channel[0] rows: %i\n", channel[0].rows);
    

    //channel[0]=Mat::zeros(ROWS, COLS, CV_8UC1);//Set B to 0
    //channel[1]=Mat::zeros(ROWS, COLS, CV_8UC1);//Set G to 0
    //channel[2]=Mat::zeros(ROWS, COLS, CV_8UC1);//Set R to 0

    
    //Merging channels into image
    //merge(channel,3,image_final);

    //write to a file (create if doesn't exist)


    //display image
    //namedWindow("Display Image", WINDOW_AUTOSIZE );
    //imshow("Display Image", image_final);
    //waitKey(0);




    return 0;
}