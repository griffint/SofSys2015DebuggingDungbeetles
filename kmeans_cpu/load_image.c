#include <stdio.h>
#include <iostream.h>
#include <vector>

#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
    char* filename = "C:\\Research\abc.pgm";  
     IplImage *img0;

    if( (img0 = cvLoadImage(filename,-1)) == 0 )
        return 0;

    cvNamedWindow( "image", 0 );
    cvShowImage( "image", img0 );
    cvWaitKey(0);  
    cvDestroyWindow("image");
    cvReleaseImage(&img0);



    return 0;
}