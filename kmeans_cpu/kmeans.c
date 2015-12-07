/*kmeans_cpu*/
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

typedef struct {
	unsigned char r;
	unsigned char g;
	unsigned char b;
} Centroid;

Centroid *randomizeCentroid(){
	unsigned char r, g, b;
	time_t t;
	int n = 3;
	
	/* Initializes random number generator */
	srand((unsigned) time(&t));
	r = rand() % 255;
	g = rand() % 255;
	b = rand() % 255;

	Centroid *centroid = (Centroid *) malloc (sizeof (Centroid));
	centroid->r = r;
	centroid->g = g;
	centroid->b = b;
	return centroid;
}

int main()
{
	int i, j, k;

	int numcentroids = 5; /* Centroids */

	int numfeatures = 3; /* RGB */
	int iterations = 0;

	Centroid *centroid = randomizeCentroid();

	printf("%d\n",centroid->r);
	printf("%d\n",centroid->g);
	printf("%d\n",centroid->b);

	/* number of ks */


	printf("Hello World\n");
	return 0;
}