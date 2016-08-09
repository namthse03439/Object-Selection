
#include <iomanip>
#include <fstream>
#include "watershedLabel.h"
#include <E:/OpenCV/Builds/install/include/opencv2/ximgproc.hpp>

#include <ctype.h>
#include <stdio.h>
#include <iostream>
#include "SeedsRevised.h"
#include "Tools.h"
#include "guidedfilter.h"

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

const int MaxR = 2000;
const int MaxC = 2000;

Mat Watershed::getMarkersLabel()
{
	int64 start = getTickCount();

	// Number of desired superpixels.
	int superpixels = 600;

	// Number of bins for color histograms (per channel).
	int numberOfBins = 10;

	// Size of neighborhood used for smoothing term, see [1] or [2].
	// 1 will be sufficient, >1 will slow down the algorithm.
	int neighborhoodSize = 1;

	// Minimum confidence, that is minimum difference of histogram intersection
	// needed for block updates: 0.1 is the default value.
	float minimumConfidence = 0.1;

	// The weighting of spatial smoothing for mean pixel updates - the euclidean
	// distance between pixel coordinates and mean superpixel coordinates is used
	// and weighted according to:
	//  (1 - spatialWeight)*colorDifference + spatialWeight*spatialDifference
	// The higher spatialWeight, the more compact superpixels are generated.
	float spatialWeight = 0.2;

	// Instantiate a new object for the given image.
	SEEDSRevisedMeanPixels seeds(src, superpixels, numberOfBins, neighborhoodSize, minimumConfidence, spatialWeight);

	int iterations = 500;
	// Initializes histograms and labels.
	seeds.initialize();
	// Runs a given number of block updates and pixel updates.
	seeds.iterate(iterations);

	// bgr color for contours:
	int bgr[] = { 255, 255, 255 };

	// seeds.getLabels() returns a two-dimensional array containing the computed
	// superpixel labels.
	cv::Mat contourImage = Draw::contourImage(seeds.getLabels(), src, bgr);

	int** labels = seeds.getLabels();

	cv::Mat markers = Mat::zeros(src.rows, src.cols, CV_32S);

	for (int i = 0; i < markers.rows; i++) {
		for (int j = 0; j < markers.cols; j++)
		{
			markers.at<int>(i, j) = labels[i][j];
		}
	}

	int64 end = cv::getTickCount();

	std::cout << double(end - start) / cv::getTickFrequency() << endl;

	return markers;
}
