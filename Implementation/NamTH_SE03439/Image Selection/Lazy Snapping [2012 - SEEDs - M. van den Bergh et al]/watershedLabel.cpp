
#include <iomanip>
#include <fstream>
#include "watershedLabel.h"
#include <E:/OpenCV/Builds/install/include/opencv2/ximgproc.hpp>

#include <ctype.h>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

static const char* window_name = "SEEDS Superpixels";

Mat Watershed::getMarkersLabel()
{
	namedWindow(window_name, 0);
	int num_iterations = 100;
	int prior = 2;
	bool double_step = false;
	int num_superpixels = 600;
	int num_levels = 12;
	int num_histogram_bins = 10;

	Mat result, mask, frame;
	Ptr<SuperpixelSEEDS> seeds;
	int width, height;
	int display_mode = 0;

	//// Create a kernel that we will use for accuting/sharpening our image
	//cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
	//	1, 1, 1,
	//	1, -8, 1,
	//	1, 1, 1);

	//cv::Mat imgLaplacian;
	//cv::Mat sharp = src; // copy source image to another temporary one
	//cv::filter2D(sharp, imgLaplacian, CV_32F, kernel);
	//src.convertTo(sharp, CV_32F);
	//cv::Mat imgResult = sharp - imgLaplacian;
	//// convert back to 8bits gray scale
	//imgResult.convertTo(imgResult, CV_8UC3);
	//imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
	//cv::imshow("Laplace Filtered Image", imgLaplacian);
	//cv::imshow("New Sharped Image", imgResult);

	//src = imgResult;

	src.copyTo(frame);

	width = frame.size().width;
	height = frame.size().height;
	seeds = createSuperpixelSEEDS(width, height, frame.channels(), num_superpixels,
		num_levels, prior, num_histogram_bins, double_step);

	Mat converted;
	cvtColor(frame, converted, COLOR_BGR2HSV);

	seeds->iterate(converted, num_iterations);
	result = frame;

	/* retrieve the segmentation result */
	Mat labels;
	seeds->getLabels(labels);

	/* get the contours for displaying */
	seeds->getLabelContourMask(mask, false);
	result.setTo(Scalar(0, 0, 255), mask);

	/* display output */
	switch (display_mode)
	{
		case 0: //superpixel contours
			imshow(window_name, result);
			break;
		case 1: //mask
			imshow(window_name, mask);
			break;
		case 2: //labels array
		{
			// use the last x bit to determine the color. Note that this does not
			// guarantee that 2 neighboring superpixels have different colors.
			const int num_label_bits = 2;
			labels &= (1 << num_label_bits) - 1;
			labels *= 1 << (16 - num_label_bits);
			imshow(window_name, labels);
		}
		break;
	}

	markers = labels;
	imwrite("watershed.bmp", result);

	ofstream ofs("SuperSeed.txt");
	for (int i = 0; i < markers.rows; i++) {
		for (int j = 0; j < markers.cols; j++)
			ofs << std::setw(4) << std::left << markers.at<int>(i, j);
		ofs << std::endl;
	}
	ofs.close();

	return markers;
}
