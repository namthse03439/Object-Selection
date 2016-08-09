#pragma once
#include <opencv2\opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utility.hpp>

class Watershed
{
private:
	cv::Mat src;
	cv::Mat gray;

public:

	Watershed() = default;
	~Watershed() = default;

	void setSourceImage(cv::Mat source) { src = source; }
	void setGrayImage(cv::Mat grayImg) { gray = grayImg; }

	cv::Mat getMarkersLabel();

};