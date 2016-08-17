
#include "opencv2\opencv.hpp"
#include <vector>
#include "LazySnapping.h"

using namespace std;
using namespace cv;

int currentMode = 0;// indicate foreground or background, foreground as default
CvScalar paintColor[2] = { CV_RGB(0,0,255), CV_RGB(255,0,0) };

vector<cv::Point> forePts;
vector<cv::Point> backPts;

char* winName = "OriginalImage";
char* resultName = "FinalSegmentation";
Mat src, imageDraw, object, gray;
LazySnapping ls;

bool isPtInVector(cv::Point pt, vector<cv::Point> points)
{
	for (int i = 0; i < points.size(); i++) {
		if (pt.x == points[i].x && pt.y == points[i].y) {
			return true;
		}
	}
	return false;
}

void lazySnapping()
{
	if (backPts.size() == 0 || forePts.size() == 0) 
	{
		return;
	}

	ls.setForegroundPoints(forePts);
	ls.setBackgroundPoints(backPts);
	ls.runMaxFlow();

	object = ls.getImageColor();

	cv::imwrite("object.jpg", object);
	cv::imshow(resultName, object);
}

void on_mouse(int event, int x, int y, int flags, void*)
{
	if (event == CV_EVENT_LBUTTONUP) {

	}
	else if (event == CV_EVENT_LBUTTONDOWN) {

	}
	else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON)) {
		cv::Point pt = cv::Point(x, y);
		if (currentMode == 0) {//foreground
			if (!isPtInVector(pt, forePts))
			{
				forePts.push_back(pt);
			}
		}
		else {//background
			if (!isPtInVector(pt, backPts))
			{
				backPts.push_back(pt);
			}
				
		}
		cv::circle(imageDraw, pt, 2, paintColor[currentMode]);
		cv::imwrite("draw.jpg", imageDraw);
		cv::imshow(winName, imageDraw);
	}
}

void on_mouse_boundary(int event, int x, int y, int flags, void*)
{
	if (event == CV_EVENT_LBUTTONUP) {
		object = ls.changeLabelSegment(y, x);
		cv::imwrite("object.jpg", object);
		cv::imshow(resultName, object);
	}
	else if (event == CV_EVENT_LBUTTONDOWN) {

	}
}

int main(int argc, char** argv)
{
	cvNamedWindow(winName, 1);
	cvSetMouseCallback(winName, on_mouse, 0);

	src = imread(argv[1]);
	if (!src.data)
	{
		cout << "No input file image";
		return -1;
	}

	gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	ls.setSourceImage(src);
	ls.setGrayImage(gray);
	ls.initMarkers();

	imageDraw = src.clone();
	cv::imshow(winName, src);

	for (;;) 
	{
		int c = cvWaitKey(0);
		c = (char)c;
		if (c == 'p') //exit
		{
			break;
		}
		else if (c == 'r') //reset
		{
			src = imread(argv[1]);
			imageDraw = src.clone();
			forePts.clear();
			backPts.clear();
			currentMode = 0;
			cv::imshow(winName, src);
		}
		else if (c == 'b') //change to background selection
		{
			currentMode = 1;

		}
		else if (c == 'f') //change to foreground selection
		{
			currentMode = 0;
		}
		else if (c == ' ') //finish set samples, run max flow
		{
			lazySnapping();
		}
	}

	cvNamedWindow(resultName, 2);
	cvSetMouseCallback(resultName, on_mouse_boundary, 0);
	cv::imshow(resultName, object);

	for (;;)
	{
		int c = cvWaitKey(0);
		c = (char)c;
		if (c == ' ') //finish change label
		{
			break;
		}
	}

	waitKey(0);

	return 0;
}