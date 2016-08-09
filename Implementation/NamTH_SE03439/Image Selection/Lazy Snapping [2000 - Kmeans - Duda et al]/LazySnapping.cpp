
#include "opencv2\opencv.hpp"
#include <vector>
#include "LazySnapping.h"

using namespace cv;

CvScalar paintColor[2] = { CV_RGB(0,0,255), CV_RGB(255,0,0) };

vector<CvPoint> forePts;
vector<CvPoint> backPts;

char* imgName = "Image";
char* initName = "Initial Label";
char* resultName = "Full Label";

IplImage* image = NULL;
IplImage* imageDraw;
IplImage* object;
Mat src;
Mat draw;
LazySnapping ls;


void on_mouse_boundary(int event, int x, int y, int flags, void*)
{
	if (event == CV_EVENT_LBUTTONUP) {
		object = ls.changeLabelSegment(y, x);
		cvShowImage(resultName, object);
		cvSaveImage("object.bmp", object);
		cout << "Finish on_mouse_boundary" << endl;
	}
	else if (event == CV_EVENT_LBUTTONDOWN) {

	}
}

bool is_file_exist(const char *fileName)
{
	std::ifstream infile(fileName);
	return infile.good();
}

void getSeed()
{
	for (int i = 0; i < draw.rows; i++)
		for (int j = 0; j < draw.cols; j++)
		{
			Vec3b pixel = draw.at<Vec3b>(i, j);
			if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 255)
			{
				forePts.push_back(cvPoint(j, i));
			}
			else if (pixel[0] == 255 && pixel[1] == 0 && pixel[2] == 0)
			{
				backPts.push_back(cvPoint(j, i));
			}
		}

	ls.setForegroundPoints(forePts);
	ls.setBackgroundPoints(backPts);
}

int main(int argc, char** argv)
{
	image = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);
	if (image == NULL)
	{
		cout << "No input file image";
		return -1;
	}

	src = imread(argv[1]);

	ls.setImage(image);
	ls.setSourceImage(src);

	char* markerLabel = argv[2];
	if (!is_file_exist(markerLabel))
	{
		cout << "No watershed-label file";
		return -1;
	}
	
	ls.setPath(markerLabel);
	imageDraw = cvLoadImage(argv[3], CV_LOAD_IMAGE_COLOR);
	if (imageDraw == NULL)
	{
		cout << "No draw file image";
		return -1;
	}
	draw = imread(argv[3]);

	cvNamedWindow(imgName, 1);
	cvShowImage(imgName, image);

	cvNamedWindow(initName, 2);
	cvShowImage(initName, imageDraw);

	getSeed();
	if (backPts.size() == 0 || forePts.size() == 0)
	{
		cout << "Sample is not set" << endl;
		return -1;
	}

	ls.initWaterShed();
	ls.initSegment();
	ls.runMaxFlow();

	object = ls.getImageColor();

	cvNamedWindow(resultName, 2);
	cvSaveImage("full-label.bmp", object);
	cvShowImage(resultName, object);
	cvSetMouseCallback(resultName, on_mouse_boundary, 0);

	for (;;)
	{
		int c = cvWaitKey(0);
		c = (char)c;
		if (c == 27) //exit
		{
			break;
		}
	}
	
	waitKey(0);

	cvReleaseImage(&image);
	cvReleaseImage(&imageDraw);
	cvReleaseImage(&object);

	return 0;
}