#pragma once

#include "opencv2\opencv.hpp"
#include "graph.h"
#include <vector>
#include <iostream>
#include <list>
#include <fstream>
#include <algorithm>
#include <iomanip>

using namespace std;
using namespace cv;

typedef Graph<double, double, double> GraphType;

#define K 64

const double INFINNITE_MAX = 1e10;

class LazySnapping
{
private:
	vector<CvPoint> forePts;
	vector<CvPoint> backPts;
	IplImage* image;
	Mat src;

	// Graph info
	GraphType* graph;

	// pre-segmentation component number
	int n;

	vector< Vec3f > centers;

	// Foreground K-mean
	Mat foreground_seeds;
	Mat foreground_centers;
	
	// Background K-mean
	Mat background_seeds;
	Mat background_centers;

	// Segment connect to Source
	vector< bool > connectToSource;

	// Segment connect to Sink
	vector< bool > connectToSink;

	// Final labelling
	vector < int > FLabel;

	int KF;
	int KB;

	Mat markers;

	int r[4] = { 0, 1, 0,-1 };
	int c[4] = { 1, 0,-1, 0 };

	vector<pair<int, int>> edgeList;

	char* currentLabelPath;

public:
	LazySnapping() : graph(NULL)
	{
		forePts.clear();
		backPts.clear();
	}

	void setImage(IplImage* image) {
		this->image = image;
	}

	void setSourceImage(Mat image)
	{
		src = image.clone();
	}

	void setPath(char* path)
	{
		currentLabelPath = path;
	}

	void setForegroundPoints(vector<CvPoint> points)
	{
		forePts.clear();
		for (int i = 0; i < points.size(); i++)
		{
			forePts.push_back(points[i]);
		}
	}

	void setBackgroundPoints(vector<CvPoint> points)
	{
		backPts.clear();
		for (int i = 0; i < points.size(); i++)
		{
			backPts.push_back(points[i]);
		}
	}
	
	void k_meanForeground()
	{
		// K-mean for foreground seed
		connectToSource = vector< bool >(n, false);

		for (auto &i : forePts)
		{
			if (0 <= i.y && i.y < markers.rows
				&& 0 <= i.x && i.x <= markers.cols)
			{
				Vec3f idensity = src.at<Vec3b>(i.y, i.x);
				foreground_seeds.push_back(idensity);
				connectToSource[markers.at<int>(i.y, i.x)] = true;
			}
		}

		Mat label;
		Mat center;
		KF = K;
		if (foreground_seeds.rows < KF)
		{
			KF = foreground_seeds.rows;
		}

		kmeans(foreground_seeds,
			KF,
			label,
			cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 50, 1.0),
			1,
			cv::KMEANS_RANDOM_CENTERS,
			center);

		foreground_centers = center;
	}

	void k_meanBackground()
	{
		// K-mean for background seed
		connectToSink = vector< bool >(n, false);

		for (auto &i : backPts)
		{
			if (0 <= i.y && i.y < markers.rows
				&& 0 <= i.x && i.x < markers.cols)
			{
				Vec3f idensity = src.at<Vec3b>(i.y, i.x);
				background_seeds.push_back(idensity);
				connectToSink[markers.at<int>(i.y, i.x)] = true;
			}
		}

		Mat label;
		Mat center;
		KB = K;
		if (background_seeds.rows < KB)
		{
			KB = background_seeds.rows;
		}

		kmeans(background_seeds,
			KB,
			label,
			cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 50, 1.0),
			1,
			cv::KMEANS_RANDOM_CENTERS,
			center);

		background_centers = center;
	}

	Vec3f getAvgColor(const Mat dataPoints)
	{
		Vec3f result = { 0, 0, 0 };
		for (int i = 0; i < dataPoints.rows; i++)
		{
			Vec3f temp = dataPoints.at<int>(i, 0);
			result[0] += temp[0];
			result[1] += temp[1];
			result[2] += temp[2];
		}

		result[0] /= dataPoints.rows;
		result[1] /= dataPoints.rows;
		result[2] /= dataPoints.rows;

		return result;
	}

	void BFS(int i, int j, int currentLabel, int reLabel, Mat markers, Mat &dataPoints, Mat &visited)
	{
		// Create a queue for BFS
		list<pair<int, int>> queue;

		// Mark the current node as visited and enqueue it
		visited.at<int>(i, j) = true;
		queue.push_back(make_pair(i, j));

		int markers_rows = markers.rows;
		int markers_cols = markers.cols;

		while (!queue.empty())
		{
			// Dequeue a vertex from queue and print it
			int x = queue.front().first;
			int y = queue.front().second;

			Vec3f intensity = (Vec3f)src.at<Vec3b>(x, y);
			dataPoints.push_back(intensity);

			markers.at<int>(x, y) = reLabel;

			queue.pop_front();

			// Get all adjacent vertices of the dequeued vertex s
			// If a adjacent has not been visited, then mark it visited
			// and enqueue it
			for (int k = 0; k < 4; k++)
			{
				int xx = x + r[k];
				int yy = y + c[k];

				if (0 <= xx && xx < markers_rows
					&& 0 <= yy && yy < markers_cols)
					if (visited.at<int>(xx, yy) == 0)
					{
						int label = markers.at<int>(xx, yy);
						if (currentLabel == label)
						{
							visited.at<int>(xx, yy) = 1;
							queue.push_back(make_pair(xx, yy));
						}
					}
			}
		}
	}

	void initSuperPixel()
	{
		Mat visited = Mat::zeros(markers.rows, markers.cols, CV_32S);
		n = 0;
		for (int i = 0; i < markers.rows; i++)
		{
			for (int j = 0; j < markers.cols; j++)
				if (visited.at<int>(i, j) == 0)
				{
					Mat dataPoints;
					int currentLabel = markers.at<int>(i, j);
					BFS(i, j, currentLabel, n, markers, dataPoints, visited);
					centers.push_back(getAvgColor(dataPoints));
					n++;
				}
		}

		ofstream ofs("new_markers.txt");
		for (int i = 0; i < markers.rows; i++) {
			for (int j = 0; j < markers.cols; j++)
				ofs << std::setw(6) << std::left << markers.at<int>(i, j);
			ofs << std::endl;
		}

		ofs << n << endl;
		ofs.close();
	}

	void initNeighboring()
	{
		edgeList.clear();
		for (int x = 0; x < markers.rows; x++)
			for (int y = 0; y < markers.cols; y++)
			{
				int currentLabel = markers.at<int>(x, y);
				for (int k = 0; k < 4; k++)
				{
					int xx = x + r[k];
					int yy = y + c[k];

					if (0 <= xx && xx < markers.rows
						&& 0 <= yy && yy < markers.cols)
					{
						int label = markers.at<int>(xx, yy);
						if (currentLabel != label)
						{
							if (currentLabel < label)
							{
								edgeList.push_back(make_pair(currentLabel, label));
							}
							else
							{
								edgeList.push_back(make_pair(label, currentLabel));
							}
						}
					}
				}
			}

		std::sort(edgeList.begin(), edgeList.end(),
			[](const pair<int, int> & a, const pair<int, int> & b) -> bool
		{
			return (a.first < b.first || (a.first == b.first && a.second < b.second));
		});

		ofstream ofs("edgeList.txt");
		for (int i = 0; i < edgeList.size(); i++)
		{
			ofs << edgeList[i].first << " " << edgeList[i].second << endl;
		}
		ofs.close();
	}

	void initSegment()
	{
		// Get number of segment
		initSuperPixel();
		initNeighboring();
	}

	void initWaterShed()
	{
		markers = Mat::zeros(src.size(), CV_32S);
		ifstream ifs(currentLabelPath);

		for (int i = 0; i < markers.rows; i++) 
			for (int j = 0; j < markers.cols; j++)
				ifs >> markers.at<int>(i, j);
	}

	void initSeeds()
	{
		k_meanForeground();
		k_meanBackground();
	}

	void getE1(int currentLabel, double* energy)
	{
		// average distance
		double df = INFINNITE_MAX;
		double db = INFINNITE_MAX;
		for (int i = 0; i < KF; i++)
		{
			double mindf = norm(centers[currentLabel], foreground_centers.at<Vec3f>(i, 0), NORM_L2);		
			df = min(df, mindf);
		}

		for (int i = 0; i < KB; i++)
		{
			double mindb = norm(centers[currentLabel], background_centers.at<Vec3f>(i, 0), NORM_L2);
			db = min(db, mindb);
		}

		energy[0] = df / (db + df);
		energy[1] = db / (db + df);
	}

	float colorDistance(Vec3f color1, Vec3f color2)
	{
		return (float)sqrt((color1[0] - color2[0])*(color1[0] - color2[0]) +
			(color1[1] - color2[1])*(color1[1] - color2[1]) +
			(color1[2] - color2[2])*(color1[2] - color2[2]));
	}

	float getE2(int xlabel, int ylabel)
	{
		const float EPSILON = 1;
		float lambda = 0.5;
		
		float distance = colorDistance(centers[xlabel], centers[ylabel]);
		return (float) lambda / (EPSILON + distance);
	}

	void addEdge(int i, int j)
	{
		float e2 = getE2(i, j);
		graph->add_edge(i, j, e2, e2);
	}

	void initGraph()
	{
		graph = new GraphType(n, 8*n);

		double e1[2];

		graph->add_node(n);

		for (int i = 0; i < n; i++)
		{
			// calculate E1 energy
			if (connectToSource[i])
			{
				e1[0] = 0;
				e1[1] = INFINNITE_MAX;
			}
			else if (connectToSink[i])
			{
				e1[0] = INFINNITE_MAX;
				e1[1] = 0;
			}
			else
			{
				getE1(i, e1);
			}

			graph->add_tweights(i, e1[0], e1[1]);
		}

		int i = edgeList.front().first, j = edgeList.front().second;
		edgeList.pop_back();
		addEdge(i, j);
		while (!edgeList.empty())
		{
			int ii = edgeList.front().first, jj = edgeList.front().second;
			edgeList.pop_back();
			if (i == ii && j == jj)
				continue;

			i = ii; j = jj;
			addEdge(i, j);
		}
	}

	void runMaxFlow()
	{
		initSeeds();
		initGraph();
		float flowValue = graph->maxflow();

		cout << "flowValue = " << flowValue << endl;

		getLabellingValue();
	}

	void getLabellingValue()
	{
		FLabel.resize(n);

		for (int i = 0; i < n; i++)
			if (graph->what_segment(i) == GraphType::SOURCE)
			{
				FLabel[i] = 1;
			}
			else if (graph->what_segment(i) == GraphType::SINK)
			{
				FLabel[i] = 0;
			}
	}

	IplImage* getImageMask()
	{
		IplImage* gray = cvCreateImage(cvGetSize(image), 8, 1);
		for (int h = 0; h < markers.rows; h++)
		{
			unsigned char* p = (unsigned char*)gray->imageData + h*gray->widthStep;
			for (int w = 0; w < markers.cols; w++) 
			{
				int currentLabel = markers.at<int>(h, w);
				if (FLabel[currentLabel] == 1)
				{
					*p = 255;
				}
				else if (FLabel[currentLabel] == 0)
				{
					*p = 0;
				}
				p++;
			}
		}

		return gray;
	}

	IplImage* getImageColor()
	{
		IplImage* gray = getImageMask();
		IplImage* showImg = cvCloneImage(image);
		for (int h = 0; h < image->height; h++) {
			unsigned char* pgray = (unsigned char*)gray->imageData + gray->widthStep*h;
			unsigned char* pimage = (unsigned char*)showImg->imageData + showImg->widthStep*h;
			for (int width = 0; width < image->width; width++) {
				if (*pgray++ == 0) {
					pimage[0] = 0;
					pimage[1] = 0;
					pimage[2] = 255;
				}
				pimage += 3;
			}
		}

		return showImg;
	}

	IplImage* changeLabelSegment(int x, int y)
	{
		FLabel[markers.at<int>(x, y)] = 1 - FLabel[markers.at<int>(x, y)];
		return getImageColor();
	}

};

