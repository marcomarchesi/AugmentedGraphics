#include "ObjectDetector.h"
#include "Utility.h"
#include "commonInclude.h"

using namespace std;
using namespace cv;

ObjectDetector::ObjectDetector(int minContourPoints, int aspectedContours):
	_minContourPoints(minContourPoints),
	_aspectedContours(aspectedContours)
{}


bool ObjectDetector::loadImage(cv::Mat& baseImage)
{
	if (baseImage.empty())
	{
		cerr << "The base image is empty";
		return false;
	}

	return findBaseShape(baseImage);
}


cv::Mat ObjectDetector::findObjectsInImage(cv::Mat& image, double hammingThreshold, double correlationThreshold)
{
	if (image.size().height > 800 || image.size().width > 800)
	{
		Size s = image.size(), small;
		small.height = s.height / 3;
		small.width = s.width / 3;

		resize(image, image, small);
	}
	
	vector<vector<vector<Point>>> approxContours = findApproxContours(image, true, 60);
	
	vector<vector<vector<Point>>> detectedObjects = processContours(approxContours, hammingThreshold, correlationThreshold);

	Mat mask = generateDetectionMask(detectedObjects, image.size(), CV_8UC1);

	Mat gray(image.size(), CV_8UC1);
	cvtColor(image, gray, CV_BGR2GRAY);

	gray += mask;

	return gray;
}