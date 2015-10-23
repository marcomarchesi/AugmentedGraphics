#include "ObjectDetector.h"
#include "Utility.h"
#include "commonInclude.h"

using namespace std;
using namespace cv;

ObjectDetector::ObjectDetector(int minContourPoints, int aspectedContours):
	_minContourPoints(minContourPoints),
	_aspectedContours(aspectedContours),
	_deleteFocus(0.80),
	_attenuationFocus(0.50)
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


cv::Mat ObjectDetector::findObjectsInImage(cv::Mat& image,
											double hammingThreshold,
											double correlationThreshold,
											std::vector<std::vector<std::vector<cv::Point>>>* detectedContours,
											int* numberOfObject)
{
	vector<vector<vector<Point>>> approxContours = findApproxContours(image, true); //prima 60
	
	vector<vector<vector<Point>>> detectedObjects = processContours(approxContours, hammingThreshold, correlationThreshold, numberOfObject);

	cvtColor(image, image, CV_BGR2BGRA);
	Mat mask = generateDetectionMask(detectedObjects, image.size(), image.type());	

	*detectedContours = detectedObjects;

	Mat alpha(image.size(), CV_8UC4);
	bitwise_and(image, mask, alpha);

	return alpha;
}

cv::Mat ObjectDetector::generateDetectionMask(
	std::vector<std::vector<std::vector<cv::Point>>> detectedObjects,
	cv::Size imageSize,
	int type)
{

	Scalar base, pen;

	if (type == CV_8UC1)
	{
		base = Scalar(0);
		pen = Scalar(255);
	}
	else
	{
		base = Scalar(0, 0, 0, 0);
		pen = Scalar(255, 255, 255);
	}


	Mat mask(imageSize, type);
	mask = base;

	for (int i = 0; i < detectedObjects.size(); i++)
	{
		/*for (int j = 0; j < detectedObjects[i].size(); j++)
		{
			for (int k = 0; k < detectedObjects[i][j].size(); k++)
			{
				line(mask, detectedObjects[i][j][k], detectedObjects[i][j][(k + 1) % detectedObjects[i][j].size()], pen, 2, CV_AA);
			}
		}*/

		drawContours(mask, detectedObjects[i], -1, pen, -1, CV_AA);
	}

	return mask;
}