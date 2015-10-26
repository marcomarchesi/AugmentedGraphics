#include "ObjectDetector.h"
#include "Utility.h"
#include "commonInclude.h"


using namespace std;
using namespace cv;
using namespace od;

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
											OutputMaskMode maskMode,
											std::vector<std::vector<std::vector<cv::Point>>>* detectedContours,
											int* numberOfObject)
{
	vector<vector<vector<Point>>> approxContours = findApproxContours(image, true); //prima 60
	
	vector<vector<vector<Point>>> detectedObjects = processContours(approxContours, hammingThreshold, correlationThreshold, numberOfObject);

	cvtColor(image, image, CV_BGR2BGRA);
	Mat mask = generateDetectionMask(detectedObjects, image, maskMode);	

	*detectedContours = detectedObjects;

	Mat alpha(image.size(), CV_8UC4);
	bitwise_and(image, mask, alpha);

	return alpha;
}

cv::Mat ObjectDetector::generateDetectionMask(
	std::vector<std::vector<std::vector<cv::Point>>> detectedObjects,
	cv::Mat& image,
	OutputMaskMode maskMode)
{

	Scalar base, pen;

	if (image.type() == CV_8UC1)
	{
		base = Scalar(0);
		pen = Scalar(255);
	}
	else
	{
		base = Scalar(0, 0, 0, 0);
		pen = Scalar(255, 255, 255);
	}

	
	Mat mask(image.size(), image.type());
	mask = base;

	for (int i = 0; i < detectedObjects.size(); i++)
	{
		if (maskMode == OutputMaskMode::PRECISE)
		{
			vector<vector<Point>> preciseContours;

			for (int j = 0; j < detectedObjects[i].size(); j++)
			{
				Rect objectRect = boundingRect(detectedObjects[i][j]);
				objectRect.width += 2;
				objectRect.height += 2;
				objectRect.x -= 2;
				objectRect.y -= 2;

				Mat rect = image(objectRect);

				Mat gray(rect.size(), rect.type());
				cvtColor(rect, gray, CV_BGRA2GRAY);

				int minThreshold = mean(gray)[0];

				if (minThreshold < 90)
					minThreshold = 60;
				else if (minThreshold >= 90 && minThreshold < 125)
					minThreshold = 100;

				threshold(gray, gray, minThreshold, 255, THRESH_BINARY);

				//imshow("Thresh", thresh);

				vector<vector<Point>> contours;
				findContours(gray, contours, CV_RETR_LIST, CHAIN_APPROX_NONE);

#ifdef DEBUG_MODE
				gray = cv::Scalar(0);
				drawContours(gray, contours, -1, cv::Scalar(255), 1, CV_AA);
#endif

				vector<Point> maxC;
				int size = 0;

				for (int k = 0; k < contours.size(); k++)
				{

#ifdef DEBUG_MODE
					gray = cv::Scalar(0);
					vector<vector<Point>> temp;
					temp.push_back(contours[k]);
					drawContours(gray, temp, -1, cv::Scalar(255), 1, CV_AA);
#endif
					if (contours[k].size() > size)
					{
						vector<Point> approx;
						approxPolyDP(contours[k], approx, contours[k].size() * 0.03, true);
						if (approx.size() == 4)
							continue;

						size = contours[k].size();
						maxC = contours[k];
					}
				}

				for (int k = 0; k < maxC.size(); k++)
				{
					maxC[k].x += objectRect.x;
					maxC[k].y += objectRect.y;
				}

				preciseContours.push_back(maxC);
			}


			drawContours(mask, preciseContours, -1, pen, -1, CV_AA);
		}
		else if (maskMode == OutputMaskMode::CONVEX_HULL)
		{
			drawContours(mask, detectedObjects[i], -1, pen, -1, CV_AA);
		}	
	}

	return mask;
}