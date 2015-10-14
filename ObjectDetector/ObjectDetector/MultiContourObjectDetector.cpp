#include "Utility.h"
#include "commonInclude.h"
#include "MultiContourObjectDetector.h"

using namespace std;
using namespace cv;

MultiContourObjectDetector::MultiContourObjectDetector(int minContourPoints, int aspectedContours) :
ObjectDetector(minContourPoints, aspectedContours)
{}


bool MultiContourObjectDetector::findBaseShape(cv::Mat& baseImage)
{
	vector<vector<vector<Point>>> approxContours = findApproxContours(baseImage, false, 150);

	if (approxContours.size() == 0)
	{
		cerr << "ERROR: No valid contours found in base image" << endl;
		return false;
	}

	for (int i = 0; i < approxContours.size(); i++)
	{
		if (approxContours[i].size() == _aspectedContours)
		{
			_baseShape = approxContours[i];
			break;
		}
	}
	return true;
}

std::vector<std::vector<std::vector<cv::Point>>> MultiContourObjectDetector::findApproxContours(
	cv::Mat image,
	bool performOpening,
	int minThresholdValue)
{
	Size imgSize = image.size();
	Mat gray(image.size(), CV_8UC1);
	Mat thresh(image.size(), CV_8UC1);

	cvtColor(image, gray, CV_BGR2GRAY);

	if (performOpening)
	{
		// PERFORM OPENING (Erosion --> Dilation)

		int erosion_size = 4;
		int dilation_size = 4;

		Mat element = getStructuringElement(0, Size(2 * erosion_size, 2 * erosion_size), Point(erosion_size, erosion_size));
		erode(gray, gray, element);
		dilate(gray, gray, element);
	}	

	threshold(gray, thresh, minThresholdValue, 255, THRESH_BINARY);

#ifdef DEBUG_MODE
	imshow("Threshold", thresh);
#endif

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<Point> approx;

	map<int, vector<vector<Point>>> hierachedContours;
	map<int, vector<vector<Point>>> approxHContours;


	try
	{
		findContours(thresh, contours, hierarchy, CV_RETR_TREE, CHAIN_APPROX_NONE);
	}
	catch (const Exception& e)
	{
		cerr << e.what();
	}


#ifdef DEBUG_MODE
	Mat tempI(image.size(), CV_8UC1);
	tempI = Scalar(0);
	drawContours(tempI, contours, -1, cv::Scalar(255), 1, CV_AA);

	imshow("Contours", tempI);
#endif


	vector<vector<Point>> temp;
	// CATALOG BY HIERARCHY LOOP
	for (int i = 0; i < contours.size(); i++)
	{
		tempI = Scalar(0);
		temp.clear();
		temp.push_back(contours[i]);
		drawContours(tempI, temp, -1, cv::Scalar(255), 1, CV_AA);

		int parent = hierarchy[i][3];
		if (parent == -1)
		{
			if (hierachedContours.count(i) == 0)
			{
				// me not found

				hierachedContours.insert(pair<int, vector<vector<Point>>>(i, vector<vector<Point>>()));
				hierachedContours[i].push_back(contours[i]);
			}
			else
			{
				// me found
				continue;
			}
		}
		else
		{
			if (hierachedContours.count(parent) == 0)
			{
				// dad not found
				hierachedContours.insert(pair<int, vector<vector<Point>>>(parent, vector<vector<Point>>()));
				hierachedContours[parent].push_back(contours[parent]);
			}
			hierachedContours[parent].push_back(contours[i]);
		}
	}


	// APPROX LOOP

	for (map<int, vector<vector<Point>>>::iterator it = hierachedContours.begin(); it != hierachedContours.end(); it++)
	{

#ifdef DEBUG_MODE
		tempI = Scalar(0);
		drawContours(tempI, it->second, -1, cv::Scalar(255), 1, CV_AA);
#endif

		for (int k = 0; k < it->second.size(); k++)
		{
			if (it->second[k].size() < 4)
			{
				if (k == 0) // padre
					break;
				else        // figlio
					continue;
			}

			double epsilon = it->second[k].size() * 0.03;
			approxPolyDP(it->second[k], approx, epsilon, true);

#ifdef DEBUG_MODE			
			tempI = Scalar(0);
			vector<vector<Point>> temp;
			temp.push_back(approx);
			drawContours(tempI, temp, -1, cv::Scalar(255), 1, CV_AA);
#endif
			if (approx.size() < 4)
			{
				if (k == 0) // padre
					break;
				else        // figlio
					continue;
			}


			if (k == 0)
			{
				approxHContours.insert(pair<int, vector<vector<Point>>>(it->first, vector<vector<Point>>()));
				approxHContours.at(it->first).push_back(approx);
			}
			else
			{
				approxHContours[it->first].push_back(approx);
			}
		}
	}


	vector<vector<vector<Point>>> lookupVector;
	for (map<int, vector<vector<Point>>>::iterator it = approxHContours.begin(); it != approxHContours.end(); it++)
	{
		if (it->second.size() <= 1)
			continue;
		lookupVector.push_back(it->second);
	}

	return lookupVector;
}

std::vector<std::vector<std::vector<cv::Point>>> MultiContourObjectDetector::processContours(
	std::vector<std::vector<std::vector<cv::Point>>> approxContours,
	double hammingThreshold,
	double correlationThreshold)
{
	Utility utility;

	vector<vector<vector<Point>>> objects;

	for (int i = 0; i < approxContours.size(); i++)
	{
		if (approxContours[i].size() != _baseShape.size())
			continue;

		double totCorrelation = 0,
			totHamming = 0;

		// C and H with external contour
		totCorrelation += utility.correlationWithBase(approxContours[i][0], _baseShape[0]);
		totHamming += utility.calculateContourPercentageCompatibility(approxContours[i][0], _baseShape[0]);

		// looking for the contour with the better cnetroids and shape match

		for (int j = 1; j < approxContours[i].size(); j++)
		{
			double maxCorrelation = numeric_limits<double>::min(),
				maxHamming = numeric_limits<double>::min();

			for (int k = 1; k < _baseShape.size(); k++)
			{
				maxCorrelation = max(maxCorrelation, utility.correlationWithBase(approxContours[i][j], _baseShape[k]));
				maxHamming = max(maxHamming, utility.calculateContourPercentageCompatibility(approxContours[i][j], _baseShape[k]));
			}

			totCorrelation += maxCorrelation;
			totHamming += maxHamming;
		}

		totCorrelation /= approxContours[i].size();
		totHamming /= approxContours[i].size();

		cout << "Middle Correlation " << to_string(i) << " with base ---> " << totCorrelation << endl;
		cout << "Middle Hamming distance" << to_string(i) << " with base ---> " << totHamming << endl;

		if (totCorrelation >= correlationThreshold && totHamming >= hammingThreshold)
			objects.push_back(approxContours[i]);
	}

	return objects;
}

cv::Mat MultiContourObjectDetector::generateDetectionMask(
	std::vector<std::vector<std::vector<cv::Point>>> detectedObjects,
	cv::Size imageSize,
	int type)
{
	/*
	Scalar base, pen;

	if (type == CV_8UC1)
	{
		base = Scalar(0);
		pen = Scalar(255);
	}
	else
	{
		base = Scalar(0, 0, 0);
		pen = Scalar(255, 255, 255);
	}
	*/

	Mat mask(imageSize, type);
	mask = Scalar(0);

	for (int i = 0; i < detectedObjects.size(); i++)
	{
		for (int j = 0; j < detectedObjects[i].size(); j++)
		{
			for (int k = 0; k < detectedObjects[i][j].size(); k++)
			{
				line(mask, detectedObjects[i][j][k], detectedObjects[i][j][(k + 1) % detectedObjects[i][j].size()], Scalar(255), 2, CV_AA);
			}
		}
	}

	return mask;
}