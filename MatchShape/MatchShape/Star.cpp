#include "Star.h"

#include <iostream>
#include <vector>
#include <time.h>
#include <stdlib.h>

//#define DEBUG_MODE

using namespace cv;
using namespace std;

/* --- PUBLIC --- */

// use the findStarShape to isolate the star in the Base image
Star::Star(cv::Mat starImage) : thresholdMinValue(127), epsilonFactor(0.05), baseSize(starImage.size())
{
	if (starImage.empty())
	{
		cout << "the Base Image is empty" << endl;
		waitKey(30);
		exit(1);
	}

	findStarShape(starImage);
	cout << "Star shape Loaded";

#ifdef DEBUG_MODE
	showContour(starShape, starImage.size());
#endif
}

// calculate the compatibility between a contour and the base starShape
double Star::checkContourPercentageCompatibility(std::vector<cv::Point> contour)
{
	
#ifdef DEBUG_MODE
	Mat tempImg(Size(640, 480), CV_8UC1);
	tempImg = cv::Scalar(0);

	vector<vector<Point>> vect;
	vect.push_back(contour);

	drawContours(tempImg, vect, -1, cv::Scalar(255), 1, CV_AA);
#endif

	if (contour.size() != 8)
		return 0.0;

	double hamming = matchShapes(contour, starShape, CV_CONTOURS_MATCH_I1, 0.0);

	return ((1 - hamming) * 100);
}

// find all the star contour in contours extracted from an image
std::vector<std::vector<cv::Point>> Star::findStarsInContours(
	std::vector<std::vector<cv::Point>> contours,
	double precision,
	cv::Mat gray)
{
	vector<vector<Point>> stars;	

	// ID -- hamming
	vector<pair<int, double>> hammingValues;
	
	for (int i = 0; i < contours.size(); i++)
	{
		double hamming = checkContourPercentageCompatibility(contours[i]);

		if (hamming == 0)
			continue;
		
#ifdef DEBUG_MODE
		Mat tempImg(Size(640, 480), CV_8UC1);
		tempImg = cv::Scalar(0);

		vector<vector<Point>> vect;
		vect.push_back(contours[i]);

		drawContours(tempImg, vect, -1, cv::Scalar(255), 1, CV_AA);
		imshow(to_string(i), tempImg);

		cout << "Hamming Percentage " << to_string(i) << " " << to_string(hamming) << endl;
#endif
		
		hammingValues.push_back(pair<int, double>(i, hamming));
		
	}
	
	for (int i = 0; i < hammingValues.size(); i++)
	{
		if (hammingValues[i].second >= precision && hammingValues[i].second != 100.0)
			stars.push_back(contours.at(hammingValues[i].first));
	}
	
	// Now i have contours that have 8 points with valid hamming distance
		
	// i want to remove contours whit low distance between point
	// and contours that, in the front view has bad hamming whit base


	vector<vector<Point>> starsFiltred;

	for (int i = 0; i < stars.size(); i++)
	{
		float minDist = numeric_limits<float>::max();

		for (int j = 0; j < stars[i].size(); j++)
		{
			Point side = stars[i][j] - stars[i][(j + 1) % stars[i].size()];
			float length = side.dot(side);
			minDist = min(minDist, length);
		}

		if (minDist > 100)
			starsFiltred.push_back(stars[i]);
	}
	

	return starsFiltred;
}


// do all the work, find the stars in the input image
// return the same image in grayscale with stars marked with a white line
cv::Mat Star::findStarInImg(cv::Mat img, double precision)
{

	if (img.size().height > 800 || img.size().width > 800)
	{
		Size s = img.size(), small;
		small.height = s.height / 5;
		small.width = s.width / 5;

		resize(img, img, small);
	}

	imgSize = img.size();
	Mat gray(img.size(), CV_8UC1);
	Mat thresh(img.size(), CV_8UC1);

	cvtColor(img, gray, CV_BGR2GRAY);

	// need a loop that decreese the threshold min value if the image is too black
	// for now 60 is ok
	threshold(gray, thresh, 60, 255, THRESH_BINARY);

	vector<vector<Point>> contours;
	vector<Point> approx;
	findContours(thresh, contours, CV_RETR_LIST, CHAIN_APPROX_NONE);

	vector<vector<Point>> approxContours;

	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() < 5)
			continue;

		
		double epsilon = contours[i].size() * 0.04;
		approxPolyDP(contours[i], approx, epsilon, true);		

		
		approxContours.push_back(approx);
	}

#ifdef DEBUG_MODE
	Mat approxContoursImage(img.size(), CV_8UC1);
	approxContoursImage = cv::Scalar(0);
	drawContours(approxContoursImage, approxContours, -1, cv::Scalar(255), 1, CV_AA);
	imshow("ApproxContoursImage", approxContoursImage);
#endif

	vector<vector<Point>> stars = findStarsInContours(approxContours, precision, gray);
	
	Mat starMask(gray.size(), gray.type());
	starMask = Scalar(0);

	for (int i = 0; i < stars.size(); i++)
	{
		for (int j = 0; j < stars[i].size(); j++)
		{
			line(starMask, stars[i][j], stars[i][(j + 1) % stars[i].size()], Scalar(255), 2, CV_AA);
		}
	}
	
	gray += starMask;
	/*
#ifdef DEBUG_MODE	
	srand(time(NULL));
	int id = rand();
	imwrite("FindStarContours_" + to_string(id), gray);
#endif
	*/
	return gray;
}


/* --- PRIVATE --- */

//find star shape in the base image
void Star::findStarShape(cv::Mat starImage)
{
	Mat thresh(starImage.size(), CV_8UC1);
	cvtColor(starImage, thresh, CV_BGR2GRAY);	

	threshold(thresh, thresh, thresholdMinValue, 255, THRESH_BINARY);	

	vector<vector<Point>> contours;
	vector<Point> approx;
	findContours(thresh, contours, CV_RETR_LIST, CHAIN_APPROX_NONE);

	vector<vector<Point>> conpatibleContours;

	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() < 5)
			continue;

		double epsilon = contours[i].size() * epsilonFactor;
		approxPolyDP(contours[i], approx, epsilon, true);

		if (approx.size() != 8)
			continue;

		conpatibleContours.push_back(approx);
	}

	if (conpatibleContours.size() == 0)
	{
		cout << "ERROR: No valid contours found in star base image" << endl;
		waitKey(30);
		exit(2);
	}

	starShape = conpatibleContours[0];
}

// display a contour in an image, used in DEBUG_MODE
void Star::showContour(std::vector<cv::Point> contour, cv::Size size)
{
	Mat contourImage(size, CV_8UC1);
	contourImage = Scalar(0);

	vector<vector<Point>> tempVector;
	tempVector.push_back(contour);
	drawContours(contourImage, tempVector, -1, cv::Scalar(255), 1, CV_AA);
	imshow("Contour", contourImage);
}