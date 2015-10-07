#include "GenericObject.h"
#include "Utility.h"

#include <iostream>
#include <vector>


//#define DEBUG_MODE

using namespace cv;
using namespace std;

GenericObject::GenericObject(cv::Mat baseImage) : baseSize(baseImage.size())
{
	if (baseImage.empty())
	{
		cout << "the Base Image is empty" << endl;
		waitKey(30);
		exit(1);
	}

	findObjectShape(baseImage);
	cout << "Star shape Loaded" << endl;

#ifdef DEBUG_MODE
	showContour(baseShape, baseSize);
#endif

	minFocus = 0.7;
	maxFocus = 0.7;
	focus = minFocus;
}


std::vector<std::vector<cv::Point>> GenericObject::findObjectsInContours(std::vector<std::vector<cv::Point>> contours,
																			double precision,
																			cv::Mat gray)
{

#ifdef DEBUG_MODE
	cout << "Contours founded: " << to_string(contours.size()) << endl;
#endif

	vector<vector<Point>> objects;
	Utility utility;

	// ID -- hamming
	vector<pair<int, double>> hammingValues;

	for (int i = 0; i < contours.size(); i++)
	{
		double correlation = utility.correlationWithBase(contours[i], baseShape);

		if (correlation < 0.91)
			continue;

#ifdef DEBUG_MODE
		cout << "Correlation " << to_string(i) << " --- " << to_string(correlation) << endl;
#endif

		double hamming = utility.calculateContourPercentageCompatibility(contours[i], baseShape);
		if (hamming == 0)
			continue;


#ifdef DEBUG_MODE

		Mat tempImg(Size(640, 480), CV_8UC1);
		tempImg = cv::Scalar(0);

		vector<vector<Point>> vect;
		vect.push_back(contours[i]);

		drawContours(tempImg, vect, -1, cv::Scalar(255), 1, CV_AA);
		imshow(to_string(i), tempImg);

		cout << to_string(i) << " Contour Hamming Percentage " << " " << to_string(hamming) << endl << endl;
#endif

		hammingValues.push_back(pair<int, double>(i, hamming));

	}

	for (int i = 0; i < hammingValues.size(); i++)
	{
		if (hammingValues[i].second >= precision && hammingValues[i].second != 100.0)
			objects.push_back(contours.at(hammingValues[i].first));
	}

#ifdef DEBUG_MODE
	cout << "Possible valid objects: " << to_string(stars.size()) << endl;
#endif
		
	return objects;
}


cv::Mat GenericObject::findObjectsInImg(cv::Mat img, double precision){

	if (img.size().height > 800 || img.size().width > 800)
	{
		Size s = img.size(), small;
		small.height = s.height / 5;
		small.width = s.width / 5;

		resize(img, img, small);
	}

	// ---- CREATE FOCUS RECT ----

#ifdef DEBUG_MODE
	cout << "focus: " << to_string(focus) << endl;
#endif

	Point centre(img.size().width / 2, img.size().height / 2);

	int focusHeight = img.size().height * focus;
	int focusWidth = img.size().width * focus;
	int x = centre.x - focusWidth / 2;
	int y = centre.y - focusHeight / 2;

	Rect focusRect(x, y, focusWidth, focusHeight);

	// -----------------------------------------------------------

	imgSize = img.size();
	Mat gray(img.size(), CV_8UC1);
	Mat thresh(img.size(), CV_8UC1);

	cvtColor(img, gray, CV_BGR2GRAY);

	// PERFORM OPENING (Erosion --> Dilation)

	int erosion_size = 5;
	int dilation_size = 5;

	Mat element = getStructuringElement(0, Size(2 * erosion_size, 2 * erosion_size), Point(erosion_size, erosion_size));
	erode(gray, gray, element);
	dilate(gray, gray, element);

	// need a loop that decreese the threshold min value if the image is too black
	// for now 60 is ok
	threshold(gray, thresh, 60, 255, THRESH_BINARY);

	//imshow("Thresh", thresh);

	vector<vector<Point>> contours;
	vector<Point> approx;
	findContours(thresh, contours, CV_RETR_LIST, CHAIN_APPROX_NONE);

	vector<vector<Point>> approxContours;

	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() < 8)
			continue;

		double epsilon = contours[i].size() * 0.04;
		approxPolyDP(contours[i], approx, epsilon, true);

		Rect box = boundingRect(approx);

		if (box.x >= focusRect.x &&
			box.y >= focusRect.y &&
			(box.x + box.width) <= (focusRect.x + focusRect.width) &&
			(box.y + box.height) <= focusRect.y + focusRect.height)
		{
			approxContours.push_back(approx);
		}
	}

#ifdef DEBUG_MODE
	Mat approxContoursImage(img.size(), CV_8UC1);
	approxContoursImage = cv::Scalar(0);
	drawContours(approxContoursImage, approxContours, -1, cv::Scalar(255), 1, CV_AA);
	imshow("ApproxContoursImage", approxContoursImage);
#endif

	vector<vector<Point>> stars = findObjectsInContours(approxContours, precision, gray);

	Mat starMask(gray.size(), gray.type());
	starMask = Scalar(0);

	for (int i = 0; i < stars.size(); i++)
	{
		for (int j = 0; j < stars[i].size(); j++)
		{
			line(starMask, stars[i][j], stars[i][(j + 1) % stars[i].size()], Scalar(255), 2, CV_AA);
		}
	}
	rectangle(gray, focusRect, Scalar(255), 2, CV_AA);

	/*
	if (stars.size() == 0)
	{
	if (focus < maxFocus)
	focus += 0.1;
	else
	focus = minFocus;
	}
	*/

	gray += starMask;

	return gray;
}



void GenericObject::findObjectShape(cv::Mat starImage){

	Mat thresh(starImage.size(), CV_8UC1);
	cvtColor(starImage, thresh, CV_BGR2GRAY);

	threshold(thresh, thresh, 127, 255, THRESH_BINARY);

	vector<vector<Point>> contours;
	vector<Point> approx;
	findContours(thresh, contours, CV_RETR_LIST, CHAIN_APPROX_NONE);

	vector<vector<Point>> conpatibleContours;

	for (int i = 0; i < contours.size(); i++)
	{
		if (contours[i].size() < 5)
			continue;

		double epsilon = contours[i].size() * 0.05;
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

	baseShape = conpatibleContours[0];

}


void GenericObject::showContour(std::vector<cv::Point> contour, cv::Size size)
{
	Mat contourImage(size, CV_8UC1);
	contourImage = Scalar(0);

	vector<vector<Point>> tempVector;
	tempVector.push_back(contour);
	drawContours(contourImage, tempVector, -1, cv::Scalar(255), 1, CV_AA);
	imshow("Contour", contourImage);
}