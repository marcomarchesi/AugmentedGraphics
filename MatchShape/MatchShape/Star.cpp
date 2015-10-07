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
	cout << "Star shape Loaded" << endl;

#ifdef DEBUG_MODE
	showContour(starShape, starImage.size());
#endif

	centroids = findCentroidsDistribution(starShape);

	minFocus = 0.7;
	maxFocus = 0.7;
	focus = minFocus;
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

	// ---- SHAPE'S POINT FILTER ----
	/*
	if (contour.size() != 8)
		return 0.0;
	*/

	double hamming = matchShapes(contour, starShape, CV_CONTOURS_MATCH_I1, 0.0);

	return ((1 - hamming) * 100);
}


double Star::correlationWithBase(std::vector<cv::Point> distribution)
{
	if (distribution.size() != 9) return 0;

	// FIND MEANS
	Point meanDistr(0,0), meanBase(0,0);
	for (int i = 0; i < distribution.size(); i++)
	{
		meanDistr.x += distribution[i].x;
		meanDistr.y += distribution[i].y;

		meanBase.x += centroids[i].x;
		meanBase.y += centroids[i].y;
	}

	meanDistr.x /= distribution.size();
	meanDistr.y /= distribution.size();

	meanBase.x /= centroids.size();
	meanBase.y /= centroids.size();

	Point2d diffDistr, diffBase;

	Point2d product = 0;
	Point2d distr2 = 0;
	Point2d base2 = 0;

	for (int i = 0; i < distribution.size(); i++)
	{
		diffDistr.x = distribution[i].x - meanDistr.x;
		diffDistr.y = distribution[i].y - meanDistr.y;

		diffBase.x = centroids[i].x - meanBase.x;
		diffBase.y = centroids[i].y - meanBase.y;
		
		product.x += diffDistr.x * diffBase.x;
		product.y += diffDistr.y * diffBase.y;

		distr2.x += diffDistr.x * diffDistr.x;
		distr2.y += diffDistr.y * diffDistr.y;

		base2.x += diffBase.x * diffBase.x;
		base2.y += diffBase.y * diffBase.y;
	}

	Point2d correlation;
	correlation.x = product.x / sqrt(distr2.x * base2.x);
	correlation.y = product.y / sqrt(distr2.y * base2.y);

	return (correlation.x+correlation.y)/2;
	
}



// find all the star contour in contours extracted from an image
std::vector<std::vector<cv::Point>> Star::findStarsInContours(
	std::vector<std::vector<cv::Point>> contours,
	double precision,
	cv::Mat gray)
{

#ifdef DEBUG_MODE
	cout << "Contours founded: " << to_string(contours.size()) << endl;
#endif

	vector<vector<Point>> stars;	

	// ID -- hamming
	vector<pair<int, double>> hammingValues;
	
	for (int i = 0; i < contours.size(); i++)
	{
		vector<Point> distribution = findCentroidsDistribution(contours[i]);
		double correlation = correlationWithBase(distribution);

		if (correlation < 0.91)
			continue;

#ifdef DEBUG_MODE
		cout << "Correlation " << to_string(i) << " --- " << to_string(correlation) << endl;
#endif

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
		
		cout << to_string(i) << " Contour Hamming Percentage " << " " << to_string(hamming) << endl << endl;
#endif
		
		hammingValues.push_back(pair<int, double>(i, hamming));
		
	}
	
	for (int i = 0; i < hammingValues.size(); i++)
	{
		if (hammingValues[i].second >= precision && hammingValues[i].second != 100.0)
			stars.push_back(contours.at(hammingValues[i].first));
	}

#ifdef DEBUG_MODE
	cout << "Possible stars: " << to_string(stars.size()) << endl;
#endif

	// Now i have contours that have 8 points with valid hamming distance		
	// i want to remove contours whit low distance between point

	

	// CHECK MIN SIDE LENGTH (EMIPIRIC)
	/*
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
	*/

	return stars;
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

	// SAVE THE IMAGE
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

std::vector<cv::Point> Star::findCentroidsDistribution(std::vector<cv::Point> contour)
{
	vector<Point> retCentr;

	int minX = numeric_limits<int>::max(),
		maxX = numeric_limits<int>::min(),
		minY = numeric_limits<int>::max(),
		maxY = numeric_limits<int>::min();

	for (int i = 0; i < contour.size(); i++)
	{
		if (contour[i].x < minX)
			minX = contour[i].x;

		if (contour[i].x > maxX)
			maxX = contour[i].x;

		if (contour[i].y < minY)
			minY = contour[i].y;

		if (contour[i].y > maxY)
			maxY = contour[i].y;
	}

	Size size(maxX+minX, maxY+minY);

	Rect box = boundingRect(contour);
	
	if (box.width % 2 == 1)
	{
		box.width += 1;
		size.width += 1;
	}
		

	if (box.height % 2 == 1)
	{
		box.height += 1;
		size.height += 1;
	}

	Mat img(size, CV_8UC1);
	img = Scalar(0);
	vector<vector<Point>> tempVector;
	tempVector.push_back(contour);

	int x, y;

	
	// Vertical Split |
	for (x = box.x; x < (box.x + box.width); x += (box.width / 2))
	{
		y = box.y;

		Rect boxSplit(x, box.y, box.width / 2, box.height);
		img = Scalar(0);
		drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);
		Mat sub = img(boxSplit);

		vector<vector<Point>> shape;
		findContours(sub, shape, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

		if (shape.size() == 0) continue;
		if (shape.size() > 1)
		{
			for (int i = 0; i < shape.size(); i++)
			{
				if (shape[i].size() > shape[0].size())
					shape[0] = shape[i];
			}
		}

		Moments m = moments(shape[0], true);
		int cx = int(m.m10 / m.m00) + x;
		int cy = int(m.m01 / m.m00) + y;

		retCentr.push_back(Point(cx, cy));
		circle(img, Point(cx, cy), 5, Scalar(255), -1, 8, 0);

		// Sub Horizontal Split --
		for (y = box.y; y < (box.y + box.height); y += (box.height / 2))
		{
			boxSplit = Rect(x, y, box.width/2, box.height / 2);
			img = Scalar(0);
			drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);
			sub = img(boxSplit);

			vector<vector<Point>> shape;
			findContours(sub, shape, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

			if (shape.size() == 0) continue;
			if (shape.size() > 1)
			{
				for (int i = 0; i < shape.size(); i++)
				{
					if (shape[i].size() > shape[0].size())
						shape[0] = shape[i];
				}
			}

			m = moments(shape[0], true);
			int cx = int(m.m10 / m.m00) + x;
			int cy = int(m.m01 / m.m00) + y;

			retCentr.push_back(Point(cx, cy));
			circle(img, Point(cx, cy), 5, Scalar(255), -1, 8, 0);
		}		
	}
	
	

	// Horizontal Split --
	for (y = box.y; y < (box.y + box.height); y += (box.height / 2))
	{
		x = box.x;

		Rect boxSplit(box.x, y, box.width, box.height / 2);
		img = Scalar(0);
		drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);
		Mat sub = img(boxSplit);

		vector<vector<Point>> shape;
		findContours(sub, shape, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

		if (shape.size() == 0) continue;
		if (shape.size() > 1)
		{
			for (int i = 0; i < shape.size(); i++)
			{
				if (shape[i].size() > shape[0].size())
					shape[0] = shape[i];
			}
		}

		Moments m = moments(shape[0], true);
		int cx = int(m.m10 / m.m00) + x;
		int cy = int(m.m01 / m.m00) + y;

		retCentr.push_back(Point(cx, cy));
		circle(img, Point(cx, cy), 5, Scalar(255), -1, 8, 0);

		// Sub Vertical Split |
		/*
		for (x = box.x; x < (box.x + box.width); x += (box.width / 2))
		{
			boxSplit = Rect(x, y, box.width / 2, box.height/2);
			img = Scalar(0);
			drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);
			sub = img(boxSplit);

			vector<vector<Point>> shape;
			findContours(sub, shape, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

			if (shape.size() == 0) continue;
			if (shape.size() > 1)
			{
				for (int i = 0; i < shape.size(); i++)
				{
					if (shape[i].size() > shape[0].size())
						shape[0] = shape[i];
				}
			}

			m = moments(shape[0], true);
			int cx = int(m.m10 / m.m00) + x;
			int cy = int(m.m01 / m.m00) + y;

			retCentr.push_back(Point(cx, cy));
			circle(img, Point(cx, cy), 5, Scalar(255), -1, 8, 0);
		}		
		*/
	}

	Moments m = moments(contour, true);
	int cx = int(m.m10 / m.m00);
	int cy = int(m.m01 / m.m00);

	retCentr.push_back(Point(cx, cy));
	
	img = Scalar(0);
	drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);

	for (int i = 0; i < retCentr.size(); i++)
		circle(img,retCentr[i], 5, Scalar(255), -1, 8, 0);

	/*
#ifdef DEBUG_MODE
	imshow("Centroidi", img);
#endif
	*/

	return retCentr;
}

