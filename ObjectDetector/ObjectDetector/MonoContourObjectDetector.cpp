#include "Utility.h"
#include "commonInclude.h"
#include "MonoContourObjectDetector.h"

using namespace std;
using namespace cv;

MonoContourObjectDetector::MonoContourObjectDetector(int minContourPoints, int aspectedContours) :
ObjectDetector(minContourPoints, aspectedContours)
{}

bool MonoContourObjectDetector::findBaseShape(cv::Mat& baseImage)
{
	vector<vector<vector<Point>>> compatibleContours = findApproxContours(baseImage, false);

	if (compatibleContours[0].size() == 0)
	{
		cerr << "ERROR: No valid contours found in the base image" << endl;
		return false;
	}

	_baseShape = compatibleContours[0][0];
	return true;
}

vector<vector<vector<Point>>> MonoContourObjectDetector::findApproxContours(
	cv::Mat image,
	bool performOpening)
{
	
	// CREATE ACTIVE ZONE 80% AND 50% ---------------------

	Point centre(image.size().width / 2, image.size().height / 2);

	int deleteHeight = image.size().height * _deleteFocus;
	int deleteWidth = image.size().width * _deleteFocus;
	int deleteX = centre.x - deleteWidth / 2;
	int deleteY = centre.y - deleteHeight / 2;

	int attenuationHeight = image.size().height * _attenuationFocus;
	int attenuationWidth = image.size().width * _attenuationFocus;
	int attenuationX = centre.x - attenuationWidth / 2;
	int attenuationY = centre.y - attenuationHeight / 2;

	Rect erase(deleteX, deleteY, deleteWidth, deleteHeight);
	_deleteRect = erase;

	Rect ease(attenuationX, attenuationY, attenuationWidth, attenuationHeight);
	_attenuationRect = ease;
	// ----------------------------------------

	Size imgSize = image.size();
	Mat gray(image.size(), CV_8UC1);
	Mat thresh(image.size(), CV_8UC1);

	if (image.channels() >= 3)
		cvtColor(image, gray, CV_BGR2GRAY);

	int minThreshold = mean(gray)[0];

	if (performOpening)
	{
		// PERFORM OPENING (Erosion --> Dilation)

		int erosion_size = 5;
		int dilation_size = 5;

		Mat element = getStructuringElement(0, Size(2 * erosion_size, 2 * erosion_size), Point(erosion_size, erosion_size));
		erode(gray, gray, element);
		dilate(gray, gray, element);

		minThreshold = mean(gray)[0];

		if (minThreshold < 90)
			minThreshold = 60;
		else if (minThreshold >= 90 && minThreshold < 125)
			minThreshold = 100;
	}

	

	threshold(gray, thresh, minThreshold, 255, THRESH_BINARY);

	//imshow("Thresh", thresh);

	vector<vector<Point>> contours;
	vector<Point> approx;
	findContours(thresh, contours, CV_RETR_LIST, CHAIN_APPROX_NONE);

#ifdef DEBUG_MODE
	Mat contoursImage(image.size(), CV_8UC1);
	contoursImage = cv::Scalar(0);
	drawContours(contoursImage, contours, -1, cv::Scalar(255), 1, CV_AA);
	imshow("ContoursImage", contoursImage);
#endif

	vector<vector<Point>> approxContours;

	for (int i = 0; i < contours.size(); i++)
	{

		if (contours[i].size() < 3)
			continue;

		convexHull(contours[i], approx, false);

		//double epsilon = contours[i].size() * 0.05;
		//approxPolyDP(contours[i], approx, epsilon, true);

#ifdef DEBUG_MODE			
		contoursImage = Scalar(0);
		vector<vector<Point>> temp;
		temp.push_back(approx);
		drawContours(contoursImage, temp, -1, cv::Scalar(255), 1, CV_AA);
#endif

		// REMOVE TOO EXTERNAL SHAPES -------------

		Moments m = moments(approx, true);
		int cx = int(m.m10 / m.m00);
		int cy = int(m.m01 / m.m00);

		Point c(cx, cy);

		if (!(c.x >= _deleteRect.x && 
			c.y >= _deleteRect.y &&
			c.x <= (_deleteRect.x + _deleteRect.width) &&
			c.y <= (_deleteRect.y + _deleteRect.height)))
			continue;
		// --------------------------------------------------

		int min, max;
		min = _minContourPoints - _minContourPoints / 1.5;
		max = _minContourPoints + _minContourPoints / 1.5;
		

		if (approx.size() >= min && approx.size() <= max)
			approxContours.push_back(approx);
	}

	vector<vector<vector<Point>>> retVector;
	retVector.push_back(approxContours);

	return retVector;

}


std::vector<std::vector<std::vector<cv::Point>>> MonoContourObjectDetector::processContours(
	std::vector<std::vector<std::vector<cv::Point>>> approxContours,
	double hammingThreshold,
	double correlationThreshold,
	int* numberOfObject)
{
#ifdef DEBUG_MODE
	//cout << "Contours founded: " << to_string(approxContours[0].size()) << endl;
#endif

	vector<vector<Point>> objects;
	Utility utility;

	// ID -- hamming
	vector<pair<int, double>> hammingValues;
	double attenuation = 0;

	for (int i = 0; i < approxContours[0].size(); i++)
	{
		attenuation = 0;

#ifdef DEBUG_MODE
		Mat tempImg(Size(1000, 1000), CV_8UC1);
		tempImg = cv::Scalar(0);

		vector<vector<Point>> vect;
		vect.push_back(approxContours[0][i]);

		drawContours(tempImg, vect, -1, cv::Scalar(255), 1, CV_AA);
		//imshow(to_string(i), tempImg);
#endif

		Moments m = moments(approxContours[0][i], true);
		int cx = int(m.m10 / m.m00);
		int cy = int(m.m01 / m.m00);

		Point c(cx, cy);

		if (!(c.x >= _attenuationRect.x &&
			c.y >= _attenuationRect.y &&
			c.x <= (_attenuationRect.x + _attenuationRect.width) &&
			c.y <= (_attenuationRect.y + _attenuationRect.height)))
			attenuation = 15;

		double correlation = utility.correlationWithBase(approxContours[0][i], _baseShape);

		if ((correlation-attenuation) < correlationThreshold)
			continue;

#ifdef DEBUG_MODE
		cout << "Correlation " << to_string(i) << " --- " << to_string(correlation) << endl;
#endif

		double hamming = utility.calculateContourPercentageCompatibility(approxContours[0][i], _baseShape);
		if (hamming == 0)
			continue;


#ifdef DEBUG_MODE
		cout << to_string(i) << " Contour Hamming Percentage " << " " << to_string(hamming-attenuation) << endl << endl;
#endif

		hammingValues.push_back(pair<int, double>(i, (hamming-attenuation)));

	}

	for (int i = 0; i < hammingValues.size(); i++)
	{
		if (hammingValues[i].second >= hammingThreshold && hammingValues[i].second != 100.0)
			objects.push_back(approxContours[0].at(hammingValues[i].first));
	}

#ifdef DEBUG_MODE
	cout << "Possible valid objects: " << to_string(objects.size()) << endl;
#endif


	*numberOfObject = (objects.size());
	
	vector<vector<vector<Point>>> retVector;
	retVector.push_back(objects);

	return retVector;
}


