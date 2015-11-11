#include "Utility.h"
#include "commonInclude.h"
#include "MonoContourObjectDetector.h"




using namespace std;
using namespace cv;
using namespace od;

MonoContourObjectDetector::MonoContourObjectDetector(int minContourPoints, int aspectedContours) :
ObjectDetector(minContourPoints, aspectedContours)
{}

bool MonoContourObjectDetector::findBaseShape(cv::Mat& baseImage)
{
	vector<vector<vector<Point>>> compatibleContours = findApproxContours(baseImage, true);

	if (compatibleContours[0].size() == 0)
	{
		cerr << "ERROR: No valid contours found in the base image" << endl;
		return false;
	}

	int minID = 0;
	int minDist = numeric_limits<int>::max();

	for (int i = 0; i < compatibleContours[0].size(); i++)
	{
		if (abs(_minContourPoints - compatibleContours[0][i].size()) < minDist)
		{
			minID = i;
			minDist = _minContourPoints - compatibleContours[0][i].size();
		}
	}

	_baseShape = compatibleContours[0][minID];
	
	Utility::findCentroidsKeypoints(_baseShape, _baseKeypoints, Utility::CentroidDetectionMode::THREE_LOOP);

	//_originalBaseShape = _originalQueryShapes[minID];
	//_originalQueryShapes.clear();

	return true;
}

vector<vector<vector<Point>>> MonoContourObjectDetector::findApproxContours(
	cv::Mat image,
	bool performOpening)
{
	
	// CREATE ACTIVE ZONE 80% AND 50% ---------------------

	Point center(image.size().width / 2, image.size().height / 2);

	int deleteHeight = image.size().height * _deleteFocus;
	int deleteWidth = image.size().width * _deleteFocus;
	int deleteX = center.x - deleteWidth / 2;
	int deleteY = center.y - deleteHeight / 2;

	int attenuationHeight = image.size().height * _attenuationFocus;
	int attenuationWidth = image.size().width * _attenuationFocus;
	int attenuationX = center.x - attenuationWidth / 2;
	int attenuationY = center.y - attenuationHeight / 2;

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


	Mat roi = thresh(_deleteRect);

	vector<vector<Point>> contours;
	vector<Point> approx, hull;
	findContours(roi, contours, CV_RETR_LIST, CHAIN_APPROX_NONE);

#ifdef DEBUG_MODE
	Mat contoursImage(roi.size(), CV_8UC1);
	contoursImage = cv::Scalar(0);
	drawContours(contoursImage, contours, -1, cv::Scalar(255), 1, CV_AA);
	imshow("ContoursImage", contoursImage);
#endif

	vector<vector<Point>>tempQueryShapes;
	vector<vector<vector<Point>>> retVector;

	boost::thread_group threadGroup;
	
	boost::container::vector<boost::container::vector<boost::container::vector<cv::Point>>> threadVector;

	boost::container::vector<boost::container::vector<cv::Point>> originalQueryShapes;
	threadVector.push_back(originalQueryShapes);

	for (int i = 0; i < contours.size(); i++)
	{
		/*
		if (contours[i].size() < _minContourPoints)
			continue;

		convexHull(contours[i], hull, false);

		double epsilon = contours[i].size() * 0.003;
		approxPolyDP(contours[i], approx, epsilon, true);


		// REMOVE TOO EXTERNAL SHAPES -------- PERFORM ROI -----
		
//		Moments m = moments(hull, true);
//		int cx = int(m.m10 / m.m00);
//		int cy = int(m.m01 / m.m00);
//
//		Point c(cx, cy);
//
//		if (!(c.x >= _deleteRect.x && 
//			c.y >= _deleteRect.y &&
//			c.x <= (_deleteRect.x + _deleteRect.width) &&
//			c.y <= (_deleteRect.y + _deleteRect.height)))
//			continue;
//		
//
//		Rect bounding = boundingRect(contours[i]);
//
//#ifdef DEBUG_MODE
//		rectangle(contoursImage, _deleteRect, Scalar(255));
//		rectangle(contoursImage, bounding, Scalar(255));
//#endif
//		
//
//		bool isInternal = bounding.x > _deleteRect.x &&
//			bounding.y > _deleteRect.y &&
//			bounding.x + bounding.width < _deleteRect.x + _deleteRect.width &&
//			bounding.y + bounding.height < _deleteRect.y + _deleteRect.height;	
//
//
//		if (!isInternal)
//			continue;
		
		// --------------------------------------------------

		int min, max;
		min = _minContourPoints - _minContourPoints / 2.4;
		max = _minContourPoints + _minContourPoints / 2.4;
		

		if (hull.size() >= min && hull.size() <= max)
		{
			approxContours.push_back(hull);
			originalQueryShapes.push_back(approx);
		}
		*/

		boost::thread filterThread;
		
		filterThread = boost::thread(&od::MonoContourObjectDetector::getValidContours,
											this,
											contours[i],
											&threadVector,
											_minContourPoints);

		threadGroup.add_thread(&filterThread);
	}
	threadGroup.join_all();


	for (int i = 0; i < threadVector[0].size(); i++)
	{
		vector<Point> temp1;
		for (int j = 0; j < threadVector[0][i].size(); j++)
		{
			temp1.push_back(threadVector[0][i][j]);
		}
		tempQueryShapes.push_back(temp1);		
	}
	retVector.push_back(tempQueryShapes);


#ifdef DEBUG_MODE			
	contoursImage = Scalar(0);
	drawContours(contoursImage, retVector, -1, cv::Scalar(255), 1, CV_AA);
	imshow("ApproxContours", contoursImage);
#endif

	
	

	return retVector;
}


void MonoContourObjectDetector::getValidContours(std::vector<cv::Point> contours,
	boost::container::vector<boost::container::vector<boost::container::vector<cv::Point>>> *threadVector,
	int minContourPoints)
{
	(*threadVector)[0].clear();

	vector<Point> approx, hull;
	//vector<vector<Point>> originalQueryShapes;

	if (contours.size() < minContourPoints)
		return;

	convexHull(contours, hull, false);

	double epsilon = contours.size() * 0.003;
	approxPolyDP(contours, approx, epsilon, true);


	// REMOVE TOO EXTERNAL SHAPES -------- PERFORM ROI -----

	//		Moments m = moments(hull, true);
	//		int cx = int(m.m10 / m.m00);
	//		int cy = int(m.m01 / m.m00);
	//
	//		Point c(cx, cy);
	//
	//		if (!(c.x >= _deleteRect.x && 
	//			c.y >= _deleteRect.y &&
	//			c.x <= (_deleteRect.x + _deleteRect.width) &&
	//			c.y <= (_deleteRect.y + _deleteRect.height)))
	//			continue;
	//		
	//
	//		Rect bounding = boundingRect(contours[i]);
	//
	//#ifdef DEBUG_MODE
	//		rectangle(contoursImage, _deleteRect, Scalar(255));
	//		rectangle(contoursImage, bounding, Scalar(255));
	//#endif
	//		
	//
	//		bool isInternal = bounding.x > _deleteRect.x &&
	//			bounding.y > _deleteRect.y &&
	//			bounding.x + bounding.width < _deleteRect.x + _deleteRect.width &&
	//			bounding.y + bounding.height < _deleteRect.y + _deleteRect.height;	
	//
	//
	//		if (!isInternal)
	//			continue;

	// --------------------------------------------------

	int min, max;
	min = minContourPoints - minContourPoints / 2.4;
	max = minContourPoints + minContourPoints / 2.4;


	if (hull.size() >= min && hull.size() <= max)
	{
		boost::container::vector<Point> boostApprox;
		for (int i = 0; i < approx.size(); i++)
			boostApprox.push_back(approx[i]);

		(*threadVector)[0].push_back(boostApprox);
	}
	
	return;
}

std::vector<std::vector<std::vector<cv::Point>>> MonoContourObjectDetector::processContours(
	std::vector<std::vector<std::vector<cv::Point>>> approxContours,
	double hammingThreshold,
	double correlationThreshold,
	int* numberOfObject)
{

	vector<vector<Point>> objects;
	
	
	double attenuation = 0;

	for (int i = 0; i < approxContours[0].size(); i++)
	{
		attenuation = 0;
		
		Moments m = moments(approxContours[0][i], true);
		int cx = int(m.m10 / m.m00);
		int cy = int(m.m01 / m.m00);

		Point c(cx, cy);

		if (!(c.x >= _attenuationRect.x &&
			c.y >= _attenuationRect.y &&
			c.x <= (_attenuationRect.x + _attenuationRect.width) &&
			c.y <= (_attenuationRect.y + _attenuationRect.height)))
			attenuation = 10;


		double hamming = Utility::calculateContourPercentageCompatibility(approxContours[0][i], _baseShape);		

		vector<Point> contourKeypoints;
		Utility::findCentroidsKeypoints(approxContours[0][i], contourKeypoints, Utility::CentroidDetectionMode::THREE_LOOP);

		double correlation = Utility::correlationWithBase(contourKeypoints, _baseKeypoints);

	
//#ifdef DEBUG_MODE
		cout << to_string(i) << " Contour Hamming Percentage " << " " << to_string(hamming - attenuation) << endl;
		cout << "Correlation " << to_string(i) << " --- " << to_string(correlation) << endl << endl;
//#endif
		
		if ((hamming - attenuation) < hammingThreshold)
			continue;

		if ((correlation - attenuation) < correlationThreshold)
			continue;

		
		objects.push_back(approxContours[0][i]);

	}

	*numberOfObject = (objects.size());
	
	vector<vector<vector<Point>>> retVector;
	retVector.push_back(objects);

	return retVector;
}


