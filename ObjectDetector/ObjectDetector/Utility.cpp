#include "Utility.h"
#include "commonInclude.h"
#include <math.h>

using namespace cv;
using namespace std;
using namespace od;

double Utility::calculateContourPercentageCompatibility(std::vector<cv::Point> contour, std::vector<cv::Point> base)
{

#ifdef DEBUG_MODE
	Mat tempImg(Size(640, 480), CV_8UC1);
	tempImg = cv::Scalar(0);

	vector<vector<Point>> vect;
	vect.push_back(contour);

	drawContours(tempImg, vect, -1, cv::Scalar(255), 1, CV_AA);
#endif


	//double hamming1 = matchShapes(contour, base, CV_CONTOURS_MATCH_I1, 0.0);
	//double hamming2 = matchShapes(contour, base, CV_CONTOURS_MATCH_I2, 0.0);
	double hamming3 = matchShapes(contour, base, CV_CONTOURS_MATCH_I3, 0.0);

	return ((1 - hamming3) * 100);
}

double Utility::correlationWithBase(std::vector<cv::Point> contour, std::vector<cv::Point> baseContour){

	/*
	vector<Point> distribution = Utility::findLargeCentroidsDistribution(contour);

	vector<Point> base = Utility::findLargeCentroidsDistribution(baseContour);

	
	if (base.size() != 23){
		cout << "FATAL ERROR, number of centroids of the base shape is wrong," << endl << "control the base image or aspectedContourPoint!!!!" << endl;
		return 0;
	}
	*/

	vector<KeyPoint> contourK, baseK;
	findCentroidsKeypoints(contour, contourK, CentroidDetectionMode::TWO_LOOP);
	findCentroidsKeypoints(baseContour, baseK, CentroidDetectionMode::TWO_LOOP);
		
	
	if (contourK.size() != baseK.size())
		return 0;


	vector<Point> c, b;
	for (int i = 0; i < contourK.size(); i++)
	{ 
		c.push_back(contourK[i].pt);
		b.push_back(baseK[i].pt);
	}

	double hausdorff = calculateHausdorffDistance(c, b);

	double centroidsCorrelation = spearmanCorrelation(c, b);	
	double distancesCorrelation = singleSpearmanCorrelation(findDistancesFromCenter(c), findDistancesFromCenter(b));
	double anglesCorrelation = singleSpearmanCorrelation(findAnglesRespectCenter(c), findAnglesRespectCenter(b));
	
	
	double correlation = (centroidsCorrelation + distancesCorrelation + anglesCorrelation) / 3;


	return correlation;
	
}

double Utility::correlationWithBaseMatcher(std::vector<cv::Point> contour, std::vector<cv::Point> baseContour)
{
	vector<KeyPoint> contourK, baseK;
	findCentroidsKeypoints(contour, contourK, CentroidDetectionMode::TWO_LOOP);
	findCentroidsKeypoints(baseContour, baseK, CentroidDetectionMode::TWO_LOOP);

	
	Rect cRect = boundingRect(contour),
		bRect = boundingRect(baseContour);

	Mat img(Size(cRect.x + cRect.width, cRect.y + cRect.height), CV_8UC1),
		sample(Size(bRect.x + bRect.width, bRect.y + bRect.height), CV_8UC1);

	img = Scalar(0);
	sample = Scalar(0);

	vector<vector<Point>> cVect, bVect;
	cVect.push_back(contour);
	bVect.push_back(baseContour);

	drawContours(img, cVect, -1, cv::Scalar(255), -1, CV_AA);
	drawContours(sample, bVect, -1, cv::Scalar(255), -1, CV_AA);
	

	//Mat img = queryImage;
	//Mat sample = baseImage;

	Mat contourDescriptors, baseDescriptors;
	
	Ptr<ORB> ex = ORB::create();
	ex->compute(img, contourK, contourDescriptors);
	ex->compute(sample, baseK, baseDescriptors);
	//ex->detectAndCompute(img, noArray(), contourK, contourDescriptors, false);
	//ex->detectAndCompute(sample, noArray(), baseK, baseDescriptors, false);


	BFMatcher bruteForce;
	vector<DMatch> matches;
	bruteForce.match(contourDescriptors, baseDescriptors, matches);
	

	Mat img_matches;
	drawMatches(img, contourK, sample, baseK,
		matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	double max_dist = numeric_limits<double>::min();
	double min_dist = numeric_limits<double>::max();

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < contourDescriptors.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}


	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	std::vector< DMatch > good_matches;

	for (int i = 0; i < contourDescriptors.rows; i++)
	{
		if (matches[i].distance <= max(2 * min_dist, max_dist/4))
		{
			good_matches.push_back(matches[i]);
		}
	}

	//-- Draw only "good" matches
	img_matches = Scalar(0);
	drawMatches(img, contourK, sample, baseK,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);



	/*
	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < contourDescriptors.rows; i++)
	{
	double dist = matches[i].distance;
	if (dist < min_dist) min_dist = dist;
	if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches;

	for (int i = 0; i < contourDescriptors.rows; i++)
	{
	if (matches[i].distance < 3 * min_dist)
	{
	good_matches.push_back(matches[i]);
	}
	}

	Mat img_matches;
	drawMatches(baseImg, baseK, img, contourK,
	matches, img_matches, Scalar::all(-1), Scalar::all(-1),
	vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < matches.size(); i++)
	{
	//-- Get the keypoints from the good matches
	obj.push_back(baseK[matches[i].queryIdx].pt);
	scene.push_back(contourK[matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, CV_RANSAC);

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(baseImg.cols, 0);
	obj_corners[2] = cvPoint(baseImg.cols, baseImg.rows); obj_corners[3] = cvPoint(0, baseImg.rows);
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform(obj_corners, scene_corners, H);

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line(img_matches, scene_corners[0] + Point2f(baseImg.cols, 0), scene_corners[1] + Point2f(baseImg.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + Point2f(baseImg.cols, 0), scene_corners[2] + Point2f(baseImg.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + Point2f(baseImg.cols, 0), scene_corners[3] + Point2f(baseImg.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + Point2f(baseImg.cols, 0), scene_corners[0] + Point2f(baseImg.cols, 0), Scalar(0, 255, 0), 4);

	//-- Show detected matches
	imshow("Good Matches & Object detection", img_matches);

	waitKey(0);
	*/

		

	return 0;
}


// COORELATION FUNCTIONS -------------------------------------------------------------------------------------------------

double Utility::spearmanCorrelation(std::vector<cv::Point>& distribution, std::vector<cv::Point>& base)
{
	int validPoints = 0;

	// FIND MEANS
	Point meanDistr(0, 0), meanBase(0, 0);
	for (int i = 0; i < distribution.size(); i++)
	{
		if (distribution[i] == Point(0, 0) || base[i] == Point(0, 0))
			continue;

		validPoints++;

		meanDistr.x += distribution[i].x;
		meanDistr.y += distribution[i].y;
		meanBase.x += base[i].x;
		meanBase.y += base[i].y;
	}

	meanDistr.x /= validPoints;
	meanDistr.y /= validPoints;

	meanBase.x /= base.size();
	meanBase.y /= base.size();

	Point2d diffDistr, diffBase;

	Point2d product = 0;
	Point2d distr2 = 0;
	Point2d base2 = 0;

	for (int i = 0; i < distribution.size(); i++)
	{
		if (distribution[i] == Point(0, 0) || base[i] == Point(0, 0))
			continue;

		diffDistr.x = distribution[i].x - meanDistr.x;
		diffDistr.y = distribution[i].y - meanDistr.y;

		diffBase.x = base[i].x - meanBase.x;
		diffBase.y = base[i].y - meanBase.y;

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

	return ((correlation.x + correlation.y) / 2) * 100;
}

double Utility::pearsonCorrelation(std::vector<cv::Point>& distribution, std::vector<cv::Point>& base)
{
	Point2d totalProduct(0,0),
		totalBase(0,0),
		totalDistr(0,0),
		totalBase2(0,0),
		totalDistr2(0,0);

	for (int i = 0; i < base.size(); i++)
	{
		totalProduct.x += distribution[i].x * base[i].x;
		totalProduct.y += distribution[i].y * base[i].y;

		totalBase.x += base[i].x;
		totalBase.y += base[i].y;

		totalDistr.x += distribution[i].x;
		totalDistr.y += distribution[i].y;

		totalBase2.x += base[i].x * base[i].x;
		totalBase2.y += base[i].y * base[i].y;

		totalDistr2.x += distribution[i].x * distribution[i].x;
		totalDistr2.y += distribution[i].y * distribution[i].y;
	}

	Point2d correlation;

	correlation.x = (totalProduct.x - (totalBase.x * totalDistr.x) / base.size()) /
		((sqrt(totalBase2.x - (totalBase.x * totalBase.x / base.size()))) * (sqrt(totalDistr2.x - (totalDistr.x * totalDistr.x / base.size()))));

	correlation.y = (totalProduct.y - (totalBase.y * totalDistr.y) / base.size()) /
		((sqrt(totalBase2.y - (totalBase.y * totalBase.y / base.size()))) * (sqrt(totalDistr2.y - (totalDistr.y * totalDistr.y / base.size()))));

	return (correlation.x + correlation.y) * 100 / 2;
}

double Utility::singlePearsonCorrelation(std::vector<double>& distribution, std::vector<double>& base)
{
	int s;
	if (distribution.size() > base.size())
		s = base.size();
	else
		s = distribution.size();

	double totalProduct = 0,
		totalBase = 0,
		totalDistr = 0,
		totalBase2 = 0,
		totalDistr2 = 0;

	for (int i = 0; i < base.size(); i++)
	{
		totalProduct += distribution[i] * base[i];
		
		totalBase += base[i];		

		totalDistr += distribution[i];		

		totalBase2 += base[i] * base[i];	

		totalDistr2 += distribution[i] * distribution[i];
		
	}

	double correlation;

	correlation = (totalProduct - (totalBase * totalDistr) / base.size()) /
		((sqrt(totalBase2 - (totalBase * totalBase / base.size()))) * (sqrt(totalDistr2 - (totalDistr * totalDistr / base.size()))));
	

	return (correlation) * 100;

	/*

	int s;
	if (distribution.size() > base.size())
	s = base.size();
	else
	s = distribution.size();

	double totalProduct = 0,
	totalBase = 0,
	totalDistr = 0,
	totalBase2 = 0,
	totalDistr2 = 0;

	// FIND MEANS
	double meanDistr = 0, meanBase = 0;
	for (int i = 0; i < distribution.size(); i++)
	{
		meanDistr += distribution[i];
		meanBase += base[i];
	}

	meanDistr /= distribution.size();
	meanBase /= base.size();


	double diffDistr, diffBase;

	double product = 0;
	double distr2 = 0;
	double base2 = 0;

	for (int i = 0; i < distribution.size(); i++)
	{
		diffDistr = distribution[i] - meanDistr;

		diffBase = base[i] - meanBase;

		product += diffDistr * diffBase;

		distr2 += diffDistr * diffDistr;

		base2 += diffBase * diffBase;
	}

	double correlation;
	correlation = product / sqrt(distr2 * base2);

	return correlation * 100;
	*/
}

double Utility::singleSpearmanCorrelation(std::vector<double>& distribution, std::vector<double>& base)
{
	
	int s;
	if (distribution.size() > base.size())
	s = base.size();
	else
	s = distribution.size();

	double totalProduct = 0,
	totalBase = 0,
	totalDistr = 0,
	totalBase2 = 0,
	totalDistr2 = 0;	

	// FIND MEANS
	double meanDistr = 0, meanBase = 0;
	for (int i = 0; i < s; i++)
	{
		meanDistr += distribution[i];
		meanBase += base[i];
	}

	meanDistr /= distribution.size();
	meanBase /= base.size();


	double diffDistr, diffBase;

	double product = 0;
	double distr2 = 0;
	double base2 = 0;

	for (int i = 0; i < distribution.size(); i++)
	{
	diffDistr = distribution[i] - meanDistr;

	diffBase = base[i] - meanBase;

	product += diffDistr * diffBase;

	distr2 += diffDistr * diffDistr;

	base2 += diffBase * diffBase;
	}

	double correlation;
	correlation = product / sqrt(distr2 * base2);

	return correlation * 100;	
}

double Utility::calculateHausdorffDistance( std::vector<cv::Point> contour, std::vector<cv::Point> base)
{
	vector<Point> contourN = contour;
	vector<Point> baseN = base;
	
	double maxDistCB = 0;

	for (int i = 0; i < contourN.size(); i++)
	{
		double minB = numeric_limits<double>::max();
		double tempDist;

		for (int j = 0; j < baseN.size(); j++)
		{
			double cat1 = (contourN[i].x - baseN[j].x);
			double cat2 = (contourN[i].y - baseN[j].y);
			tempDist = sqrt(cat1*cat1 + cat2*cat2);

			if (tempDist < minB)
				minB = tempDist;
		}
		maxDistCB += minB;
	}

	Moments m = moments(contourN);
	maxDistCB /= m.m00;

	// ===========================================================

	double maxDistBC = 0;

	for (int i = 0; i < baseN.size(); i++)
	{
		double minC = numeric_limits<double>::max();
		double tempDist;

		for (int j = 0; j < contourN.size(); j++)
		{
			double cat1 = (baseN[i].x - contourN[j].x);
			double cat2 = (baseN[i].y - contourN[j].y);
			tempDist = sqrt(cat1*cat1 + cat2*cat2);

			if (tempDist < minC)
				minC = tempDist;
		}
		maxDistBC += minC;
	}

	Moments m1 = moments(baseN);
	maxDistBC /= m1.m00;

	return max(maxDistBC, maxDistCB);
}

// -----------------------------------------------------------------------------------------------------------------------

std::vector<cv::Point> Utility::normalize(std::vector<cv::Point> source)
{
	vector<Point> norm;

	Rect bound = boundingRect(source);
	int minX = bound.x;
	int minY = bound.y;
	

	for (int i = 0; i < source.size(); i++)
	{
		norm.push_back(source[i]);
		norm[i].x = ((source[i].x - minX) * 4000 / bound.width)+2;
		norm[i].y = ((source[i].y - minY) * 4000 / bound.height)+2;
	}

	return norm;
}

// FEATURE DETECTION FUNCTIONS ---------------------------------------------------------------------------------------------

std::vector<cv::Point> Utility::findCentroidsDistribution(std::vector<cv::Point> contour){

	vector<Point> retCentr;

	/*
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

	Size size(maxX + minX, maxY + minY);
	*/

	

//	vector<Point> normalized = normalize(contour);

	Rect box = boundingRect(contour);
	Size size(box.x + box.width, box.y + box.height);

/*
#ifdef DEBUG_MODE
	Mat tempImg(Size(1000, 1000), CV_8UC1);
	tempImg = cv::Scalar(0);
	vector<vector<Point>> vect;
	vect.push_back(contour);
	drawContours(tempImg, vect, -1, cv::Scalar(255), 1, CV_AA);

	vector<vector<Point>> v;
	v.push_back(normalized);
	drawContours(tempImg, v, -1, cv::Scalar(255), 1, CV_AA);
#endif
*/

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
	box.height += 2;
	box.width += 2;
	size.width += 2;
	size.height += 2;

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
			boxSplit = Rect(x, y, box.width / 2, box.height / 2);
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
	}

	Moments m = moments(contour, true);
	int cx = int(m.m10 / m.m00);
	int cy = int(m.m01 / m.m00);

	retCentr.push_back(Point(cx, cy));

	img = Scalar(0);
	drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);

	//retCentr = normalize(retCentr);
	
	for (int i = 0; i < retCentr.size(); i++)
		circle(img, retCentr[i], 5, Scalar(255), -1, 8, 0);

	return retCentr;

}

std::vector<cv::Point> Utility::findLargeCentroidsDistribution(std::vector<cv::Point> contour)
{
	vector<Point> retCentr;
	Rect box = boundingRect(contour);
	
	if (box.width % 4 != 0)
	{
		box.width += (4 - box.width % 4);
		//size.width += (4 - size.width % 4);
	}


	if (box.height % 4 != 0)
	{
		box.height += (4 - box.height % 4);
		//size.height += (4 - size.height % 4);
	}
	box.height += 4;
	box.width += 4;


	Size size(box.x + box.width, box.y + box.height);
	Mat img(size, CV_8UC1);
	img = Scalar(0);	
	vector<vector<Point>> tempVector;
	tempVector.push_back(contour);

	int x, y;


	Moments m = moments(contour, true);
	int cx = int(m.m10 / m.m00);
	int cy = int(m.m01 / m.m00);
	Point center(cx, cy);

	// Vertical Split |
	for (x = box.x; x < (box.x + box.width); x += (box.width / 4))
	{
		y = box.y;

		Rect boxSplit(x, box.y, box.width / 4, box.height);
		img = Scalar(0);
		drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);
		Mat sub = img(boxSplit);

		vector<vector<Point>> shape;
		findContours(sub, shape, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

		if (shape.size() == 0)
		{
			//retCentr.push_back(center);
			retCentr.push_back(Point(0, 0));
		}
		else if (shape.size() > 1)
		{
			/*
			
			*/

			Moments m = moments(sub, true);
			int cx = int(m.m10 / m.m00) + x;
			int cy = int(m.m01 / m.m00) + y;
			retCentr.push_back(Point(cx, cy));
			circle(img, Point(cx, cy), 5, Scalar(255), -1, 8, 0);
		}
		else
		{
			Moments m = moments(shape[0], true);
			int cx = int(m.m10 / m.m00) + x;
			int cy = int(m.m01 / m.m00) + y;

			retCentr.push_back(Point(cx, cy));
			circle(img, Point(cx, cy), 5, Scalar(255), -1, 8, 0);
		}
		

		// Sub Horizontal Split --
		for (y = box.y; y < (box.y + box.height); y += (box.height / 4))
		{
			boxSplit = Rect(x, y, box.width / 4, box.height / 4);
			img = Scalar(0);
			drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);
			sub = img(boxSplit);

			vector<vector<Point>> shape;
			findContours(sub, shape, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

			if (shape.size() == 0)
			{
				//retCentr.push_back(center);
				retCentr.push_back(Point(0, 0));
			}
			else if (shape.size() > 1)
			{
				/*
				for (int i = 0; i < shape.size(); i++)
				{
					if (shape[i].size() > shape[0].size())
						shape[0] = shape[i];
				}
				*/
				Moments m = moments(sub, true);
				int cx = int(m.m10 / m.m00) + x;
				int cy = int(m.m01 / m.m00) + y;
				retCentr.push_back(Point(cx, cy));
				circle(img, Point(cx, cy), 5, Scalar(255), -1, 8, 0);
				
			}
			else
			{
				m = moments(shape[0], true);
				int cx = int(m.m10 / m.m00) + x;
				int cy = int(m.m01 / m.m00) + y;

				retCentr.push_back(Point(cx, cy));
				circle(img, Point(cx, cy), 5, Scalar(255), -1, 8, 0);
			}		
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

		if (shape.size() == 0)
		{
			//retCentr.push_back(center);	
			retCentr.push_back(Point(0, 0));
		}
		else if (shape.size() > 1)
		{
			/*
			for (int i = 0; i < shape.size(); i++)
			{
				if (shape[i].size() > shape[0].size())
					shape[0] = shape[i];
			}
			*/

			Moments m = moments(sub, true);
			int cx = int(m.m10 / m.m00) + x;
			int cy = int(m.m01 / m.m00) + y;
			retCentr.push_back(Point(cx, cy));
			circle(img, Point(cx, cy), 5, Scalar(255), -1, 8, 0);			
		}
		else
		{
			Moments m = moments(shape[0], true);
			int cx = int(m.m10 / m.m00) + x;
			int cy = int(m.m01 / m.m00) + y;

			retCentr.push_back(Point(cx, cy));
			circle(img, Point(cx, cy), 5, Scalar(255), -1, 8, 0);
		}		
	}

	//retCentr.push_back(center);

	img = Scalar(0);
	drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);

	//retCentr = normalize(retCentr);

	for (int i = 0; i < retCentr.size(); i++)
		circle(img, retCentr[i], 5, Scalar(255), -1, 8, 0);

	return retCentr;
}

void Utility::findCentroidsKeypoints(std::vector<cv::Point> contour,
													std::vector<cv::KeyPoint>& centroids,													
													CentroidDetectionMode mode)
{
	Rect box = boundingRect(contour);

	box.height += 4;
	box.width += 4;
	
	if (box.x == 1)
		box.x -= 1;
	if (box.y == 1)
		box.y -= 1;	

	
	// center of the contour
	Moments m = moments(contour, true);
	int cx = int(m.m10 / m.m00);
	int cy = int(m.m01 / m.m00);
	Point2f center(cx, cy);

	
	Size size(box.x + box.width + cx, box.y + box.height + cy);
	Mat img(size, CV_8UC1);
	img = Scalar(0);
	vector<vector<Point>> tempVector;
	tempVector.push_back(contour);
	drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);


	circle(img, center, 3, Scalar(255), -1, 8, 0);

	centroids.push_back(KeyPoint(center, 1));

	
	// SPLIT ======================= = LEVEL 1 = ===============================

	int x = box.x;
	int y = box.y;
	int h = box.height;
	int w = box.width;

	Rect leftSplit(x, y, cx - x, h);
	Rect rigthSplit(cx, y, w, h);
	Rect topSplit(x, y, w, cy - y);
	Rect bottomSplit(x, cy, w, h);

	vector<Rect> splitRect;
	splitRect.push_back(leftSplit);
	splitRect.push_back(rigthSplit);
	splitRect.push_back(topSplit);
	splitRect.push_back(bottomSplit);

	for (int i = 0; i < splitRect.size(); i++)
	{
		img = Scalar(0);
		drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);
		Mat sub = img(splitRect[i]);

		int dilation_size = 1;
		Mat element = getStructuringElement(0, Size(2 * dilation_size, 2 * dilation_size), Point(dilation_size, dilation_size));
		dilate(sub, sub, element);

		vector<vector<Point>> shape;
		findContours(sub, shape, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

		if (shape.size() != 0) // FIRST LEVEL LOOP
		{
			int cx_1, cy_1;

			if (shape.size() == 1)
			{
				// center of the contour
				Moments m = moments(shape[0], true);
				cx_1 = int(m.m10 / m.m00) + splitRect[i].x;
				cy_1 = int(m.m01 / m.m00) + splitRect[i].y;
			}
			else
			{
				// center of the sub image
				Moments m = moments(sub, true);
				cx_1 = int(m.m10 / m.m00) + splitRect[i].x;
				cy_1 = int(m.m01 / m.m00) + splitRect[i].y;
			}

			Point2f center(cx_1, cy_1);

			//Size size(box.x + box.width + cx, box.y + box.height + cy);
			//Mat img(size, CV_8UC1);
			//img = Scalar(0);
			//drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);


			circle(img, center, 6, Scalar(255), -1, 8, 0);
			centroids.push_back(KeyPoint(center, 2));

			if (mode != CentroidDetectionMode::ONE_LOOP)
			{
				// SPLIT ========================== = LEVEL 2 = ===============================

				int x = splitRect[i].x;
				int y = splitRect[i].y;
				int h = splitRect[i].height;
				int w = splitRect[i].width;
				

				Rect leftSplit_1(x, y, cx_1 - x, h);
				
				Rect rigthSplit_1(cx_1, y, (w + x) - cx_1, h);

				Rect topSplit_1(x, y, w, cy_1 - y);
		
				Rect bottomSplit_1(x, cy_1, w, (h + y) - cy_1);	

				vector<Rect> splitRect_1;
				splitRect_1.push_back(leftSplit_1);
				splitRect_1.push_back(rigthSplit_1);
				splitRect_1.push_back(topSplit_1);
				splitRect_1.push_back(bottomSplit_1);

				for (int j = 0; j < splitRect_1.size(); j++)
				{
					img = Scalar(0);
					drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);
					Mat sub = img(splitRect_1[j]);

					int dilation_size = 1;
					Mat element = getStructuringElement(0, Size(2 * dilation_size, 2 * dilation_size), Point(dilation_size, dilation_size));
					dilate(sub, sub, element);

					vector<vector<Point>> shape_1;
					findContours(sub, shape_1, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

					if (shape_1.size() != 0) // SECOND LEVEL LOOP
					{
						int cx_2, cy_2;

						if (shape_1.size() == 1)
						{
							// center of the contour
							Moments m = moments(shape_1[0], true);
							cx_2 = int(m.m10 / m.m00) + splitRect_1[j].x;
							cy_2 = int(m.m01 / m.m00) + splitRect_1[j].y;
						}
						else
						{
							// center of the sub image
							Moments m = moments(sub, true);
							cx_2 = int(m.m10 / m.m00) + splitRect_1[j].x;
							cy_2 = int(m.m01 / m.m00) + splitRect_1[j].y;

						}
						

						Point2f center(cx_2, cy_2);


						//Size size(box.x + box.width + cx, box.y + box.height + cy);
						//Mat img(size, CV_8UC1);
						//img = Scalar(0);
						//drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);


						circle(img, center, 9, Scalar(255), -1, 8, 0);
						centroids.push_back(KeyPoint(center, 3));

						/*
						if (mode == CentroidDetectionMode::THREE_LOOP) // SPLIT ========================== = LEVEL 3 = ===============================
						{
							int x = splitRect_1[j].x;
							int y = splitRect_1[j].y;
							int h = splitRect_1[j].height;
							int w = splitRect_1[j].width;


							Rect leftSplit_2(x, y, cx - x, h);
							Rect rigthSplit_2(cx, y, w, h);
							Rect topSplit_2(x, y, w, cy - y);
							Rect bottomSplit_2(x, cy, w, h);

							vector<Rect> splitRect_2;
							splitRect_2.push_back(leftSplit_2);
							splitRect_2.push_back(rigthSplit_2);
							splitRect_2.push_back(topSplit_2);
							splitRect_2.push_back(bottomSplit_2);

							for (int k = 0; k < splitRect_2.size(); k++)
							{
								img = Scalar(0);
								drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);
								Mat sub = img(splitRect_2[k]);

								int dilation_size = 1;
								Mat element = getStructuringElement(0, Size(2 * dilation_size, 2 * dilation_size), Point(dilation_size, dilation_size));
								dilate(sub, sub, element);

								vector<vector<Point>> shape_2;
								findContours(sub, shape_2, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

								if (shape_2.size() == 1) // THIRD LEVEL LOOP
								{
									// center of the contour
									Moments m = moments(shape_2[0], true);
									int cx = int(m.m10 / m.m00) + splitRect_2[j].x;
									int cy = int(m.m01 / m.m00) + splitRect_2[j].y;
									Point2f center(cx, cy);


									//Size size(box.x + box.width + cx, box.y + box.height + cy);
									//Mat img(size, CV_8UC1);
									//img = Scalar(0);
									//drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);


									circle(img, center, 9, Scalar(255), -1, 8, 0);
									centroids.push_back(KeyPoint(center, 3));

								}

							}

						}//  ========================== = LEVEL 3 = ===============================					
						*/
					}

				}

			}// ========================== = LEVEL 2 = ===============================			

		}

	}
}



std::vector<double> Utility::findDistancesFromCenter(std::vector<cv::Point> distribution)
{
	Moments m = moments(distribution);
	Point center(m.m10 / m.m00, m.m01 / m.m00);

	vector<double> distances;
	
	for (int i = 0; i < distribution.size(); i++)
	{
		if (distribution[i] == Point(0, 0))
			continue;

		double cat1 = (distribution[i].x - center.x);
		double cat2 = (distribution[i].y - center.y);
		double dist = sqrt(cat1*cat1 + cat2*cat2);
		distances.push_back(dist);
	}

	return distances;
}

std::vector<double> Utility::findAnglesRespectCenter(std::vector<cv::Point> distribution)
{
	Moments m = moments(distribution);
	Point center(m.m10 / m.m00, m.m01 / m.m00);

	vector<double> angles;

	for (int i = 0; i < distribution.size(); i++)
	{
		if (distribution[i] == Point(0, 0))
			continue;

		double cat1 = (distribution[i].x - center.x);
		double cat2 = (distribution[i].y - center.y);
		double ip = sqrt(cat1*cat1 + cat2*cat2);
		double ang = asin(cat1 / ip);

		angles.push_back(ang);
	}

	return angles;
}

// --------------------------------------------------------------------------------------------------------------------------

bool Utility::isLessThan(Point a, Point b, Point center)
{
	if (a.x - center.x >= 0 && b.x - center.x < 0)
		return true;
	if (a.x - center.x < 0 && b.x - center.x >= 0)
		return false;
	if (a.x - center.x == 0 && b.x - center.x == 0) {
		if (a.y - center.y >= 0 || b.y - center.y >= 0)
			return a.y > b.y;
		return b.y > a.y;
	}

	// compute the cross product of vectors (center -> a) x (center -> b)
	int det = (a.x - center.x) * (b.y - center.y) - (b.x - center.x) * (a.y - center.y);
	if (det < 0)
		return true;
	if (det > 0)
		return false;

	// points a and b are on the same line from the center
	// check which point is closer to the center
	int d1 = (a.x - center.x) * (a.x - center.x) + (a.y - center.y) * (a.y - center.y);
	int d2 = (b.x - center.x) * (b.x - center.x) + (b.y - center.y) * (b.y - center.y);
	return d1 > d2;
}

vector<Point> Utility::sortPoints(vector<Point> points)
{
	vector<Point> ret;
	int sortingPositions[18] = { 0, 2, 1, 6, 5, 20, 10, 11, 16, 17, 15, 18, 19, 14, 21, 9, 4, 3 };
	
	for (int i = 0; i < 18; i++)
	{
		if (points[sortingPositions[i]] == Point(0, 0))
			continue;
		ret.push_back(points[sortingPositions[i]]);
	}

#ifdef DEBUG_MODE
	Mat tempImg(Size(4000, 4000), CV_8UC1);
	tempImg = cv::Scalar(0);
	vector<vector<Point>> vect;
	vect.push_back(ret);
	drawContours(tempImg, vect, -1, cv::Scalar(255), 1, CV_AA);
#endif

	return ret;
}