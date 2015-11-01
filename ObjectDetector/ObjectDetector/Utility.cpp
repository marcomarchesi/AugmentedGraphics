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

	vector<Point> distribution = Utility::findLargeCentroidsDistribution(contour);

	//if (distribution.size() != 23) return 0;

	vector<Point> base = Utility::findLargeCentroidsDistribution(baseContour);

	/*
	if (base.size() != 23){
		cout << "FATAL ERROR, number of centroids of the base shape is wrong," << endl << "control the base image or aspectedContourPoint!!!!" << endl;
		return 0;
	}
	*/
	
	if (distribution.size() != base.size())
		return 0;

	/*
	vector<Point> normC = normalize(contour);
	vector<Point> normB = normalize(baseContour);

#ifdef DEBUG_MODE
	Mat tempImg(Size(4000, 4000), CV_8UC1);
	tempImg = cv::Scalar(0);

	vector<vector<Point>> vect;
	vect.push_back(normC);
	drawContours(tempImg, vect, -1, cv::Scalar(255), 1, CV_AA);

	vect.push_back(normB);
	drawContours(tempImg, vect, -1, cv::Scalar(255), 1, CV_AA);

#endif


	Moments mDistr, mBase;
	mDistr = moments(normC);
	mBase = moments(normB);

	double eccDistr, eccBase, distrMajor, distrMinor, baseMajor, baseMinor;
	
	double deltaDistr = sqrt(4 * pow(mDistr.m11, 2) - pow((mDistr.m20 - mDistr.m02), 2));
	double deltaBase = sqrt(4 * pow(mBase.m11, 2) - pow((mBase.m20 - mBase.m02), 2));

	distrMajor = sqrt(abs(((mDistr.m20 + mDistr.m02) / 2) + deltaDistr));
	distrMinor = sqrt(abs(((mDistr.m20 + mDistr.m02) / 2) - deltaDistr));	
	baseMajor = sqrt(abs(((mBase.m20 + mBase.m02) / 2) + deltaBase));
	baseMinor = sqrt(abs(((mBase.m20 + mBase.m02) / 2) - deltaBase));

	double distrAdiff = distrMajor - distrMinor;
	double baseAdiff = baseMajor - baseMinor;
	double axesDifference = abs(distrAdiff - baseAdiff);
	*/
	
	double centroidsCorrelation = spearmanCorrelation(distribution, base);	
	double distancesCorrelation = singleSpearmanCorrelation(findDistancesFromCenter(distribution), findDistancesFromCenter(base));
	double anglesCorrelation = singleSpearmanCorrelation(findAnglesRespectCenter(distribution), findAnglesRespectCenter(base));
	double centroidHamming = calculateContourPercentageCompatibility(sortPoints(distribution), sortPoints(base));

	double correlation = (centroidsCorrelation + distancesCorrelation + anglesCorrelation + centroidHamming) / 4;

	

	return correlation;

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