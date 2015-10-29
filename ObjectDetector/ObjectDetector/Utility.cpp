#include "Utility.h"
#include "commonInclude.h"


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

	// ---- SHAPE'S POINT FILTER ----
	/*
	if (contour.size() != 8)
	return 0.0;
	*/

	double hamming = matchShapes(contour, base, CV_CONTOURS_MATCH_I1, 0.0);

	return ((1 - hamming) * 100);
}

double Utility::correlationWithBase(std::vector<cv::Point> contour, std::vector<cv::Point> baseContour){

	vector<Point> distribution = Utility::findCentroidsDistribution(contour);

	if (distribution.size() != 9) return 0;

	vector<Point> base = Utility::findCentroidsDistribution(baseContour);

	if (base.size() != 9){
		cout << "FATAL ERROR, number of centroids of the base shape is wrong," << endl << "control the base image or aspectedContourPoint!!!!" << endl;
		return 0;
	}

	return pearsonCorrelation(distribution, base);

	//return pearsonCorrelation(distribution, base);

}

double Utility::spearmanCorrelation(std::vector<cv::Point>& distribution, std::vector<cv::Point>& base)
{
	// FIND MEANS
	Point meanDistr(0, 0), meanBase(0, 0);
	for (int i = 0; i < distribution.size(); i++)
	{
		meanDistr.x += distribution[i].x;
		meanDistr.y += distribution[i].y;

		meanBase.x += base[i].x;
		meanBase.y += base[i].y;
	}

	meanDistr.x /= distribution.size();
	meanDistr.y /= distribution.size();

	meanBase.x /= base.size();
	meanBase.y /= base.size();

	Point2d diffDistr, diffBase;

	Point2d product = 0;
	Point2d distr2 = 0;
	Point2d base2 = 0;

	for (int i = 0; i < distribution.size(); i++)
	{
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

std::vector<cv::Point> Utility::findCentroidsDistribution(std::vector<cv::Point> contour){

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

	Size size(maxX + minX, maxY + minY);

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

	for (int i = 0; i < retCentr.size(); i++)
		circle(img, retCentr[i], 5, Scalar(255), -1, 8, 0);

	/*
#ifdef DEBUG_MODE
	imshow("Centroidi", img);
#endif
	*/

	return retCentr;

}