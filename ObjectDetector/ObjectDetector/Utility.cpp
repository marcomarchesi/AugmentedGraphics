#include "Utility.h"
#include "commonInclude.h"
#include <math.h>

using namespace cv;
using namespace std;
using namespace od;


std::vector<cv::Rect> Utility::splitRect(Rect box, Point centroid, int level, SplitMode mode)
{
	vector<Rect> splitRect;

	int x = box.x;
	int y = box.y;
	int h = box.height;
	int w = box.width;

	int cx = centroid.x;
	int cy = centroid.y;

	if (mode == SplitMode::MODE_1)
	{
		if (level == 1)
		{
			Rect leftSplit(x, y, cx - x, h);
			Rect rigthSplit(cx, y, w, h);
			Rect topSplit(x, y, w, cy - y);
			Rect bottomSplit(x, cy, w, h);

			splitRect.push_back(leftSplit);
			splitRect.push_back(rigthSplit);
			splitRect.push_back(topSplit);
			splitRect.push_back(bottomSplit);
		}
		else if (level == 2)
		{
			Rect leftSplit_1(x, y, cx - x, h);
			Rect rigthSplit_1(cx, y, (w + x) - cx, h);
			Rect topSplit_1(x, y, w, cy - y);
			Rect bottomSplit_1(x, cy, w, (h + y) - cy);

			splitRect.push_back(leftSplit_1);
			splitRect.push_back(rigthSplit_1);
			splitRect.push_back(topSplit_1);
			splitRect.push_back(bottomSplit_1);
		}
		else if (level == 3)
		{
			Rect leftSplit_1(x, y, cx - x, h);
			Rect rigthSplit_1(cx, y, (w + x) - cx, h);
			Rect topSplit_1(x, y, w, cy - y);
			Rect bottomSplit_1(x, cy, w, (h + y) - cy);

			splitRect.push_back(leftSplit_1);
			splitRect.push_back(rigthSplit_1);
			splitRect.push_back(topSplit_1);
			splitRect.push_back(bottomSplit_1);
		}
	}
	else
	{
		if (level == 1)
		{
			Rect TLSplit(x, y, cx - x, cy - y);
			Rect TRSplit(cx, y, w, cy - y);
			Rect BLSplit(x, cy, cx - x, h);
			Rect BRSplit(cx, cy, w, h);

			splitRect.push_back(TLSplit);
			splitRect.push_back(TRSplit);
			splitRect.push_back(BLSplit);
			splitRect.push_back(BRSplit);
		}
		else if (level > 1)
		{
			Rect TLSplit(x, y, cx - x, cy - y);
			Rect TRSplit(cx, y, (w + x) - cx, cy - y);
			Rect BLSplit(x, cy, cx - x, (h + y) - cy);
			Rect BRSplit(cx, cy, (w + x) - cx, (h + y) - cy);

			splitRect.push_back(TLSplit);
			splitRect.push_back(TRSplit);
			splitRect.push_back(BLSplit);
			splitRect.push_back(BRSplit);
		}
		

	}
	

	return splitRect;
}

double Utility::calculateContourPercentageCompatibility(std::vector<cv::Point> contour, std::vector<cv::Point> base, HammingMode mode)
{
	vector<vector<vector<Point>>> contourSplitContour, baseSplitContour;

	
	Rect boxC = boundingRect(contour);
	Rect boxB = boundingRect(base);


	Moments mB = moments(base, true);
	int bx = int(mB.m10 / mB.m00);
	int by = int(mB.m01 / mB.m00);
	Point2f centerB(bx, by);

	// center of the contour
	Moments mC = moments(contour, true);
	int cx = int(mC.m10 / mC.m00);
	int cy = int(mC.m01 / mC.m00);
	Point2f centerC(cx, cy);
	
	Size sizeC(boxC.x + boxC.width + cx, boxC.y + boxC.height + cy);
	Mat imgC(sizeC, CV_8UC1);
	imgC = Scalar(0);
	vector<vector<Point>> tempVectorC;
	tempVectorC.push_back(contour);
	drawContours(imgC, tempVectorC, -1, cv::Scalar(255), 1, CV_AA);

	Size sizeB(boxB.x + boxB.width + bx, boxB.y + boxB.height + by);
	Mat imgB(sizeB, CV_8UC1);
	imgB = Scalar(0);
	vector<vector<Point>> tempVectorB;
	tempVectorB.push_back(base);
	drawContours(imgB, tempVectorB, -1, cv::Scalar(255), 1, CV_AA);




	// ========================== DETECT SPLIT CONTOURS ==================================================

	// SPLIT ======================= = LEVEL 1 = ===============================

	
	vector<Rect> splitC = splitRect(boxC, centerC, 1, SplitMode::MODE_1);
	vector<Rect> splitB = splitRect(boxB, centerB, 1, SplitMode::MODE_1);
	

	for (int i = 0; i < splitC.size(); i++)
	{
		imgC = Scalar(0);
		imgB = Scalar(0);
		drawContours(imgC, tempVectorC, -1, cv::Scalar(255), 1, CV_AA);
		drawContours(imgB, tempVectorB, -1, cv::Scalar(255), 1, CV_AA);

		
		Mat subC = imgC(splitC[i]);
		Mat subB = imgB(splitB[i]);

		int dilation_size = 1;
		Mat element = getStructuringElement(1, Size(2 * dilation_size, 2 * dilation_size), Point(dilation_size, dilation_size));
		dilate(subC, subC, element);
		dilate(subB, subB, element);
		

		vector<vector<Point>> shapeC;
		vector<vector<Point>> shapeB;
		findContours(subC, shapeC, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
		findContours(subB, shapeB, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

		contourSplitContour.push_back(shapeC);
		baseSplitContour.push_back(shapeB);

		int cx_1, cy_1;
		int bx_1, by_1;

		
		if (!(shapeC.size() == 0 || shapeB.size() == 0)) // FIRST LEVEL LOOP
		{
			Moments mC = moments(shapeC[0], true);
			cx_1 = int(mC.m10 / mC.m00) + splitC[i].x;
			cy_1 = int(mC.m01 / mC.m00) + splitC[i].y;

			Moments mB = moments(shapeB[0], true);
			bx_1 = int(mB.m10 / mB.m00) + splitB[i].x;
			by_1 = int(mB.m01 / mB.m00) + splitB[i].y;
		}
		
		
		Point2f centerC(cx_1, cy_1);
		Point2f centerB(bx_1, by_1);

				
		if (mode != CentroidDetectionMode::ONE_LOOP)
		{
			// SPLIT ========================== = LEVEL 2 = ===============================

			vector<Rect> splitC_1 = splitRect(splitC[i], centerC, 2, SplitMode::MODE_1);
			vector<Rect> splitB_1 = splitRect(splitB[i], centerB, 2, SplitMode::MODE_1);


			for (int j = 0; j < splitC_1.size(); j++)
			{
				imgC = Scalar(0);
				imgB = Scalar(0);
				drawContours(imgC, tempVectorC, -1, cv::Scalar(255), 1, CV_AA);
				drawContours(imgB, tempVectorB, -1, cv::Scalar(255), 1, CV_AA);

				Mat subC = imgC(splitC_1[j]);
				Mat subB = imgB(splitB_1[j]);

				int dilation_size = 1;
				Mat element = getStructuringElement(0, Size(2 * dilation_size, 2 * dilation_size), Point(dilation_size, dilation_size));
				dilate(subC, subC, element);
				dilate(subB, subB, element);
				
				vector<vector<Point>> shapeC_1;
				vector<vector<Point>> shapeB_1;
				findContours(subC, shapeC_1, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
				findContours(subB, shapeB_1, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

				contourSplitContour.push_back(shapeC_1);
				baseSplitContour.push_back(shapeB_1);

				int cx_2, cy_2;
				int bx_2, by_2;

				if (!(shapeC_1.size() == 0 || shapeB_1.size() == 0)) // FIRST LEVEL LOOP
				{
					// center of the contour
					Moments mC = moments(shapeC_1[0], true);
					cx_2 = int(mC.m10 / mC.m00) + splitC[i].x;
					cy_2 = int(mC.m01 / mC.m00) + splitC[i].y;

					Moments mB = moments(shapeB_1[0], true);
					bx_2 = int(mB.m10 / mB.m00) + splitB[i].x;
					by_2 = int(mB.m01 / mB.m00) + splitB[i].y;

				}
				

				Point2f centerC(cx_2, cy_2);
				Point2f centerB(bx_2, by_2);	
			}

		}// ========================== = LEVEL 2 = ===============================		

	}


	// ========================== FIND BEST MATCH ==================================================

	

	vector<DMatch> goodMatch;
	map<int, double> matched;

	Mat cc(Size(1000, 1000), CV_8UC1),
		cb(Size(1000, 1000) , CV_8UC1);

	vector<vector<Point>> tempC, tempB;

	tempC.push_back(contour);
	tempB.push_back(base);
	contourSplitContour.push_back(tempC);
	baseSplitContour.push_back(tempB);

	/*
	for (int i = 0; i < contourSplitContour.size(); i++)
	{
		if (contourSplitContour[i].size() != 0)
		{

			double maxHamming = numeric_limits<double>::lowest();
			int maxID = -1;

			if (i == 18)
				cout << "";

			for (int j = 0; j < baseSplitContour.size(); j++)
			{
				double hamming = 0;

				/*
				if (contourSplitContour[i].size() == 1 && baseSplitContour[j].size() == 1)
				{
					hamming = matchShapes(contourSplitContour[i], baseSplitContour[j], mode, 0.0);					
				}
				else
				{
					/*
					for (int k = 0; k < contourSplitContour[i].size(); k++)
					{
						vector<Point> hullC, hullB;
						//convexHull(contourSplitContour[i][k], hullC);

						//double epsilon = contourSplitContour[i][k].size() * 0.003;
						//approxPolyDP(contourSplitContour[i][k], hullC, epsilon, true);

						hullC = contourSplitContour[i][k];

						if (contourSplitContour[i].size() < baseSplitContour[j].size())
						{
							//convexHull(baseSplitContour[j][k + 1], hullB);
							//double epsilon = baseSplitContour[j][k + 1].size() * 0.003;
							//approxPolyDP(baseSplitContour[j][k + 1], hullB, epsilon, true);

							hullB = baseSplitContour[j][k+1];
						}
						else if (contourSplitContour[i].size() == baseSplitContour[j].size())
						{
							//convexHull(baseSplitContour[j][k], hullB);
							//double epsilon = baseSplitContour[j][k].size() * 0.003;
							//approxPolyDP(baseSplitContour[j][k], hullB, epsilon, true);

							hullB = baseSplitContour[j][k];
						}
						else
						{
							if (k < baseSplitContour[j].size())
							{
								//convexHull(baseSplitContour[j][k], hullB);
								//double epsilon = baseSplitContour[j][k].size() * 0.003;
								//approxPolyDP(baseSplitContour[j][k], hullB, epsilon, true);

								hullB = baseSplitContour[j][k];

							}
						}

						tempC.clear();
						tempB.clear();
						tempC.push_back(hullC);
						tempB.push_back(hullB);

						cc = Scalar(0);
						cb = Scalar(0);

						drawContours(cc, tempC, -1, cv::Scalar(255), 1, CV_AA);
						drawContours(cb, tempB, -1, cv::Scalar(255), 1, CV_AA);

						hamming += matchShapes(hullC, hullB, mode, 0.0);
					}
					hamming /= contourSplitContour.size();
				}
				

				cc = Scalar(0);
				cb = Scalar(0);

				drawContours(cc, contourSplitContour[i], -1, cv::Scalar(255), -1, CV_AA);
				drawContours(cb, baseSplitContour[j], -1, cv::Scalar(255), -1, CV_AA);

				hamming = matchShapes(cc, cb, mode, 0.0);

				hamming = (1 - hamming) * 100;
				if (hamming > maxHamming && matched.count(j) == 0)
				{
					maxHamming = hamming;
					maxID = j;					
				}
			}

			goodMatch.push_back(DMatch(i, maxID, maxHamming));
			matched.insert(pair<int, double>(maxID, maxHamming));
		}
		else
		{
			goodMatch.push_back(DMatch(i, -1, 0));
		}
	}
	*/
	

	double hamming = 0, H = 0;
	for (int i = 0; i < contourSplitContour.size(); i++)
	{
		cc = Scalar(0);
		cb = Scalar(0);

		drawContours(cc, contourSplitContour[i], -1, cv::Scalar(255), -1, CV_AA);
		drawContours(cb, baseSplitContour[i], -1, cv::Scalar(255), -1, CV_AA);

		hamming = matchShapes(cc, cb, mode, 0.0);
		hamming = (1 - hamming) * 100;

		H += hamming;
	}
	
	/*
	double H = 0;

	for (int i = 0; i < goodMatch.size(); i++)
	{
		//if (goodMatch[i].distance > 0)
			H += goodMatch[i].distance;
	}
	*/
	H /= (contourSplitContour.size() + 1);
	

	return H;
	
}


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

	vector<Point> contourK, baseK;
	findCentroidsKeypoints(contour, contourK, CentroidDetectionMode::THREE_LOOP);
	findCentroidsKeypoints(baseContour, baseK, CentroidDetectionMode::THREE_LOOP);

	
#ifdef DEBUG_MODE
	Mat tempImg(Size(1920, 1080), CV_8UC3);
	tempImg = Scalar(0);

	for (int i = 0; i < baseK.size(); i++)
		circle(tempImg, baseK[i], 6, Scalar(0, 0, 255), -1, 8, 0);

	for (int i = 0; i < contourK.size(); i++)
		circle(tempImg, contourK[i], 3, Scalar(0, 255, 0), -1, 8, 0);
#endif

	//checkKeyPointsMatch(contourK, baseK);

	
	double correlation = 0;
			
	double centroidsCorrelation = spearmanCorrelation(contourK, baseK);	
	//double distancesCorrelation = singleSpearmanCorrelation(findDistancesFromCenter(c), findDistancesFromCenter(b));
	//double anglesCorrelation = singleSpearmanCorrelation(findAnglesRespectCenter(c), findAnglesRespectCenter(b));
	
	
	//correlation += (centroidsCorrelation + distancesCorrelation + anglesCorrelation) / 3;


	return centroidsCorrelation;
	
}
/*
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



	

		

	return 0;
}
*/

// COORELATION FUNCTIONS -------------------------------------------------------------------------------------------------

double Utility::spearmanCorrelation(std::vector<cv::Point>& distribution, std::vector<cv::Point>& base)
{
	int validPoints = 0;

	int size = min(distribution.size(), base.size());
	
	
	// FIND MEANS
	Point meanDistr(0, 0), meanBase(0, 0);
	for (int i = 0; i < size; i++)
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

	for (int i = 0; i < size; i++)
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

	int validPoints = 0;

	for (int i = 0; i < base.size(); i++)
	{
		if (distribution[i] == Point(0, 0) || base[i] == Point(0, 0))
			continue;
		validPoints++;

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
	double totalProduct = 0,
		totalBase = 0,
		totalDistr = 0,
		totalBase2 = 0,
		totalDistr2 = 0;

	for (int i = 0; i < base.size(); i++)
	{
		if (distribution[i] == 0 || base[i] == 0)
			continue;

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
	
	double totalProduct = 0,
	totalBase = 0,
	totalDistr = 0,
	totalBase2 = 0,
	totalDistr2 = 0;	

	int size = min(distribution.size(), base.size());
	int validPoints = 0;

	// FIND MEANS
	double meanDistr = 0, meanBase = 0;
	for (int i = 0; i <size; i++)
	{
		if (distribution[i] == 0 || base[i] == 0 || distribution[i] == numeric_limits<double>::max() || base[i] == numeric_limits<double>::max())
			continue;

		validPoints++;

		meanDistr += distribution[i];
		meanBase += base[i];
	}

	meanDistr /= validPoints;
	meanBase /= validPoints;


	double diffDistr, diffBase;

	double product = 0;
	double distr2 = 0;
	double base2 = 0;

	for (int i = 0; i < size; i++)
	{

		if (distribution[i] == 0 || base[i] == 0 || distribution[i] == numeric_limits<double>::max() || base[i] == numeric_limits<double>::max())
			continue;
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



// FEATURE DETECTION FUNCTIONS ---------------------------------------------------------------------------------------------

std::vector<cv::Point> Utility::findCentroidsDistribution(std::vector<cv::Point> contour){

	vector<Point> retCentr;

	

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
													std::vector<cv::Point>& centroids,													
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

	centroids.push_back(center);

	
	// SPLIT ======================= = LEVEL 1 = ===============================

	vector<Rect> split = splitRect(box, center, 1, SplitMode::MODE_2);

	for (int i = 0; i < split.size(); i++)
	{
		img = Scalar(0);
		drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);
		Mat sub = img(split[i]);

		int dilation_size = 1;
		Mat element = getStructuringElement(0, Size(2 * dilation_size, 2 * dilation_size), Point(dilation_size, dilation_size));
		dilate(sub, sub, element);

		vector<vector<Point>> shape;
		findContours(sub, shape, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
		
		int cx_1, cy_1;
		
		if (shape.size() == 1)
		{
			// center of the contour
			Moments m = moments(shape[0], true);
			cx_1 = int(m.m10 / m.m00) + split[i].x;
			cy_1 = int(m.m01 / m.m00) + split[i].y;
		}
		else
		{
			// center of the sub image
			Moments m = moments(sub, true);
			cx_1 = int(m.m10 / m.m00) + split[i].x;
			cy_1 = int(m.m01 / m.m00) + split[i].y;
		}

		Point2f center(cx_1, cy_1);

		//Size size(box.x + box.width + cx, box.y + box.height + cy);
		//Mat img(size, CV_8UC1);
		//img = Scalar(0);
		//drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);


		circle(img, center, 6, Scalar(255), -1, 8, 0);
		
		if(shape.size() != 0 && cx_1 > 0 && cy_1 > 0)
			centroids.push_back(center);
		else
			centroids.push_back(Point2f(0,0));


		if (cx_1 > 0 && cy_1 > 0)
		{
			if (mode != CentroidDetectionMode::ONE_LOOP)
			{
				// SPLIT ========================== = LEVEL 2 = ===============================

				vector<Rect> splitRect_1 = splitRect(split[i], center, 2, SplitMode::MODE_2);


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

					if (shape_1.size() != 0 && cx_2 > 0 && cy_2 > 0)
						centroids.push_back(center);
					else
						centroids.push_back(Point2f(0, 0));


					if (cx_2 > 0 && cy_2 > 0)
					{
						if (mode == CentroidDetectionMode::THREE_LOOP) // SPLIT ========================== = LEVEL 3 = ===============================
						{
							vector<Rect> splitRect_2 = splitRect(splitRect_1[j], center, 3, SplitMode::MODE_2);

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
								int cx_3, cy_3;

								if (shape_2.size() == 1) // THIRD LEVEL LOOP
								{
									// center of the contour
									Moments m = moments(shape_2[0], true);
									cx_3 = int(m.m10 / m.m00) + splitRect_2[k].x;
									cy_3 = int(m.m01 / m.m00) + splitRect_2[k].y;

								}
								else
								{
									// center of the image
									Moments m = moments(sub, true);
									cx_3 = int(m.m10 / m.m00) + splitRect_2[k].x;
									cy_3 = int(m.m01 / m.m00) + splitRect_2[k].y;
								}

								Point2f center(cx_3, cy_3);


								//Size size(box.x + box.width + cx, box.y + box.height + cy);
								//Mat img(size, CV_8UC1);
								//img = Scalar(0);
								//drawContours(img, tempVector, -1, cv::Scalar(255), 1, CV_AA);


								circle(img, center, 9, Scalar(255), -1, 8, 0);

								if (shape_2.size() != 0 && cx_3 > 0 && cy_3 > 0)
									centroids.push_back(center);
								else
									centroids.push_back(Point2f(0, 0));


							}

						}//  ========================== = LEVEL 3 = ===============================	

					}
					else
					{
						for (int i = 0; i < 4; i++)
						{
							centroids.push_back(Point2f(0, 0));
						}
					}


				}

			}// ========================== = LEVEL 2 = ===============================	
		}
		else
		{
			int loop = (mode == TWO_LOOP) ? 4 : 16;
			
			for (int i = 0; i < loop; i++)
			{
				centroids.push_back(Point2f(0, 0));
			}
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
		{
			distances.push_back(0);
			continue;
		}

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
		{
			angles.push_back(numeric_limits<double>::max());
			continue;
		}

		double cat1 = (distribution[i].x - center.x);
		double cat2 = (distribution[i].y - center.y);
		double ip = sqrt(cat1*cat1 + cat2*cat2);
		double ang = asin(cat1 / ip);

		angles.push_back(ang);
	}

	return angles;
}

// --------------------------------------------------------------------------------------------------------------------------

double Utility::checkKeyPointsMatch(std::vector<cv::KeyPoint> &contour, std::vector<cv::KeyPoint> &base)
{
	vector<Point2f> contourK, baseK;
	double delta = 40;


	for (int i = 0; i < base.size(); i++)
		baseK.push_back(base[i].pt);
	
	for (int i = 0; i < contour.size(); i++)
		contourK.push_back(contour[i].pt);

	Rect baseRect = boundingRect(baseK);
	Rect contourRect = boundingRect(contourK);

	Mat H = findHomography(contourK, baseK, 0);

	
	vector<Point2f> contourKT;
	perspectiveTransform(contourK, contourKT, H);

#ifdef DEBUG_MODE
	Size bs(baseRect.x + baseRect.width, baseRect.y + baseRect.height);
	Mat tempImg(bs, CV_8UC3);
	tempImg = Scalar(0);

	for (int i = 0; i < baseK.size(); i++)
		circle(tempImg, baseK[i], 6, Scalar(0, 0, 255), -1, 8, 0);

	for (int i = 0; i < contourK.size(); i++)
	{
		circle(tempImg, contourKT[i], 3, Scalar(0, 255, 0), -1, 8, 0);
		circle(tempImg, contourK[i], 3, Scalar(0, 255, 0), -1, 8, 0);
	}
#endif


	map<int, double> matchMap;
	vector<Point2f> goodMatch;
	int tot = 0;

	for (int i = 0; i < contourKT.size(); i++)
	{
		
		double minDist = numeric_limits<double>::max();
		Point2f matchBasePoint(0, 0);
		int matchID = -1;

		for (int j = 0; j < baseK.size(); j++)
		{
			double cat1 = (contourKT[i].x - baseK[j].x);
			double cat2 = (contourKT[i].y - baseK[j].y);
			double dist = sqrt(cat1*cat1 + cat2*cat2);

			if (dist < minDist && matchMap.count(j) == 0)
			{
				minDist = dist;
				matchBasePoint = baseK[j];
				matchID = j;
			}
		}

		if (minDist <= delta)
		{
			matchMap.insert(pair<int, double>(matchID, minDist));
			//goodMatch.push_back(matchBasePoint);
			goodMatch.push_back(contourKT[i]);
			tot++;
		}
		else
		{
			//goodMatch.push_back(Point2f(0, 0));
			goodMatch.push_back(baseK[i]);

		}
		
	}

	double matchPercentage = tot * 100 / baseK.size();

	return matchPercentage/2;
}