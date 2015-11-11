#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
//#include "opencv2/features2d/features2d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/shape.hpp"

namespace od
{
	class Utility{

		

	public:

		enum CentroidDetectionMode{
			ONE_LOOP,
			TWO_LOOP,
			THREE_LOOP
		};

		enum HammingMode{
			CV_CONTOURS_MATCH_I1,
			CV_CONTOURS_MATCH_I3
		};

		enum SplitMode{
			MODE_1,
			MODE_2
		};

		/*
		uses the cv::matchShape to calculate the hamming distance between the contour and the baseShape
		retrun a the percentage value of the hamming distance
		@contour: one of the contour of the input image
		@base: the contour of the base image
		*/
		double static calculateContourPercentageCompatibility(std::vector<cv::Point> contour, std::vector<cv::Point> base);
		double static calculateContourPercentageCompatibility(std::vector<cv::Point> contour, std::vector<cv::Point> base, HammingMode mode);

		/*
		calculate che correlation between two distributions of points
		@contour: one of the contour of the input image
		@base: the contour of the base image
		*/
		double static correlationWithBase(std::vector<cv::Point> contourKeypoints, std::vector<cv::Point> baseKeypoints);
		double static correlationWithBaseMatcher(std::vector<cv::Point> contour, std::vector<cv::Point> baseContour);

		/*
		find the centroids distribution of a contour
		generate a vector of 9 points
		@contour: a vector of point that represent the shape of an object
		*/
		std::vector<cv::Point> static findCentroidsDistribution(std::vector<cv::Point> contour);

		std::vector<cv::Point> static findLargeCentroidsDistribution(std::vector<cv::Point> contour);

		void static findCentroidsKeypoints(std::vector<cv::Point> contour,
														std::vector<cv::Point>& centroids,
														CentroidDetectionMode mode);

	private:

		std::vector<cv::Rect> static splitRect(cv::Rect box, cv::Point centroid, int level, SplitMode mode);

		double static pearsonCorrelation(std::vector<cv::Point>& distribution, std::vector<cv::Point>& base);

		double static singlePearsonCorrelation(std::vector<double>& distribution, std::vector<double>& base);

		double static spearmanCorrelation(std::vector<cv::Point>& distribution, std::vector<cv::Point>& base);

		double static singleSpearmanCorrelation(std::vector<double>& distribution, std::vector<double>& base);

		std::vector<double> static findDistancesFromCenter(std::vector<cv::Point> distribution);

		std::vector<double> static findAnglesRespectCenter(std::vector<cv::Point> distribution);

		double static calculateHausdorffDistance(std::vector<cv::Point> contour, std::vector<cv::Point> base);

		double static checkKeyPointsMatch(std::vector<cv::KeyPoint> &contour, std::vector<cv::KeyPoint> &base);

	};
}