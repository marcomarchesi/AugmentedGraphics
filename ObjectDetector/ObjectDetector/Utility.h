#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
//#include "opencv2/features2d/features2d.hpp"
#include "opencv2/features2d.hpp"

namespace od
{
	class Utility{

		

	public:

		enum CentroidDetectionMode{
			ONE_LOOP,
			TWO_LOOP
		};

		/*
		uses the cv::matchShape to calculate the hamming distance between the contour and the baseShape
		retrun a the percentage value of the hamming distance
		@contour: one of the contour of the input image
		@base: the contour of the base image
		*/
		double static calculateContourPercentageCompatibility(std::vector<cv::Point> contour, std::vector<cv::Point> base);

		/*
		calculate che correlation between two distributions of points
		@contour: one of the contour of the input image
		@base: the contour of the base image
		*/
		double static correlationWithBase(std::vector<cv::Point> contour, std::vector<cv::Point> baseContour);
		double static correlationWithBaseMatcher(std::vector<cv::Point> contour, std::vector<cv::Point> baseContour);

		/*
		find the centroids distribution of a contour
		generate a vector of 9 points
		@contour: a vector of point that represent the shape of an object
		*/
		std::vector<cv::Point> static findCentroidsDistribution(std::vector<cv::Point> contour);

		std::vector<cv::Point> static findLargeCentroidsDistribution(std::vector<cv::Point> contour);

		void static findCentroidsKeypoints(std::vector<cv::Point> contour,
														std::vector<cv::KeyPoint>& centroids,
														CentroidDetectionMode mode);

	private:

		double static pearsonCorrelation(std::vector<cv::Point>& distribution, std::vector<cv::Point>& base);

		double static singlePearsonCorrelation(std::vector<double>& distribution, std::vector<double>& base);

		double static spearmanCorrelation(std::vector<cv::Point>& distribution, std::vector<cv::Point>& base);

		double static singleSpearmanCorrelation(std::vector<double>& distribution, std::vector<double>& base);

		std::vector<cv::Point> static normalize(std::vector<cv::Point> source);

		std::vector<double> static findDistancesFromCenter(std::vector<cv::Point> distribution);

		std::vector<double> static findAnglesRespectCenter(std::vector<cv::Point> distribution);

		bool static isLessThan(cv::Point a, cv::Point b, cv::Point center);

		std::vector<cv::Point> static sortPoints(std::vector<cv::Point> points);

		double static calculateHausdorffDistance(std::vector<cv::Point> contour, std::vector<cv::Point> base);
	};
}