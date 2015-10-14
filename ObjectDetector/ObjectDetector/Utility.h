#include "opencv2/opencv.hpp"

class Utility{


public:

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

	/* 
		find the centroids distribution of a contour 
		generate a vector of 9 points
		@contour: a vector of point that represent the shape of an object
	*/
	std::vector<cv::Point> static findCentroidsDistribution(std::vector<cv::Point> contour);


};