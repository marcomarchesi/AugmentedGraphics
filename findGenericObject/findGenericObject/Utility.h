#include "opencv2/opencv.hpp"

class Utility{


public:

	// uses the cv::matchShape to calculate the hamming distance between the contour and the baseShape
	double calculateContourPercentageCompatibility(std::vector<cv::Point> contour, std::vector<cv::Point> base);

	// calculate che correlation between two distributions of points
	double correlationWithBase(std::vector<cv::Point> contour, std::vector<cv::Point> baseContour);

	// find the centroids distribution of a contour
	// generate a vector of 9 points
	std::vector<cv::Point> findCentroidsDistribution(std::vector<cv::Point> contour);


};