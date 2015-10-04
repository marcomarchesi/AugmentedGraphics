#include "opencv2/opencv.hpp"


class Star{

public:

	// use the findStarShape to isolate the star in the Base image
	Star(cv::Mat starImage);

	// calculate the compatibility between a contour and the base starShape
	double checkContourPercentageCompatibility(std::vector<cv::Point> contour);
	
	// find all the star contour in contours extracted from an image
	std::vector<std::vector<cv::Point>> findStarsInContours(std::vector<std::vector<cv::Point>> contours, double precision, cv::Mat gray);

	// do all the work, find the stars in the input image
	// return the same image in grayscale with stars marked with a white line
	cv::Mat findStarInImg(cv::Mat img, double precision);

private:

	//find star shape in the base image
	void findStarShape(cv::Mat starImage);

	// display a contour in an image, used in DEBUG_MODE
	void showContour(std::vector<cv::Point> contour, cv::Size size);

	std::vector<cv::Point> starShape;
	int thresholdMinValue;
	double epsilonFactor;

	cv::Size baseSize, imgSize;
	
};