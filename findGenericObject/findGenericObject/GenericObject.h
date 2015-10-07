#include "opencv2/opencv.hpp"


class GenericObject{

public:

	// use the findStarShape to isolate the star in the Base image
	GenericObject(cv::Mat baseImage);

	
	// find all the star contour in contours extracted from an image
	std::vector<std::vector<cv::Point>> findObjectsInContours(std::vector<std::vector<cv::Point>> contours, double precision, cv::Mat gray);

	// do all the work, find the stars in the input image
	// return the same image in grayscale with stars marked with a white line
	cv::Mat findObjectsInImg(cv::Mat img, double precision);	

private:

	//find star shape in the base image
	void findObjectShape(cv::Mat starImage);

	// display a contour in an image, used in DEBUG_MODE
	void showContour(std::vector<cv::Point> contour, cv::Size size);

	cv::Size baseSize, imgSize;

	vector<Point> baseShape;

	double focus,
		minFocus,
		maxFocus;
};