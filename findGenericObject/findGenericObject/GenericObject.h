#include "opencv2/opencv.hpp"


class GenericObject{

public:

	/*	
		use the findObjectShape to found the base object in the input image
		@baseImage: a color or grayscale image with the basic shape that you are
		looking for
		@aspectedContoursPoint: the number of points that you know the shape has
	*/	
	GenericObject(cv::Mat baseImage, int aspectedContoursPoint);

	
	/*
		find all the object contours in the input vector, uses cv::matchShape and centroids distribution correlation
		@contours: a vector of contours from the input image
		@hammingThreshold: percentage threshold for the matchShape
		@correlationThreshold: percentage threshold for the centroids distribution correlation
	*/
	std::vector<std::vector<cv::Point>> findObjectsInContours(std::vector<std::vector<cv::Point>> contours, double hammingThreshold, double correlationThreshold);

	/*	
		do all the work, find the objects in the input image
		return the same image in grayscale with objects marked with a white line
		@img: input image
		@hammingThreshold: percentage threshold for the matchShape
		@correlationThreshold: percentage threshold for the centroids distribution correlation
	*/
	cv::Mat findObjectsInImg(cv::Mat img, double hammingThreshold, double correlationThreshold);	

private:

	/*
		find the object shape in the input image
		@baseImage: a color or grayscale image with the basic shape that you are
		looking for
		@aspectedContoursPoint: the number of points that you know the shape has
	*/
	void findObjectShape(cv::Mat baseImage, int aspectedContoursPoint);

	// display a contour in an image, used in DEBUG_MODE
	void showContour(std::vector<cv::Point> contour, cv::Size size);

	cv::Size baseSize, imgSize;

	std::vector<cv::Point> baseShape;

	double focus,
		minFocus,
		maxFocus;
};