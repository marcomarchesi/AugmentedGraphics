#include "opencv2/opencv.hpp"

class MultiContoursObject
{
public:
	/*
	use the findObjectShape to found the base object in the input image
	@baseImage: a color or grayscale image with the basic shape that you are
	looking for
	@minContourPoint: min points for contour
	*/
	MultiContoursObject(cv::Mat baseImage, int minContourPoint, int aspectedContours);


	/*
	find all the object contours in the input vector, uses cv::matchShape and centroids distribution correlation
	@hierachyContours: contours devided by hierarchy level
	@hammingThreshold: percentage threshold for the matchShape
	@correlationThreshold: percentage threshold for the centroids distribution correlation
	*/
	std::vector<std::vector<std::vector<cv::Point>>> findObjectsInContours(std::vector<std::vector<std::vector<cv::Point>>> hierachyContours, double hammingThreshold, double correlationThreshold);

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
	@minContourPoint: min points for contour
	*/
	void findObjectShape(cv::Mat baseImage, int minContourPoint, int aspectedContours);

	// display a contour in an image, used in DEBUG_MODE
	void showContours(std::vector<std::vector<cv::Point>> contours, cv::Size size);

	cv::Size baseSize, imgSize;

	std::vector<std::vector<cv::Point>> baseShape;

	std::vector<cv::Point> baseTotal;

};