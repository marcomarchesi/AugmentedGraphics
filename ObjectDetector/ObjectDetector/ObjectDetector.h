#ifndef OBJECT_DETECTOR
#define OBJECT_DETECTOR

#include "opencv2/opencv.hpp"

class ObjectDetector
{
public:
	
	ObjectDetector(int minContourPoints, int aspectedContours);
	
	cv::Mat findObjectsInImage(cv::Mat& image, double hammingThreshold, double correlationThreshold);

	bool loadImage(cv::Mat& baseImage);

private:

	virtual bool findBaseShape(cv::Mat& baseImage) = 0;

	virtual std::vector<std::vector<std::vector<cv::Point>>> findApproxContours(
						cv::Mat image,
						bool performOpening,
						int minThresholdValue) = 0;

	virtual std::vector<std::vector<std::vector<cv::Point>>> processContours(
						std::vector<std::vector<std::vector<cv::Point>>> approxContours,
						double hammingThreshold,
						double correlationThreshold) = 0;

	virtual cv::Mat generateDetectionMask(
						std::vector<std::vector<std::vector<cv::Point>>> detectedObjects,
						cv::Size imageSize,
						int type) = 0;

protected:
	cv::Size _baseSize;
	const int _minContourPoints;
	const int _aspectedContours;
};
#endif