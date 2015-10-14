#include "ObjectDetector.h"

class MultiContourObjectDetector : public ObjectDetector{

public:

	MultiContourObjectDetector(int minContourPoints, int aspectedContours);

private:

	bool findBaseShape(cv::Mat& baseImage);

	std::vector<std::vector<std::vector<cv::Point>>> findApproxContours(
		cv::Mat image,
		bool performOpening,
		int minThresholdValue);

	std::vector<std::vector<std::vector<cv::Point>>> processContours(
		std::vector<std::vector<std::vector<cv::Point>>> approxContours,
		double hammingThreshold,
		double correlationThreshold);

	cv::Mat generateDetectionMask(
		std::vector<std::vector<std::vector<cv::Point>>> detectedObjects,
		cv::Size imageSize,
		int type);


	std::vector<std::vector<cv::Point>> _baseShape;
};