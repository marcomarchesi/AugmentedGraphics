#include "ObjectDetector.h"

namespace od
{
	class MultiContourObjectDetector : public ObjectDetector{

	public:

		MultiContourObjectDetector(int minContourPoints, int aspectedContours);

	private:

		bool findBaseShape(cv::Mat& baseImage);

		std::vector<std::vector<std::vector<cv::Point>>> findApproxContours(
			cv::Mat image,
			bool performOpening);

		void processContours(
			std::vector<std::vector<std::vector<cv::Point>>> approxContours,
			std::vector<std::vector<std::vector<cv::Point>>> &detectedObjects,
			double hammingThreshold,
			double correlationThreshold,
			int* numberOfObject);


		std::vector<std::vector<cv::Point>> _baseShape;
	};
}