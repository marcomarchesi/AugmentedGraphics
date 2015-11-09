#include "ObjectDetector.h"

namespace od
{
	class MonoContourObjectDetector : public ObjectDetector{

	public:

		MonoContourObjectDetector(int minContourPoints, int aspectedContours);

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



		std::vector<cv::Point> _baseShape;
		std::vector<cv::Point> _baseKeypoints;
		//std::vector<cv::Point> _originalBaseShape;

		//std::vector<std::vector<cv::Point>> _originalQueryShapes;
		
	};
}