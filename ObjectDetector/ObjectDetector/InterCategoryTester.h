#include "CategoryTester.h"

#include "opencv2/opencv.hpp"



namespace od
{
	class InterCategoryTester : public CategoryTester{

	public:

		InterCategoryTester(od::ObjectDetector* detector);

		InterCategoryTester(od::ObjectDetector* detector, char* datasetDirectory);

		InterCategoryTester(od::ObjectDetector* detector, char* datasetDirectory, double hammingThreshold, double correlationThreshold);

		InterCategoryTester(od::ObjectDetector* detector, double hammingThreshold, double correlationThreshold);

		double categoryDetectionRate();
	};
}