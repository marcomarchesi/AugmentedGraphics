
#include "ObjectDetector.h"

namespace od
{
	class CategoryTester
	{
	public:
	
		CategoryTester(od::ObjectDetector* detector);

		CategoryTester(od::ObjectDetector* detector, char* datasetDirectory);

		CategoryTester(od::ObjectDetector* detector, char* datasetDirectory, double hammingThreshold, double correlationThreshold);

		CategoryTester(od::ObjectDetector* detector, double hammingThreshold, double correlationThreshold);

		std::vector<std::string> loadCategories();

		bool setCategory(char* category);

		virtual double startTest() = 0;

	protected:

		char* _datasetDirectory;
		std::vector<std::string> _categories;
		char* _choosenCategory;

		od::ObjectDetector* _detector;
		double _hammingThreshold;
		double _correlationThreshold;
	};
}


