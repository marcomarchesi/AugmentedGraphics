#include "ObjectDetector.h"

class ObjectDetectorFactory
{
public:
	static ObjectDetector* getObjectDetector(int minContourPoints, int contoursNumber);
};