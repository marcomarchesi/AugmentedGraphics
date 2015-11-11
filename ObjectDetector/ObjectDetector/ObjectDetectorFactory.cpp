#include "ObjectDetectorFactory.h"
#include "MultiContourObjectDetector.h"
#include "MonoContourObjectDetector.h"
#include "commonInclude.h"


using namespace std;
using namespace od;

ObjectDetector* ObjectDetectorFactory::getObjectDetector(int contoursNumber)
{
	if (contoursNumber == 1)
		return (ObjectDetector*)new MonoContourObjectDetector(contoursNumber);

	else if (contoursNumber > 1)
		return (ObjectDetector*)new MultiContourObjectDetector(contoursNumber);

	else
	{
		cerr << "invalid contoursNumber" << endl;
		return NULL;
	}
}