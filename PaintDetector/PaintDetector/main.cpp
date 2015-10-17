#include "opencv2/opencv.hpp"
#include "ObjectDetector.h"
#include "ObjectdetectorFactory.h"
using namespace cv;
using namespace std;

typedef struct CallbackParams
{
	Mat input;
	Mat output;
	ObjectDetector* detector;
	Scalar* color;
};

typedef struct ColorPicker
{
	Mat palette;
	Scalar* color;
};

void changeColorCallback(int event, int x, int y, int flags, void* userdata)
{
	if (event != EVENT_LBUTTONUP)
		return;

	ColorPicker* picker = (ColorPicker*)userdata;

	Mat palette = picker->palette;

	Rect point(x, y, 1,1);
	Mat pixel = palette(point);

	*picker->color = mean(pixel);

}

void paintCallback(int event, int x, int y, int flags, void* userdata)
{
	CallbackParams* params = (CallbackParams*)userdata;

	Mat input = params->input;
	Mat output = params->output;
	ObjectDetector* detector = params->detector;
	Scalar color = *params->color;

	if (event == EVENT_MOUSEMOVE && flags == EVENT_FLAG_LBUTTON)
	{
		//press left button and drag
		circle(input, Point(x, y), 1, color, -1);
	}
	else if (event == EVENT_LBUTTONUP)
	{
		vector<vector<vector<Point>>> objects;
		int numberOfObjects = 0;
		Mat mask = detector->findObjectsInImage(input, 70, 90, &objects, &numberOfObjects);
		output += mask;
	}
}

void START(int event, int x, int y, int flags, void* userdata)
{
	if (event != CV_EVENT_LBUTTONDOWN)
		return;

	destroyWindow("welcome");

	
	Size size(600, 600);
	Mat drawingArea(size, CV_8UC3),
		outputImage(size, CV_8UC3);


	drawingArea = Scalar(255, 255, 255);
	outputImage = Scalar(0, 0, 0);

	Mat baseImage = imread("ninja.jpg");
	if (baseImage.empty())
		exit(1);

	ObjectDetector* detector = ObjectDetectorFactory::getObjectDetector(8, 2);
	if (!detector->loadImage(baseImage))
		exit(1);
	
	// SHOW COLOR PALETTE ----------------------------------------------

	Mat palette = imread("palette.jpg");
	namedWindow("palette", 1);
	ColorPicker* colorPicker = new ColorPicker;
	colorPicker->palette = palette;
	colorPicker->color = new Scalar(0, 0, 0);
	setMouseCallback("palette", changeColorCallback, colorPicker);
	
	moveWindow("palette", 0, 50);
	// ----------------------------------------------------------------


	CallbackParams* params = new CallbackParams;
	params->input = drawingArea;
	params->output = outputImage;
	params->detector = detector;
	params->color = colorPicker->color;

	namedWindow("Paint it!!", 1);
	namedWindow("Detected Result", 1);

	moveWindow("Paint it!!", 100, 50);
	moveWindow("Detected Result", 705, 50);

	setMouseCallback("Paint it!!", paintCallback, params);

	for (;;)
	{
		imshow("palette", palette);
		imshow("Paint it!!", drawingArea);
		imshow("Detected Result", outputImage);

		if (waitKey(30) >= 0)
			break;
	}
	waitKey(0);
}



int main(int, char)
{
		
	Mat homeImage = imread("home.jpg");
	if (homeImage.empty())
		exit(1);

	putText(homeImage, "CLICK TO START", Point(100, 409), FONT_HERSHEY_COMPLEX, 1.0f, Scalar(0, 0, 0));
	namedWindow("welcome", 1);

	setMouseCallback("welcome", START);

	imshow("welcome", homeImage);
	waitKey(0);
	return 0;
}