#include "opencv2/opencv.hpp"
#include "ObjectDetector.h"
#include "ObjectdetectorFactory.h"

#include "opencv2/highgui/highgui.hpp"

#include <fstream>
#include <string>
#include <time.h>

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

typedef struct BaseImage
{
	string filename;
	int minContourPoints;
	int contoursNumber;
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

	if (flags == EVENT_FLAG_LBUTTON)
	{
		if (x >= 495 && x <= 600 && y >= 0 && y <= 32)
		{
			input = Scalar(255, 255, 255);
			output = Scalar(0, 0, 0);
			putText(input, "CLEAR", Point(495, 30), FONT_HERSHEY_COMPLEX, 1.0f, Scalar(0, 0, 0), 2);
		}

		if (event == EVENT_MOUSEMOVE)	//press left button and drag
			circle(input, Point(x, y), 1, color, -1);
	}
	else if (event == EVENT_LBUTTONUP)
	{
		vector<vector<vector<Point>>> objects;
		int numberOfObjects = 0;
		Mat mask = detector->findObjectsInImage(input, 1, 20, &objects, &numberOfObjects);

#ifdef DEBUG_MODE
		Mat out(output.size(), output.type());
		out = Scalar(0, 0, 0);
		for (int i = 0; i < objects.size(); i++)
		{
			drawContours(out, objects[i], -1, Scalar(255, 255, 255), -1, CV_AA);
		}
		imshow("detected", out);
#endif

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
		outputImage(size, CV_8UC4);


	drawingArea = Scalar(255, 255, 255);
	outputImage = Scalar(0, 0, 0,0);
	putText(drawingArea, "CLEAR", Point(495, 30), FONT_HERSHEY_COMPLEX, 1.0f, Scalar(0, 0, 0), 2);
	
	// SHOW COLOR PALETTE ----------------------------------------------

	Mat palette = imread("palette.jpg");
	namedWindow("palette", 1);
	ColorPicker* colorPicker = new ColorPicker;
	colorPicker->palette = palette;
	colorPicker->color = new Scalar(0, 0, 0);
	setMouseCallback("palette", changeColorCallback, colorPicker);
	
	moveWindow("palette", 0, 50);
	// ----------------------------------------------------------------

	// READ BASE IMAGES -----------------------------------------------

	ifstream in("shapes.txt");

	vector<BaseImage> images;
	int i = 0;
	while (in)
	{
		char temp[256];
		in.getline(temp, 256);

		char* tok;
		BaseImage b;
		i = 0;
		tok = strtok(temp, " ");
		while (tok != NULL)
		{
			if (i == 0)
				b.filename = tok;
			else if (i == 1)
				b.minContourPoints = atoi(tok);
			else
				b.contoursNumber = atoi(tok);

			tok = strtok(NULL, " ");
			i++;
		}
		images.push_back(b);
	}
	in.close();
	
	

	// ----------------------------------------------------------------

	srand(time(NULL));
	int id = rand() % (images.size()-1);	

	Mat bi = imread(images[id].filename);
	if (bi.empty())
		exit(1);

	ObjectDetector* detector = ObjectDetectorFactory::getObjectDetector(images[id].minContourPoints, images[id].contoursNumber);
	if (!detector->loadImage(bi))
		exit(1);

	namedWindow("Base", 1);
	moveWindow("Base", 1000, 800);
	imshow("Base", bi);

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
	moveWindow("welcome", 500, 100);
	imshow("welcome", homeImage);
	waitKey(0);
	return 0;
}