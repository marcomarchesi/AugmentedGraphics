#include "MultiContoursObject.h"
#include "Utility.h"
#include "commonInclude.h"

using namespace cv;
using namespace std;

MultiContoursObject::MultiContoursObject(cv::Mat baseImage, int minContourPoint, int aspectedContours)
{
	if (baseImage.empty())
	{
		cout << "the Base Image is empty" << endl;
		waitKey(30);
		exit(1);
	}

	findObjectShape(baseImage, minContourPoint, aspectedContours);
	showContours(baseShape, baseImage.size());

	/*for (int i = 0; i < baseShape.size(); i++)
	{
		for (int j = 0; j < baseShape[i].size(); j++)
		{
			baseTotal.push_back(baseShape[i][j]);
		}
	}*/

}


std::vector<std::vector<std::vector<cv::Point>>> MultiContoursObject::findObjectsInContours(
	std::vector<std::vector<std::vector<cv::Point>>> hierachyContours,
	double hammingThreshold,
	double correlationThreshold)
{
	Utility utility;

	vector<vector<vector<Point>>> objects;
	
	for (int i = 0; i < hierachyContours.size(); i++)
	{
		if (hierachyContours[i].size() != baseShape.size())
			continue;

		double totCorrelation = 0,
			totHamming = 0;

		// C and H with external contour
		totCorrelation += utility.correlationWithBase(hierachyContours[i][0], baseShape[0]);
		totHamming += utility.calculateContourPercentageCompatibility(hierachyContours[i][0], baseShape[0]);

		// looking for the contour with the better cnetroids and shape match

		for (int j = 1; j < hierachyContours[i].size(); j++)
		{
			double maxCorrelation = numeric_limits<double>::min(),
				maxHamming = numeric_limits<double>::min();

			for (int k = 1; k < baseShape.size(); k++)
			{
				maxCorrelation = max(maxCorrelation, utility.correlationWithBase(hierachyContours[i][j], baseShape[k]));
				maxHamming = max(maxHamming, utility.calculateContourPercentageCompatibility(hierachyContours[i][j], baseShape[k]));
			}

			totCorrelation += maxCorrelation;
			totHamming += maxHamming;
		}

		totCorrelation /= hierachyContours[i].size();
		totHamming /= hierachyContours[i].size();

		cout << "Middle Correlation " << to_string(i) << " with base ---> " << totCorrelation << endl;
		cout << "Middle Hamming distance" << to_string(i) << " with base ---> " << totHamming << endl;

		if (totCorrelation >= correlationThreshold && totHamming >= hammingThreshold)
			objects.push_back(hierachyContours[i]);
	}

	return objects;
}


cv::Mat MultiContoursObject::findObjectsInImg(cv::Mat img, double hammingThreshold, double correlationThreshold)
{
	
	if (img.size().height > 800 || img.size().width > 800)
	{
		Size s = img.size(), small;
		small.height = s.height / 2;
		small.width = s.width / 2;

		resize(img, img, small);
	}
	
	imgSize = img.size();
	Mat gray(img.size(), CV_8UC1);
	Mat thresh(img.size(), CV_8UC1);

	cvtColor(img, gray, CV_BGR2GRAY);

	
	// PERFORM OPENING (Erosion --> Dilation)
	
	int erosion_size = 4;
	int dilation_size = 4;

	Mat element = getStructuringElement(0, Size(2 * erosion_size, 2 * erosion_size), Point(erosion_size, erosion_size));
	erode(gray, gray, element);
	dilate(gray, gray, element);
	
	// need a loop that decreese the threshold min value if the image is too black
	// for now 60 is ok
	threshold(gray, thresh, 100, 255, THRESH_BINARY);

#ifdef DEBUG_MODE
	imshow("Threshold", thresh);
#endif

	//imshow("Thresh", thresh);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<Point> approx;

	map<int, vector<vector<Point>>> hierachedContours;
	map<int, vector<vector<Point>>> approxHContours;

	
	try
	{
		findContours(thresh, contours, hierarchy, CV_RETR_TREE, CHAIN_APPROX_NONE);
	}
	catch (const Exception& e)
	{
		cerr << e.what();
	}


#ifdef DEBUG_MODE
	Mat image(img.size(), CV_8UC1);
	image = Scalar(0);
	drawContours(image, contours, -1, cv::Scalar(255), 1, CV_AA);

	imshow("Contours", image);
#endif

	
	vector<vector<Point>> temp;
	// CATALOG BY HIERARCHY LOOP
	for (int i = 0; i < contours.size(); i++)
	{
		image = Scalar(0);
		temp.clear();
		temp.push_back(contours[i]);
		drawContours(image, temp, -1, cv::Scalar(255), 1, CV_AA);

		int parent = hierarchy[i][3];
		if (parent == -1)
		{
			if (hierachedContours.count(i) == 0)
			{
				// me not found

				hierachedContours.insert(pair<int, vector<vector<Point>>>(i, vector<vector<Point>>()));
				hierachedContours[i].push_back(contours[i]);
			}
			else
			{
				// me found
				continue;
			}
		}
		else
		{
			if (hierachedContours.count(parent) == 0)
			{
				// dad not found
				hierachedContours.insert(pair<int, vector<vector<Point>>>(parent, vector<vector<Point>>()));
				hierachedContours[parent].push_back(contours[parent]);
			}
			hierachedContours[parent].push_back(contours[i]);
		}
	}


	// APPROX LOOP
	
	for (map<int, vector<vector<Point>>>::iterator it = hierachedContours.begin(); it != hierachedContours.end(); it++)
	{

#ifdef DEBUG_MODE
		image = Scalar(0);
		drawContours(image, it->second, -1, cv::Scalar(255), 1, CV_AA);
#endif

		for (int k = 0; k < it->second.size(); k++)
		{
			if (it->second[k].size() < 4)
			{
				if (k == 0) // padre
					break;
				else        // figlio
					continue;
			}

			double epsilon = it->second[k].size() * 0.03;
			approxPolyDP(it->second[k], approx, epsilon, true);

#ifdef DEBUG_MODE			
			image = Scalar(0);
			vector<vector<Point>> temp;
			temp.push_back(approx);
			drawContours(image, temp, -1, cv::Scalar(255), 1, CV_AA);
#endif
			if (approx.size() < 4)
			{
				if (k == 0) // padre
					break;
				else        // figlio
					continue;
			}


			if (k == 0)
			{
				approxHContours.insert(pair<int, vector<vector<Point>>>(it->first, vector<vector<Point>>()));
				approxHContours.at(it->first).push_back(approx);
			}
			else
			{
				approxHContours[it->first].push_back(approx);
			}
		}
	}

	
	vector<vector<vector<Point>>> lookupVector;
	for (map<int, vector<vector<Point>>>::iterator it = approxHContours.begin(); it != approxHContours.end(); it++)
	{
		if (it->second.size() <= 1)
			continue;
		lookupVector.push_back(it->second);
	}

	vector<vector<vector<Point>>> objects = findObjectsInContours(lookupVector, hammingThreshold, correlationThreshold);
		

	// FIND OBJECT LOOP

	
	
	Mat starMask(gray.size(), gray.type());
	starMask = Scalar(0);

	for (int i = 0; i < objects.size(); i++)
	{
		for (int j = 0; j < objects[i].size(); j++)
		{
			for (int k = 0; k < objects[i][j].size(); k++)
			{
				line(starMask, objects[i][j][k], objects[i][j][(k + 1) % objects[i][j].size()], Scalar(255), 2, CV_AA);
			}
		}
	}

	gray += starMask;

	return gray;

	

	
	
}


void MultiContoursObject::findObjectShape(cv::Mat baseImage, int minContourPoint, int aspectedContours)
{
	Mat thresh(baseImage.size(), CV_8UC1);
	cvtColor(baseImage, thresh, CV_BGR2GRAY);

	/*
	int erosion_size = 5;
	int dilation_size = 5;

	Mat element = getStructuringElement(0, Size(2 * erosion_size, 2 * erosion_size), Point(erosion_size, erosion_size));
	erode(thresh, thresh, element);
	dilate(thresh, thresh, element);
	*/

	threshold(thresh, thresh, 150, 255, THRESH_BINARY);

	vector<vector<Point>> contours, approxTemp;
	vector<Vec4i> hierarchy;
	vector<Point> approx;

	map<int, vector<vector<Point>>> hierachedContours;
	map<int, vector<vector<Point>>> approxHContours;
		
	findContours(thresh, contours, hierarchy, CV_RETR_TREE, CHAIN_APPROX_NONE);
		
	Mat image(baseImage.size(), CV_8UC1);
	image = Scalar(0);
	
	

	for (int i = 0; i < contours.size(); i++)
	{	
		/*
			Se è padre
				Se c'è già id i
					passo avanti
				Se non c'è
					aggiungo vettore con id i
					ci metto il padre
			Se è figlio
				Se c'è già mio padre
					inserisco figlio in coda al padre
				Se non c'è padre
					aggiungo vettore con id padre
					aggiungo padre
					aggiungo figlio in coda a padre
		*/

		int parent = hierarchy[i][3];
		if (parent == -1)
		{
			if (hierachedContours.count(i) == 0)
			{
				// me not found

				hierachedContours.insert(pair<int, vector<vector<Point>>>(i, vector<vector<Point>>()));
				hierachedContours[i].push_back(contours[i]);
			}
			else
			{
				// me found
				continue;
			}
		}
		else
		{
			if (hierachedContours.count(parent) == 0)
			{
				// dad not found
				hierachedContours.insert(pair<int, vector<vector<Point>>>(parent, vector<vector<Point>>()));
				hierachedContours[parent].push_back(contours[parent]);
			}
			hierachedContours[parent].push_back(contours[i]);
		}
	}

	/*
		Per ogni oggetto in gerarchia
			Per ogni contorno dell'oggetto
				Se il contorno ha meno di 4 punti
					Se è il padre
						rimuovi oggetto
					Se è figlio
						rimuovi contorno

				Approssima contorno
				
				Se il contorno ha meno dei punti minimi
					Se è il padre
						rimuovi oggetto
					Se è figlio
						rimuovi contorno
				
				Se è padre
					aggiungi nuovo oggetto approssimato
					aggiungi padre
				Se è figlio
					aggiungi figlio
	*/
	
	/*
	for (int i = 0; i < hierachedContours.size(); i++)
	{
		for (int k = 0; k < hierachedContours[i].size(); k++)
		{
			if (hierachedContours[i][k].size() < 4)
			{
				if (k == 0) // padre
					break;
				else        // figlio
					continue;
			}

			double epsilon = hierachedContours[i][k].size() * 0.03;
			approxPolyDP(hierachedContours[i][k], approx, epsilon, true);

			Mat contourImage(baseImage.size(), CV_8UC1);
			contourImage = Scalar(0);
			vector<vector<Point>> temp;
			temp.push_back(approx);
			drawContours(contourImage, temp, -1, cv::Scalar(255), 1, CV_AA);

			if (approx.size() < minContourPoint)
			{
				if (k == 0)
					break;
				else
					continue;
			}

			if (k == 0)
			{
				approxHContours.insert(pair<int, vector<vector<Point>>>(i, vector<vector<Point>>()));
				approxHContours[i].push_back(approx);
			}
			else
			{
				approxHContours[i].push_back(approx);
			}
		}			
	}
	*/

	for (map<int, vector<vector<Point>>>::iterator it = hierachedContours.begin(); it != hierachedContours.end(); it++)
	{
		for (int k = 0; k < it->second.size(); k++)
		{
			if (it->second[k].size() < 4)
			{
				if (k == 0) // padre
					break;
				else        // figlio
					continue;
			}

			double epsilon = it->second[k].size() * 0.03;
			approxPolyDP(it->second[k], approx, epsilon, true);

			Mat contourImage(baseImage.size(), CV_8UC1);
			contourImage = Scalar(0);
			vector<vector<Point>> temp;
			temp.push_back(approx);
			drawContours(contourImage, temp, -1, cv::Scalar(255), 1, CV_AA);

			if (approx.size() < minContourPoint)
			{
				if (k == 0) // padre
					break;
				else        // figlio
					continue;
			}


			if (k == 0)
			{
				approxHContours.insert(pair<int, vector<vector<Point>>>(it->first, vector<vector<Point>>()));
				approxHContours.at(it->first).push_back(approx);
			}
			else
			{
				approxHContours[it->first].push_back(approx);
			}
		}
	}

	if (approxHContours.size() == 0)
	{
		cout << "ERROR: No valid contours found in base image" << endl;
		Mat m = Mat(Scalar(255));
		imshow("ERROR", m);
		waitKey(0);
		exit(2);
	}
	
	

	for (map<int, vector<vector<Point>>>::iterator it = approxHContours.begin(); it != approxHContours.end(); it++)
	{
		if (it->second.size() == aspectedContours)
		{
			baseShape = it->second;
			break;
		}
	}	
}


void MultiContoursObject::showContours(std::vector<std::vector<cv::Point>> contours, cv::Size size)
{
	Mat contourImage(size, CV_8UC1);
	contourImage = Scalar(0);

	drawContours(contourImage, contours, -1, cv::Scalar(255), 1, CV_AA);
	imshow("Base", contourImage);
}