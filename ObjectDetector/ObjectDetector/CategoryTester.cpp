#include "CategoryTester.h"
#include "commonInclude.h"

#include <windows.h>
#include <tchar.h> 
#include <stdio.h>
#include <strsafe.h>

using namespace std;
using namespace cv;
using namespace od;

CategoryTester::CategoryTester(od::ObjectDetector* detector) :
			_detector(detector),
			_datasetDirectory("C:\\Users\\Notebook\\Desktop\\TESI\\GITHUB\\101_test"),
			_hammingThreshold(75.0),
			_correlationThreshold(50.0)
{}

CategoryTester::CategoryTester(od::ObjectDetector* detector, char* datasetDirectory) :
			_detector(detector),
			_datasetDirectory(datasetDirectory),
			_hammingThreshold(75.0),
			_correlationThreshold(50.0)
{}

CategoryTester::CategoryTester(od::ObjectDetector* detector, double hammingThreshold, double correlationThreshold) :
			_detector(detector),
			_datasetDirectory("C:\\Users\\Notebook\\Desktop\\TESI\\GITHUB\\101_test"),
			_hammingThreshold(hammingThreshold),
			_correlationThreshold(correlationThreshold)
{}

CategoryTester::CategoryTester(od::ObjectDetector* detector, char* datasetDirectory, double hammingThreshold, double correlationThreshold) :
			_detector(detector),
			_datasetDirectory(datasetDirectory),
			_hammingThreshold(hammingThreshold),
			_correlationThreshold(correlationThreshold)
{}

std::vector<std::string> CategoryTester::loadCategories()
{
	_categories.clear();

	TCHAR szDir[MAX_PATH];
	WIN32_FIND_DATA ffd;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	StringCchCopy(szDir, MAX_PATH, _datasetDirectory);
	StringCchCat(szDir, MAX_PATH, TEXT("\\*"));

	hFind = FindFirstFile(szDir, &ffd);

	
	while (FindNextFile(hFind, &ffd) != 0)
	{
		if (ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
		{
			_categories.push_back(ffd.cFileName);
		}
	}
	FindClose(hFind);

	return _categories;
}

bool CategoryTester::setCategory(char* category)
{
	bool find = false;

	for (int i = 0; i < _categories.size() && !find; i++)
	if (category == _categories[i])
		find = true;

	if (find)
	{
		StringCchCopy(_choosenCategory, MAX_PATH, _datasetDirectory);
		StringCchCat(_choosenCategory, MAX_PATH, TEXT("\\"));
		StringCchCat(_choosenCategory, MAX_PATH, category);
		StringCchCat(_choosenCategory, MAX_PATH, TEXT("\\*"));
	}

	return find;
}