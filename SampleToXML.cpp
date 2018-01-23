#include < stdio.h>
#include < opencv2\opencv.hpp>

using namespace cv;

#ifdef _DEBUG        
#pragma comment(lib, "opencv_core2413d.lib")         
#pragma comment(lib, "opencv_imgproc2413d.lib")  
#pragma comment(lib, "opencv_objdetect2413d.lib")
#pragma comment(lib, "opencv_highgui2413d.lib")        

#else        
#pragma comment(lib, "opencv_core2413.lib")        
#pragma comment(lib, "opencv_imgproc2413.lib")        
#pragma comment(lib, "opencv_objdetect2413.lib")        
#pragma comment(lib, "opencv_highgui2413.lib")        
#endif 

void main() {
	char FullFileName[100];
	char FirstFileName[100] = "C:\\Users\\user\\Desktop\\SVM_training\\neg-4D-lowVP\\neg ("; // samples' location 
	char SaveHogDesFileName[100] = "neg.xml"; // output XML fileName 
	int FileNum = 13611; // num of pics 

	vector< vector < float> > v_descriptorsValues;
	vector< vector < Point> > v_locations;
	vector<float> descriptorsValues;
	vector<Point> locations;

	for (int i = 0; i < FileNum; i++) {
		sprintf_s(FullFileName, "%s%d%s.JPG", FirstFileName, i + 1, ")");

		//read img and preprocess 
		Mat img, img_gray;
		img = imread(FullFileName);
		resize(img, img, Size(64, 64));
		cvtColor(img, img_gray, CV_RGB2GRAY);

		//extract feature
		HOGDescriptor d(Size(64, 64), Size(32, 32), Size(16, 16), Size(16, 16), 9);
		d.compute(img_gray, descriptorsValues, Size(0, 0), Size(0, 0), locations);
		v_descriptorsValues.push_back(descriptorsValues);
		v_locations.push_back(locations);
	}

	//save to xml
	FileStorage hogXml(SaveHogDesFileName, FileStorage::WRITE);	
	int row = v_descriptorsValues.size(), col = v_descriptorsValues[0].size();
	Mat M(row, col, CV_32F);

	//save Mat to XML
	for (int i = 0; i< row; ++i)
		memcpy(&(M.data[col * i * sizeof(float)]), v_descriptorsValues[i].data(), col * sizeof(float));

	//write xml
	write(hogXml, "Descriptor_of_images", M);
	hogXml.release();
}