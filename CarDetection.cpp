#include <iostream>
#include "opencv/cv.hpp"
#include <opencv/ml.h>
#include<ctime>
//gggg
using namespace cv;
using namespace std;

int detection(Mat gray, Rect roi_rect, Ptr<ml::SVM> svm, HOGDescriptor d, Mat frame, int frameNum) {
   float result;
   Mat currFrame = frame.clone(); 
   vector< float> descriptorsValues;
   Mat roi = gray(roi_rect);
   imshow("currentROI", roi);
   resize(roi, roi, Size(64, 64));

   d.compute(roi, descriptorsValues);
   Mat fm = Mat(descriptorsValues);
   fm = fm.reshape(0, 1);
   fm.convertTo(fm, CV_32F);
   result = svm->predict(fm);

   if(result == 1 ) {
      Mat curr =  frame(roi_rect);
      rectangle(currFrame, roi_rect, Scalar(155, 0, 155), 3, 8, 0);
      char currentDetected[100];  
   }
   imshow("detected", currFrame); 
   return result;
}

void warning(cv::Mat& frame, int type) { 
    int roiX = 230; 
    int roiY = 1; 
    int height = 235; 
    int distance = 0; 
 
    if (type == 0) 
        distance = 18; 
    else if (type == 1) 
        distance = 12; 
    else if (type == 2) 
        distance = 6; 
 
    cv::circle(frame, cv::Point(roiX + height, roiY + 100), 30, CV_RGB(0, 250, 0), -1, CV_AA); 
    cv::circle(frame, cv::Point(roiX + height, roiY + 100), 30, CV_RGB(0, 0, 255), 30, CV_AA); 
}

int main(void) {
	/**********************************************************************************************/
   Ptr<ml::SVM> svm = ml::SVM::create();
   svm = Algorithm::load<ml::SVM>("SVM_FINAL.xml");
   HOGDescriptor d(Size(64, 64), Size(32, 32), Size(16, 16), Size(16, 16), 9);
   VideoCapture capture("test2.avi");
   Mat frame, gray;
	
   capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
   capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	
   cv::VideoWriter outputVideo;
   cv::Size size = cv::Size((int)capture.get(cv::CAP_PROP_FRAME_WIDTH),	(int)capture.get(cv::CAP_PROP_FRAME_HEIGHT));
   int fps = capture.get(cv::CAP_PROP_FPS);
	
   float result, result2, result3;

   int roiX = 230; 
   int roiY = 1;

   int key, frame_rate = 30, frameNum = 0;

   char textfps[255];

   Rect currentRect(0, roiY, 80, 80);
   int currentResult = 0;
   int roiSize = 80;
	/**********************************************************************************************/
   while (capture.read(frame)) {
      outputVideo << frame;
      cvtColor(frame, gray, CV_RGB2GRAY);
      for (int i = 3; i >= 0; i--) {
         if (i == 0) { // 18m  
            roiSize = 48; 
            roiY = 185; 
         } 
         else if (i == 1) { // 12m  
            roiSize = 55; 
            roiY = 180; 
         } 
         else if (i == 2){ // 6m  
            roiSize = 100; 
            roiY = 160; 
         } 
         else if (i == 3) {  
            roiSize = 200; 
            roiY = 120;                  
         }  
         for (int current_X = 0; current_X < (frame.cols - roiSize); current_X += 10) {
            Rect currentROI(current_X, roiY, roiSize, roiSize);
            currentResult = detection(gray, currentROI, svm, d, frame, frameNum);
            if (currentResult == 1) {
               warning(frame, i);
               break;
            }
         }
      
         if (currentResult == 1) {
            break;
         }
      }
   
   	// key
      key = waitKey(frame_rate);
      if (key == 32) {
         if (frame_rate == 30)
            frame_rate = 0;
         else
            frame_rate = 30;
      }
      else if (key == ']') {
         capture.set(CV_CAP_PROP_POS_FRAMES, frameNum + 90);
         frameNum += 90;
      }
      else if (key == '[') {
         capture.set(CV_CAP_PROP_POS_FRAMES, frameNum - 90);
         frameNum -= 90;
      }
      else if (key == 27) {
         break;
      }
      frameNum++;
   }

}
