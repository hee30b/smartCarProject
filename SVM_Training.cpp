#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv\ml.h>

using namespace cv;
using namespace std;


#ifdef _DEBUG        
#pragma comment(lib, "opencv_core2413d.lib")         
#pragma comment(lib, "opencv_imgproc2413d.lib")     
#pragma comment(lib, "opencv_objdetect2413d.lib")
#pragma comment(lib, "opencv_highgui2413d.lib")        
#pragma comment(lib, "opencv_ml2413d.lib")      

#else        
#pragma comment(lib, "opencv_core2413.lib")        
#pragma comment(lib, "opencv_imgproc2413.lib")        
#pragma comment(lib, "opencv_objdetect2413.lib")        
#pragma comment(lib, "opencv_highgui2413.lib")        
#pragma comment(lib, "opencv_ml2413.lib")          
#endif 

class MySvm : public CvSVM {
public:
	int get_alpha_count() {
		return this->sv_total;
	}

	int get_sv_dim() {
		return this->var_all;
	}

	int get_sv_count() {
		return this->decision_func->sv_count;
	}

	double* get_alpha() {
		return this->decision_func->alpha;
	}

	float** get_sv() {
		return this->sv;
	}

	float get_rho() {
		return this->decision_func->rho;
	}
};


void main() {
	//create xml to read
	cout << "Input Files Load " << endl;
	FileStorage read_PositiveXml("POS.xml", FileStorage::READ);
	FileStorage read_NegativeXml("neg.xml", FileStorage::READ);
	char SVMSaveFile[100] = "SVM_Result.xml";

	//Positive Mat, Read Row, Cols
	Mat pMat;
	read_PositiveXml["Descriptor_of_images"] >> pMat;
	int pRow, pCol;
	pRow = pMat.rows; pCol = pMat.cols;
	read_PositiveXml.release();

	//Negative Mat, Read Row, Cols
	Mat nMat;
	read_NegativeXml["Descriptor_of_images"] >> nMat;
	int nRow, nCol;
	nRow = nMat.rows; nCol = nMat.cols;
	read_NegativeXml.release();

	//Make training data for SVM
	cout << "Make Training data for SVM  " << endl;
	Mat PN_Descriptor_mtx(pRow + nRow, pCol, CV_32FC1); // pCol and nCol must be the same
	memcpy(PN_Descriptor_mtx.data, pMat.data, sizeof(float)* pMat.cols * pMat.rows);
	int startP = sizeof(float)* pMat.cols * pMat.rows;
	memcpy(&(PN_Descriptor_mtx.data[startP]), nMat.data, sizeof(float)* nMat.cols * nMat.rows);

	//data labeling
	cout << "Data Labeling  " << endl;
	Mat labels(pRow + nRow, 1, CV_32FC1, Scalar(-1.0));
	labels.rowRange(0, pRow) = Scalar(1.0);

	//Set svm parameters
	cout << "Set SVM parameters " << endl;
	CvSVM svm;
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100000, 1e-6);

	//Training
	cout << "Training " << endl;
	svm.train_auto(PN_Descriptor_mtx, labels, Mat(), Mat(), params);
	params = svm.get_params();

	//Trained data save
	cout << "Save File " << endl;
	svm.save(SVMSaveFile);
}