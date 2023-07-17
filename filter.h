#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/parallel/parallel.hpp>

using namespace std;
using namespace cv;

class Filter {
public:
    Mat meanFilter(const Mat& im, const Mat& mask, const string& form, double degree, Mat& res);
    Mat maxFilter(const Mat& im, const Mat& mask);
    Mat minFilter(const Mat& im, const Mat& mask);
    void adaptiveCalc(const Mat& im, int row, int col, const Mat& mask, int& A1, int& A2, int& zmin, int& zmax, int& zmed, int& zxy);
    Mat medianFilter(const Mat& im, const Mat& mask, const string& form);

private: 
};
