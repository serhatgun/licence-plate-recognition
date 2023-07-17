#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/parallel/parallel.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class Utils {
private:
    vector<int> frequency;
    vector<double> pdf;
    vector<double> cdf;
    vector<double> cumulative;
    vector<int> out;

public:
    void my_histogram_equalization(const Mat& im, Mat& finalResult, vector<double>& histogram);
    Mat my_imgthresholding(const Mat& im, double T = -1);
    void preprocessing(const Mat& im, Mat& resized_im, Mat& gray_im, Mat& eq_im, Mat& filtered_im, Mat& bin_im, bool show);
};
