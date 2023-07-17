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
};
