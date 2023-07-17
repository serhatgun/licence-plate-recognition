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

    void adaptiveCalc(const Mat& im, int row, int col, const Mat& mask, int& A1, int& A2, int& zmin, int& zmax, int& zmed, int& zxy);

public:
    void histogramEqualization(const Mat& im, Mat& finalResult, vector<double>& histogram);
    Mat imgThresholding(const Mat& im, double T = -1);
    void preProcessing(const Mat& im, Mat& resized_im, Mat& gray_im, Mat& eq_im, Mat& filtered_im, Mat& bin_im, bool show);
    void extractPlateRegion(const Mat& eq_im, const Mat& bin_im, Mat& extracted_im, bool show);
    void extractCharacters(const Mat& extracted_plate, const Size& charSize, Mat& characters, bool show);
};
