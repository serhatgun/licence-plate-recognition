#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/parallel/parallel.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

class SVMClassifier {
public:
    SVMClassifier(const Size& cellSize, const Size& trainImSize);

    void train();
    void visualizeHOG();
    vector<int> predict(const vector<Mat>& characters);

private:
    Size cellSize_;
    Size trainImSize_;
    string trainDataPath_;
    int hogFeatureSize_;
    Ptr<SVM> classifier_;
    bool isTrained_;

    void loadTrainingData(const string& trainDataPath, vector<Mat>& trainingImages, vector<int>& trainingLabels);
    void extractHOGFeatures(const Mat& image, vector<float>& features);
};
