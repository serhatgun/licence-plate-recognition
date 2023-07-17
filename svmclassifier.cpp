include "svmclassifier.cpp"

SVMClassifier::SVMClassifier(const Size& cellSize, const Size& trainImSize)
    : cellSize_(cellSize), trainImSize_(trainImSize), isTrained_(false) {
    trainDataPath_ = "classifier" + to_string(trainImSize_.height) + ".xml";
}

void SVMClassifier::loadTrainingData(const string& trainDataPath, vector<Mat>& trainingImages, vector<int>& trainingLabels) {
    FileStorage fs(trainDataPath, FileStorage::READ);
    Mat labelsMat;
    fs["TrainingLabels"] >> labelsMat;
    trainingLabels = labelsMat.reshape(1, 1);

    int numImages = fs["NumImages"];
    trainingImages.resize(numImages);
    for (int i = 0; i < numImages; ++i) {
        string imageName = "TrainingImage" + to_string(i);
        fs[imageName] >> trainingImages[i];
    }

    fs.release();
}

void SVMClassifier::extractHOGFeatures(const Mat& image, vector<float>& features) {
    HOGDescriptor hog;
    hog.winSize = trainImSize_;
    hog.cellSize = cellSize_;

    vector<Point> locations;
    hog.compute(image, features, cellSize_, Size(), locations);
}

void SVMClassifier::train() {
    if (isTrained_) {
        cout << "SVM classifier is already trained." << endl;
        return;
    }

    vector<Mat> trainingImages;
    vector<int> trainingLabels;
    loadTrainingData(trainDataPath_, trainingImages, trainingLabels);

    int numImages = trainingImages.size();
    hogFeatureSize_ = 0;

    parallel_for_(Range(0, numImages), [&](const Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            Mat img = trainingImages[i];
            resize(img, img, trainImSize_, 0, 0, INTER_CUBIC);

            if (img.channels() == 3)
                cvtColor(img, img, COLOR_BGR2GRAY);

            vector<float> features;
            extractHOGFeatures(img, features);

            if (hogFeatureSize_ == 0)
                hogFeatureSize_ = features.size();

            Mat(features).copyTo(trainingImages[i]);
        }
    });

    Mat trainingData(numImages, hogFeatureSize_, CV_32FC1);
    for (int i = 0; i < numImages; ++i) {
        Mat row(trainingImages[i]);
        row.copyTo(trainingData.row(i));
    }

    Mat labels(numImages, 1, CV_32SC1, trainingLabels.data());

    classifier_ = SVM::create();
    classifier_->setType(SVM::C_SVC);
    classifier_->setKernel(SVM::LINEAR);
    classifier_->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    classifier_->train(trainingData, ROW_SAMPLE, labels);

    classifier_->save(trainDataPath_);

    isTrained_ = true;
}

void SVMClassifier::visualizeHOG() {
    if (!isTrained_) {
        cout << "SVM classifier is not trained." << endl;
        return;
    }

    vector<Mat> trainingImages;
    vector<int> trainingLabels;
    loadTrainingData(trainDataPath_, trainingImages, trainingLabels);

    Mat img = trainingImages[206];
    resize(img, img, trainImSize_, 0, 0, INTER_CUBIC);

    if (img.channels() == 3)
        cvtColor(img, img, COLOR_BGR2GRAY);

    vector<float> hog_2x2, hog_4x4, hog_8x8;
    vector<Point> locations;

    HOGDescriptor hog_2x2, hog_4x4, hog_8x8;
    hog_2x2.winSize = trainImSize_;
    hog_2x2.cellSize = cellSize_;
    hog_4x4.winSize = trainImSize_;
    hog_4x4.cellSize = Size(4, 4);
    hog_8x8.winSize = trainImSize_;
    hog_8x8.cellSize = Size(8, 8);

    hog_2x2.compute(img, hog_2x2, Size(), Size(), locations);
    hog_4x4.compute(img, hog_4x4, Size(), Size(), locations);
    hog_8x8.compute(img, hog_8x8, Size(), Size(), locations);

    imshow("Original Image", img);

    imshow("CellSize = [2 2]", hog_2x2);
    imshow("CellSize = [4 4]", hog_4x4);
    imshow("CellSize = [8 8]", hog_8x8);

    waitKey(0);
}

vector<int> SVMClassifier::predict(const vector<Mat>& characters) {
    if (!isTrained_) {
        cout << "SVM classifier is not trained." << endl;
        return vector<int>();
    }

    vector<int> predictedLabels;
    for (const Mat& character : characters) {
        Mat img = character.clone();
        resize(img, img, trainImSize_, 0, 0, INTER_CUBIC);

        if (img.channels() == 3)
            cvtColor(img, img, COLOR_BGR2GRAY);

        vector<float> features;
        extractHOGFeatures(img, features);

        Mat testFeatures(1, hogFeatureSize_, CV_32FC1, features.data());
        int predictedLabel = static_cast<int>(classifier_->predict(testFeatures));
        predictedLabels.push_back(predictedLabel);
    }

    return predictedLabels;
}

vector<Mat> readImagesFromFolder(const string& folderPath) {
    vector<Mat> images;
    vector<String> fileNames;
    glob(folderPath, fileNames);

    for (const auto& fileName : fileNames) {
        Mat image = imread(fileName);
        if (!image.empty())
            images.push_back(image);
    }

    return images;
}
