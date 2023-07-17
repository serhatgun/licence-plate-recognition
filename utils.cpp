#include "utils.h"

void Utils::adaptiveCalc(const Mat& im, int row, int col, const Mat& mask, int& A1, int& A2, int& zmin, int& zmax, int& zmed, int& zxy) {
    int u = mask.rows;
    int v = mask.cols;

    for (int s = -(u - u / 2); s <= u - u / 2; ++s) {
        for (int t = -(v - v / 2); t <= v - v / 2; ++t) {
            mask.at<double>(s + u / 2, t + v / 2) = static_cast<double>(im.at<uchar>(row + s, col + t));
        }
    }

    zmin = static_cast<int>(min(mask)[0]);
    zmax = static_cast<int>(max(mask)[0]);
    zmed = static_cast<int>(median(mask)[0]);
    zxy = static_cast<int>(im.at<uchar>(row, col));

    A1 = zmed - zmin;
    A2 = zmed - zmax;
}

void Utils::histogramEqualization(const Mat& im, Mat& finalResult, vector<double>& histogram) {
    int rows = im.rows;
    int cols = im.cols;
    int pixelNumber = rows * cols;

    finalResult = Mat(rows, cols, CV_8UC1);
    histogram = vector<double>(256, 0.0);

    frequency = vector<int>(256, 0);

    parallel_for_(Range(0, rows), [&](const Range& range) {
        for (int im_rows = range.start; im_rows < range.end; ++im_rows) {
            for (int im_cols = 0; im_cols < cols; ++im_cols) {
                int gray_level = im.at<uchar>(im_rows, im_cols);
                frequency[gray_level]++;
            }
        }
    });

    pdf = vector<double>(256);
    cdf = vector<double>(256);
    cumulative = vector<double>(256);
    out = vector<int>(256);

    reduce(frequency, pdf, 0, REDUCE_SUM);

    int sum = 0;
    int L = 255;

    for (int i = 0; i < pdf.size(); ++i) {
        sum += static_cast<int>(pdf[i]);
        cumulative[i] = static_cast<double>(sum);
        cdf[i] = cumulative[i] / pixelNumber;
        out[i] = static_cast<int>(round(cdf[i] * L));
    }

    parallel_for_(Range(0, rows), [&](const Range& range) {
        for (int im_rows = range.start; im_rows < range.end; ++im_rows) {
            for (int im_cols = 0; im_cols < cols; ++im_cols) {
                finalResult.at<uchar>(im_rows, im_cols) = static_cast<uchar>(out[im.at<uchar>(im_rows, im_cols)]);
            }
        }
    });

    parallel_for_(Range(0, rows), [&](const Range& range) {
        for (int im_rows = range.start; im_rows < range.end; ++im_rows) {
            for (int im_cols = 0; im_cols < cols; ++im_cols) {
                int gray_level = finalResult.at<uchar>(im_rows, im_cols);
                atomicAdd(&frequency[gray_level], 1);
            }
        }
    });

    reduce(frequency, histogram, 0, REDUCE_SUM);
    histogram /= pixelNumber;
}

Mat Utils::imgThresholding(const Mat& im, double T) {
    int x = im.rows;
    int y = im.cols;
    vector<uchar> G1;
    vector<uchar> G2;
    double meanGL = mean(im)[0];

    double th;
    if (T < 0)
        th = meanGL;
    else
        th = T;

    double newTh = 0;
    int it = 0;
    while (abs(th - newTh) > 0.1) {
        th = newTh;
        G1.clear();
        G2.clear();

        parallel_for_(Range(0, x), [&](const Range& range) {
            for (int row = range.start; row < range.end; ++row) {
                for (int col = 0; col < y; ++col) {
                    if (static_cast<double>(im.at<uchar>(row, col)) > th)
                        G1.push_back(im.at<uchar>(row, col));
                    else
                        G2.push_back(im.at<uchar>(row, col));
                }
            }
        });

        double u1 = mean(G1)[0];
        double u2 = mean(G2)[0];
        newTh = (u1 + u2) / 2;
        it++;
    }

    Mat new_im(x, y, CV_8UC1);

    parallel_for_(Range(0, x), [&](const Range& range) {
        for (int row = range.start; row < range.end; ++row) {
            for (int col = 0; col < y; ++col) {
                if (im.at<uchar>(row, col) <= static_cast<uchar>(round(newTh)))
                    new_im.at<uchar>(row, col) = 0;
                else
                    new_im.at<uchar>(row, col) = 255;
            }
        }
    });

    cout << "Number of iterations: " << it << endl;

    return new_im;
}


void Utils::preProcessing(const Mat& im, Mat& resized_im, Mat& gray_im, Mat& eq_im, Mat& filtered_im, Mat& bin_im, bool show) {
    // Resize image
    resize(im, resized_im, Size(1024, 768), 0, 0, INTER_CUBIC);

    // Convert to grayscale
    cvtColor(resized_im, gray_im, COLOR_BGR2GRAY);

    // Apply histogram equalization
    my_histogram_equalization(gray_im, eq_im);

    // Apply median filter
    Mat filtered_im_temp;
    medianBlur(eq_im, filtered_im_temp, 3);

    // Apply local Laplacian filter
    filtered_im = locallapfilt(filtered_im_temp, 0.2, 0.5);

    // Binarize the image
    double level = threshold(filtered_im, bin_im, 0, 255, THRESH_BINARY | THRESH_OTSU);

    if (show) {
        // Display the images
        vector<Mat> images = {resized_im, gray_im, eq_im, bin_im};
        vector<string> titles = {"Resized Image", "Gray Level", "Histogram Equalized", "Binary Image"};
        int numImages = images.size();

        for (int i = 0; i < numImages; ++i) {
            namedWindow(titles[i], WINDOW_NORMAL);
            imshow(titles[i], images[i]);
        }

        waitKey(0);
    }
}


void Utils::extractPlateRegion(const Mat& eq_im, const Mat& bin_im, Mat& extracted_im, bool show) {
    Mat edges, filled;
    Canny(bin_im, edges, 100, 200);
    morphologyEx(edges, filled, MORPH_CLOSE, Mat(), Point(-1, -1), 3);

    vector<vector<Point>> contours;
    findContours(filled, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    vector<Rect> boundRect(contours.size());

    for (size_t i = 0; i < contours.size(); i++) {
        if (contourArea(contours[i]) < 5000 && contourArea(contours[i]) > 250) {
            boundRect[i] = boundingRect(contours[i]);
            rectangle(extracted_im, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2);
        }
    }

    if (show) {
        namedWindow("Bounding Boxes", WINDOW_NORMAL);
        imshow("Bounding Boxes", extracted_im);
        waitKey(0);
    }
}

void Utils::extractCharacters(const Mat& extracted_plate, const Size& charSize, Mat& characters, bool show) {
    Mat gray_plate;
    cvtColor(extracted_plate, gray_plate, COLOR_BGR2GRAY);
    threshold(gray_plate, gray_plate, 0, 255, THRESH_BINARY | THRESH_OTSU);

    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(gray_plate, gray_plate, element);

    vector<vector<Point>> contours;
    findContours(gray_plate, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    characters = Mat::zeros(charSize, CV_8UC1);

    for (size_t i = 0; i < contours.size(); i++) {
        Rect boundRect = boundingRect(contours[i]);
        Mat charImage = gray_plate(boundRect);

        resize(charImage, charImage, charSize);
        characters.col(i) = charImage.reshape(1, 1);

        if (show) {
            imshow("Extracted Characters", charImage);
            waitKey(0);
        }
    }
}
