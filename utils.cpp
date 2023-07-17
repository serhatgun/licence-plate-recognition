#include "utils.h"

void Utils::my_histogram_equalization(const Mat& im, Mat& finalResult, vector<double>& histogram) {
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

Mat Utils::my_imgthresholding(const Mat& im, double T) {
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


void Utils::preprocessing(const Mat& im, Mat& resized_im, Mat& gray_im, Mat& eq_im, Mat& filtered_im, Mat& bin_im, bool show) {
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
