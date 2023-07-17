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
