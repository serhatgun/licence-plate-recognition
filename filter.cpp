#include "filter.h"

Filter::Filter() {}

Filter::~Filter() {}

void Filter::meanFilter(const Mat& im, const Mat& mask, const string& form, double degree, Mat& res) {
    int u = mask.rows;
    int v = mask.cols;
    int x = im.rows;
    int y = im.cols;

    Mat new_im(x, y, CV_8UC1, Scalar(0));

    Mat padded_im;
    copyMakeBorder(im, padded_im, (u - 1) / 2, (u - 1) / 2, (v - 1) / 2, (v - 1) / 2, BORDER_REPLICATE);

    x = padded_im.rows;
    y = padded_im.cols;

    parallel_for_(Range((u + 1) / 2, x - ((u - 1) / 2)), [&](const Range& range) {
        for (int row = range.start; row < range.end; ++row) {
            for (int col = (v + 1) / 2; col < y - ((v - 1) / 2); ++col) {
                double val = 0;
                double val2 = 0;

                parallel_for_(Range(-(u - u / 2), u - u / 2 + 1), [&](const Range& range_s) {
                    for (int s = range_s.start; s < range_s.end; ++s) {
                        for (int t = -(v - v / 2); t <= v - v / 2; ++t) {
                            double mul = padded_im.at<uchar>(row + s, col + t) * mask.at<double>(s + u / 2, t + v / 2);
                            val += pow(mul, degree + 1);
                            val2 += pow(mul, degree);
                        }
                    }
                });

                if (form == "geometric")
                    new_im.at<uchar>(row - (u - 1) / 2, col - (v - 1) / 2) = pow(val, 1 / (u * v));
                else if (form == "harmonic")
                    new_im.at<uchar>(row - (u - 1) / 2, col - (v - 1) / 2) = (u * v) / val;
                else if (form == "contra")
                    new_im.at<uchar>(row - (u - 1) / 2, col - (v - 1) / 2) = val / val2;
                else if (form == "alpha")
                    new_im.at<uchar>(row - (u - 1) / 2, col - (v - 1) / 2) = ((1 / (u * v)) - degree) * val;
                else
                    new_im.at<uchar>(row - (u - 1) / 2, col - (v - 1) / 2) = val;
            }
        }
    });

    res = new_im.clone();
}

void Filter::maxFilter(const Mat& im, const Mat& mask, Mat& res) {
    int u = mask.rows;
    int v = mask.cols;
    int x = im.rows;
    int y = im.cols;

    Mat new_im(x, y, CV_8UC1, Scalar(0));

    Mat padded_im;
    copyMakeBorder(im, padded_im, (u - 1) / 2, (u - 1) / 2, (v - 1) / 2, (v - 1) / 2, BORDER_REPLICATE);

    x = padded_im.rows;
    y = padded_im.cols;

    parallel_for_(Range((u + 1) / 2, x - ((u - 1) / 2)), [&](const Range& range) {
        for (int row = range.start; row < range.end; ++row) {
            for (int col = (v + 1) / 2; col < y - ((v - 1) / 2); ++col) {
                Mat maxValues = Mat::zeros(u, v, CV_8UC1);

                for (int s = -(u - u / 2); s <= u - u / 2; ++s) {
                    for (int t = -(v - v / 2); t <= v - v / 2; ++t) {
                        maxValues.at<uchar>(s + u / 2, t + v / 2) = padded_im.at<uchar>(row + s, col + t);
                    }
                }

                new_im.at<uchar>(row - (u - 1) / 2, col - (v - 1) / 2) = max(maxValues)[0];
            }
        }
    });

    res = new_im.clone();
}

void Filter::minFilter(const Mat& im, const Mat& mask, Mat& res) {
    int u = mask.rows;
    int v = mask.cols;
    int x = im.rows;
    int y = im.cols;

    Mat new_im(x, y, CV_8UC1, Scalar(0));

    Mat padded_im;
    copyMakeBorder(im, padded_im, (u - 1) / 2, (u - 1) / 2, (v - 1) / 2, (v - 1) / 2, BORDER_REPLICATE);

    x = padded_im.rows;
    y = padded_im.cols;

    parallel_for_(Range((u + 1) / 2, x - ((u - 1) / 2)), [&](const Range& range) {
        for (int row = range.start; row < range.end; ++row) {
            for (int col = (v + 1) / 2; col < y - ((v - 1) / 2); ++col) {
                Mat minValues = Mat::zeros(u, v, CV_8UC1);

                for (int s = -(u - u / 2); s <= u - u / 2; ++s) {
                    for (int t = -(v - v / 2); t <= v - v / 2; ++t) {
                        minValues.at<uchar>(s + u / 2, t + v / 2) = padded_im.at<uchar>(row + s, col + t);
                    }
                }

                new_im.at<uchar>(row - (u - 1) / 2, col - (v - 1) / 2) = min(minValues)[0];
            }
        }
    });

    res = new_im.clone();
}

void Filter::adaptiveCalc(const Mat& im, int row, int col, const Mat& mask, int& A1, int& A2, int& zmin, int& zmax, int& zmed, int& zxy) {
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

void Filter::medianFilter(const Mat& im, const Mat& mask, const string& form, Mat& res) {
    int u = mask.rows;
    int v = mask.cols;
    int x = im.rows;
    int y = im.cols;

    Mat new_im(x, y, CV_8UC1, Scalar(0));

    Mat padded_im;
    copyMakeBorder(im, padded_im, (u - 1) / 2, (u - 1) / 2, (v - 1) / 2, (v - 1) / 2, BORDER_REPLICATE);

    x = padded_im.rows;
    y = padded_im.cols;

    if (form == "adaptive") {
        int smax = 7;

        parallel_for_(Range((u + 1) / 2, x - ((u - 1) / 2)), [&](const Range& range) {
            for (int row = range.start; row < range.end; ++row) {
                for (int col = (v + 1) / 2; col < y - ((v - 1) / 2); ++col) {
                    int A1, A2, zmin, zmax, zmed, zxy;
                    Mat adaptiveMask(u, v, CV_64FC1);

                    adaptiveCalc(padded_im, row, col, adaptiveMask, A1, A2, zmin, zmax, zmed, zxy);

                    if (A1 > 0 && A2 < 0) {
                        int B1 = zxy - zmin;
                        int B2 = zxy - zmax;

                        if (B1 > 0 && B2 < 0)
                            new_im.at<uchar>(row - (u - 1) / 2, col - (v - 1) / 2) = static_cast<uchar>(zxy);
                        else
                            new_im.at<uchar>(row - (u - 1) / 2, col - (v - 1) / 2) = static_cast<uchar>(zmed);
                    } else {
                        mask.setTo(1.0);
                        adaptiveCalc(padded_im, row, col, adaptiveMask, A1, A2, zmin, zmax, zmed, zxy);
                        new_im.at<uchar>(row - (u - 1) / 2, col - (v - 1) / 2) = static_cast<uchar>(zmed);
                    }
                }
            }
        });
    } else {
        parallel_for_(Range((u + 1) / 2, x - ((u - 1) / 2)), [&](const Range& range) {
            for (int row = range.start; row < range.end; ++row) {
                for (int col = (v + 1) / 2; col < y - ((v - 1) / 2); ++col) {
                    Mat medianValues = Mat::zeros(u, v, CV_8UC1);

                    for (int s = -(u - u / 2); s <= u - u / 2; ++s) {
                        for (int t = -(v - v / 2); t <= v - v / 2; ++t) {
                            medianValues.at<uchar>(s + u / 2, t + v / 2) = padded_im.at<uchar>(row + s, col + t);
                        }
                    }

                    new_im.at<uchar>(row - (u - 1) / 2, col - (v - 1) / 2) = median(medianValues)[0];
                }
            }
        });
    }

    res = new_im.clone();
}
