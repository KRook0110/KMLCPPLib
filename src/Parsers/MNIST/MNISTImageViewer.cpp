#include <KMLCPPLib/Parsers/MNIST/MNISTImageViewer.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>

namespace kmlcpplib {

namespace {
double ratio(double val, double low, double high) {
    return std::clamp((val - low) / (high - low), 0.0, 1.1);
}
} // namespace

void renderImage(Eigen::MatrixXd data, std::string ramp, double data_min,
                 double data_max) {
    for (int i = 0; i < data.rows(); i++) {
        for (int j = 0; j < data.cols(); j++) {
            double data_ratio = ratio(data(i, j), data_min, data_max);
            int rampIdx =
                static_cast<int>(std::lerp(0, ramp.size(), data_ratio));
            if (rampIdx == ramp.size())
                rampIdx--;
            std::cout << ramp[rampIdx];
        }
        std::cout << std::endl;
    }
}

void renderImages(std::vector<Eigen::MatrixXd> data, std::string ramp,
                  double data_min, double data_max) {
    for (int i = 0; i < data.size(); i++) {
        std::cout << "--- Image [ " + std::to_string(i + 1) + "] ---" << std::endl;
        renderImage(data[i], ramp, data_min, data_max);
    }
}
} // namespace kmlcpplib
