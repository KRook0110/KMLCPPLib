
#pragma once

#include <Eigen/Dense>
namespace kmlcpplib {
namespace ramps {

const std::string full =
    "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ";
const std::string simple = "@%#*+=-:. ";

} // namespace ramps

void renderImage(Eigen::MatrixXd data, std::string ramp = ramps::full,
                 double data_min = 0, double data_max = 255);
void renderImages(std::vector<Eigen::MatrixXd> data,
                  std::string ramp = ramps::full, double data_min = 0,
                  double data_max = 255);
} // namespace kmlcpplib
