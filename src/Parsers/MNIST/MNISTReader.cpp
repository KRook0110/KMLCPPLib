
#include "KMLCPPLib/Parsers/MNIST/MNISTReader.hpp"

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <stdlib.h>
#include <string>

namespace kmlcpplib {

namespace {
struct ImageReadHeader {
    int32_t magic, n, rows, cols;
};

struct LabelReadHeader {
    int32_t magic, n;
};

} // namespace

namespace fs = std::filesystem;
MNISTReader::ImageReadOuput MNISTReader::readImage(fs::path imagePath) {
    std::ifstream imageFile(imagePath, std::ios::binary | std::ios::ate);
    if (!imageFile.is_open()) {
        throw std::runtime_error("Failed to open " + imagePath.string());
    }

    std::streamsize size = imageFile.tellg();
    imageFile.seekg(0);

    std::vector<unsigned char> image_data;

    ImageReadHeader header;

    imageFile.read(reinterpret_cast<char *>(&header), sizeof(header));
    header.magic = __builtin_bswap32(header.magic);
    header.n = __builtin_bswap32(header.n);
    header.rows = __builtin_bswap32(header.rows);
    header.cols = __builtin_bswap32(header.cols);

    if (header.magic != imageMagicNumber) {
        throw std::runtime_error("Magic number mismatch, it should be " +
                                 std::to_string(imageMagicNumber) + ", got " +
                                 std::to_string(header.magic));
    }

    uint32_t data_size = header.n * header.rows * header.cols;
    image_data.resize(data_size);
    imageFile.read(reinterpret_cast<char *>(image_data.data()), data_size);

    std::vector<Eigen::MatrixXd> images(
        header.n, Eigen::MatrixXd(header.rows, header.cols));
    int counter = 0;
    for (auto &image : images) {
        for (int i = 0; i < header.rows; i++) {
            for (int j = 0; j < header.cols; j++) {
                image(i, j) = image_data[counter++];
            }
        }
    }

    imageFile.close();

    return {.magic = header.magic,
            .n = header.n,
            .rows = header.rows,
            .cols = header.cols,
            .data = std::move(images)};
}

MNISTReader::LabelReadOuput MNISTReader::readLabel(fs::path labelPath) {

    std::ifstream labelFile(labelPath, std::ios::binary | std::ios::ate);

    if (!labelFile.is_open()) {
        throw std::runtime_error("Failed to open " + labelPath.string());
    }

    std::vector<unsigned char> labels;
    LabelReadHeader header;

    std::streamsize size = labelFile.tellg();
    labelFile.seekg(0);

    labelFile.read(reinterpret_cast<char *>(&header), sizeof(header));

    header.magic = __builtin_bswap32(header.magic);
    header.n = __builtin_bswap32(header.n);

    if (header.magic != labelMagicNumber) {
        throw std::runtime_error("Magic number mismatch, it should be " +
                                 std::to_string(labelMagicNumber) + ", got " +
                                 std::to_string(header.magic));
    }

    labels.resize(header.n);

    labelFile.read(reinterpret_cast<char *>(labels.data()), header.n);

    labelFile.close();

    return {.magic = header.magic, .n = header.n, .data = std::move(labels)};
}

MNISTReader::ReadOutput MNISTReader::read(fs::path imagePath,
                                          fs::path labelPath) {
    return {.label = readLabel(labelPath), .image = readImage(imagePath)};
}

MNISTReader::MNISTDataset MNISTReader::loadData() {
    return {read(trainingImagesPath, trainingLabelsPath),
            read(testImagesPath, testLabelsPath)};
}

} // namespace kmlcpplib
