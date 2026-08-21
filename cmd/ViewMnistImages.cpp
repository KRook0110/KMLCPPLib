#include <KMLCPPLib/Parsers/MNIST/MNISTImageViewer.hpp>
#include <KMLCPPLib/Parsers/MNIST/MNISTReader.hpp>
#include <filesystem>
#include <iostream>
#include <algorithm>

namespace fs = std::filesystem;

int main() {
    fs::path datasetDir = "datasets/mnist";
    if (!fs::exists(datasetDir)) {
        datasetDir = "../datasets/mnist";
    }
    if (!fs::exists(datasetDir)) {
        datasetDir = "../../datasets/mnist";
    }

    fs::path trainImagesPath = datasetDir / "train-images.idx3-ubyte";
    fs::path trainLabelsPath = datasetDir / "train-labels.idx1-ubyte";
    fs::path testImagesPath  = datasetDir / "t10k-images.idx3-ubyte";
    fs::path testLabelsPath  = datasetDir / "t10k-labels.idx1-ubyte";

    try {
        kmlcpplib::MNISTReader reader(trainImagesPath, trainLabelsPath, testImagesPath, testLabelsPath);

        std::cout << "Loading test dataset from " << datasetDir << "..." << std::endl;
        auto testData = reader.read(testImagesPath, testLabelsPath);

        std::cout << "Loaded " << testData.image.n << " test images ("
                  << testData.image.rows << "x" << testData.image.cols << ") and "
                  << testData.label.n << " labels.\n" << std::endl;

        // Render first 5 images using ASCII viewer
        int numToRender = std::min<int>(5, static_cast<int>(testData.image.data.size()));
        for (int i = 0; i < numToRender; ++i) {
            std::cout << "=== Image #" << (i + 1)
                      << " | Label: " << static_cast<int>(testData.label.data[i])
                      << " ===" << std::endl;
            kmlcpplib::renderImage(testData.image.data[i], kmlcpplib::ramps::simple);
            std::cout << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
