#pragma once

#include <Eigen/Core>
#include <filesystem>
#include <vector>

namespace kmlcpplib {

namespace fs = std::filesystem;

class MNISTReader {

    fs::path trainingImagesPath;
    fs::path trainingLabelsPath;
    fs::path testImagesPath;
    fs::path testLabelsPath;

  public:
    static constexpr int32_t labelMagicNumber = 2049;
    static constexpr int32_t imageMagicNumber = 2051;

    struct ImageReadOuput {
        int32_t magic, n, rows, cols;
        std::vector<Eigen::MatrixXd> data;
    };
    struct LabelReadOuput {
        int32_t magic, n;
        std::vector<unsigned char> data;

    };
    struct ReadOutput {
        LabelReadOuput label;

        ImageReadOuput image;
    };

    struct MNISTDataset {
        ReadOutput train;
        ReadOutput test;
    };

    MNISTReader(fs::path trainingImagesPath, fs::path trainingLabelsPath,
                fs::path testImagesPath, fs::path testLabelsPath)
        : trainingImagesPath(std::move(trainingImagesPath)),
          trainingLabelsPath(std::move(trainingLabelsPath)),
          testImagesPath(std::move(testImagesPath)),
          testLabelsPath(std::move(testLabelsPath)) {}

    ImageReadOuput readImage(fs::path imagePath);
    LabelReadOuput readLabel(fs::path labelPath);
    ReadOutput read( fs::path imagePath, fs::path labelPath);

    MNISTDataset loadData();
};

} // namespace kmlcpplib
