#pragma once

#include <filesystem>
#include <tuple>
#include <vector>

namespace kmlcpplib {

namespace fs = std::filesystem;

class MNISTReader {

    fs::path trainingImagesPath;
    fs::path trainingLabelsPath;
    fs::path testImagesPath;
    fs::path testLabelsPath;

  public:
    struct ReadOutput {
    };

    MNISTReader(fs::path trainingImagesPath, fs::path trainingLabelsPath,
                fs::path testImagesPath, fs::path testLabelsPath)
        : trainingImagesPath(std::move(trainingImagesPath)),
          trainingLabelsPath(std::move(trainingLabelsPath)),
          testImagesPath(std::move(testImagesPath)),
          testLabelsPath(std::move(testLabelsPath)) {}

    ReadOutput read( fs::path imagePath, fs::path labelPath);

    std::tuple<ReadOutput, ReadOutput> loadData();
};

} // namespace kmlcpplib
