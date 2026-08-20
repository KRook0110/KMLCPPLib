#include <gtest/gtest.h>
#include "KMLCPPLib/Parsers/MNIST/MNISTReader.hpp"
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

class MNISTReaderTest : public ::testing::Test {
protected:
    fs::path tempDir;

    void SetUp() override {
        tempDir = fs::temp_directory_path() / "mnist_reader_test_dir";
        fs::create_directories(tempDir);
    }

    void TearDown() override {
        fs::remove_all(tempDir);
    }

    void createDummyLabelFile(const fs::path& filePath, int32_t magicNumber, int32_t itemCount) {
        std::ofstream file(filePath, std::ios::binary);
        magicNumber = __builtin_bswap32(magicNumber);
        itemCount = __builtin_bswap32(itemCount);
        file.write(reinterpret_cast<const char*>(&magicNumber), sizeof(magicNumber));
        file.write(reinterpret_cast<const char*>(&itemCount), sizeof(itemCount));
    }
    void createDummyImageFile(const fs::path& filePath, int32_t magicNumber, int32_t itemCount, int32_t rows = 28, int32_t cols = 28) {
        std::ofstream file(filePath, std::ios::binary);
        magicNumber = __builtin_bswap32(magicNumber);
        itemCount = __builtin_bswap32(itemCount);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);
        file.write(reinterpret_cast<const char*>(&magicNumber), sizeof(magicNumber));
        file.write(reinterpret_cast<const char*>(&itemCount), sizeof(itemCount));
        file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    }
};

TEST_F(MNISTReaderTest, ThrowsOnFileNotFound) {
    kmlcpplib::MNISTReader reader("", "", "", "");
    EXPECT_THROW(reader.read("non_existent_img.idx3", "non_existent_lbl.idx1"), std::runtime_error);
}

TEST_F(MNISTReaderTest, ThrowsOnInvalidMagicNumber) {
    fs::path dummyLabel = tempDir / "bad_magic.idx1-ubyte";
    fs::path dummyImage = tempDir / "dummy_img.idx3-ubyte";

    createDummyLabelFile(dummyLabel, 1234, 10);
    createDummyImageFile(dummyImage, kmlcpplib::MNISTReader::imageMagicNumber, 10);

    kmlcpplib::MNISTReader reader("", "", "", "");
    EXPECT_THROW(reader.read(dummyImage, dummyLabel), std::runtime_error);
}

TEST_F(MNISTReaderTest, ValidMagicNumberDoesNotThrow) {
    fs::path dummyLabel = tempDir / "valid_magic.idx1-ubyte";
    fs::path dummyImage = tempDir / "dummy_img.idx3-ubyte";

    createDummyLabelFile(dummyLabel, kmlcpplib::MNISTReader::labelMagicNumber, 10);
    createDummyImageFile(dummyImage, kmlcpplib::MNISTReader::imageMagicNumber, 10);

    kmlcpplib::MNISTReader reader("", "", "", "");
    EXPECT_NO_THROW(reader.read(dummyImage, dummyLabel));
}

TEST_F(MNISTReaderTest, TestingActualDataset) {
    fs::path datasetDir = "datasets/mnist";
    if (!fs::exists(datasetDir)) {
        datasetDir = "../../datasets/mnist";
    }
    if (!fs::exists(datasetDir)) {
        datasetDir = "../datasets/mnist";
    }

    ASSERT_TRUE(fs::exists(datasetDir)) << "datasets/mnist directory not found";

    fs::path trainImages = datasetDir / "train-images.idx3-ubyte";
    fs::path trainLabels = datasetDir / "train-labels.idx1-ubyte";
    fs::path testImages = datasetDir / "t10k-images.idx3-ubyte";
    fs::path testLabels = datasetDir / "t10k-labels.idx1-ubyte";

    ASSERT_TRUE(fs::exists(trainImages)) << trainImages << " missing";
    ASSERT_TRUE(fs::exists(trainLabels)) << trainLabels << " missing";
    ASSERT_TRUE(fs::exists(testImages)) << testImages << " missing";
    ASSERT_TRUE(fs::exists(testLabels)) << testLabels << " missing";

    kmlcpplib::MNISTReader reader(trainImages, trainLabels, testImages, testLabels);

    // Test label reading
    auto trainLabelOutput = reader.readLabel(trainLabels);
    EXPECT_EQ(trainLabelOutput.magic, kmlcpplib::MNISTReader::labelMagicNumber);
    EXPECT_EQ(trainLabelOutput.n, 60000);
    EXPECT_EQ(trainLabelOutput.data.size(), 60000);

    auto testLabelOutput = reader.readLabel(testLabels);
    EXPECT_EQ(testLabelOutput.magic, kmlcpplib::MNISTReader::labelMagicNumber);
    EXPECT_EQ(testLabelOutput.n, 10000);
    EXPECT_EQ(testLabelOutput.data.size(), 10000);

    // Test image reading
    auto trainImageOutput = reader.readImage(trainImages);
    EXPECT_EQ(trainImageOutput.magic, kmlcpplib::MNISTReader::imageMagicNumber);
    EXPECT_EQ(trainImageOutput.n, 60000);
    EXPECT_EQ(trainImageOutput.rows, 28);
    EXPECT_EQ(trainImageOutput.cols, 28);
    EXPECT_EQ(trainImageOutput.data.size(), 60000);
    if (!trainImageOutput.data.empty()) {
        EXPECT_EQ(trainImageOutput.data[0].rows(), 28);
        EXPECT_EQ(trainImageOutput.data[0].cols(), 28);
    }

    auto testImageOutput = reader.readImage(testImages);
    EXPECT_EQ(testImageOutput.magic, kmlcpplib::MNISTReader::imageMagicNumber);
    EXPECT_EQ(testImageOutput.n, 10000);
    EXPECT_EQ(testImageOutput.rows, 28);
    EXPECT_EQ(testImageOutput.cols, 28);
    EXPECT_EQ(testImageOutput.data.size(), 10000);

    // Test full dataset loading
    auto dataset = reader.loadData();
    EXPECT_EQ(dataset.train.label.n, 60000);
    EXPECT_EQ(dataset.train.image.n, 60000);
    EXPECT_EQ(dataset.test.label.n, 10000);
    EXPECT_EQ(dataset.test.image.n, 10000);
}

