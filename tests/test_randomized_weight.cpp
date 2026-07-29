#include <gtest/gtest.h>
#include <KMLCPPLib/Parameter.hpp>
#include <KMLCPPLib/WeightInitializors/RandomizedWeight.hpp>
#include <memory>
#include <vector>

TEST(RandomizedWeightTest, RandomizeSingleParameterReference) {
    kmlcpplib::Parameter param(3, 4);
    EXPECT_EQ(param.rows(), 3);
    EXPECT_EQ(param.cols(), 4);
    EXPECT_TRUE(param.value.isZero());

    kmlcpplib::randomize_weights(param);

    EXPECT_EQ(param.rows(), 3);
    EXPECT_EQ(param.cols(), 4);
    EXPECT_FALSE(param.value.isZero());

    // Eigen setRandom() generates values in [-1, 1]
    for (int r = 0; r < param.rows(); ++r) {
        for (int c = 0; c < param.cols(); ++c) {
            EXPECT_GE(param.value(r, c), -1.0);
            EXPECT_LE(param.value(r, c), 1.0);
        }
    }
}

TEST(RandomizedWeightTest, RandomizeSharedPtrParameter) {
    auto param = std::make_shared<kmlcpplib::Parameter>(4, 4);
    EXPECT_TRUE(param->value.isZero());

    kmlcpplib::randomize_weights(param);

    EXPECT_EQ(param->rows(), 4);
    EXPECT_EQ(param->cols(), 4);
    EXPECT_FALSE(param->value.isZero());
}

TEST(RandomizedWeightTest, RandomizeVectorOfSharedPtrs) {
    std::vector<std::shared_ptr<kmlcpplib::Parameter>> weights = {
        std::make_shared<kmlcpplib::Parameter>(2, 3),
        std::make_shared<kmlcpplib::Parameter>(4, 1),
        std::make_shared<kmlcpplib::Parameter>(3, 5)
    };

    for (const auto& w : weights) {
        EXPECT_TRUE(w->value.isZero());
    }

    kmlcpplib::randomize_weights(weights);

    for (const auto& w : weights) {
        EXPECT_FALSE(w->value.isZero());
    }

    EXPECT_EQ(weights[0]->rows(), 2);
    EXPECT_EQ(weights[0]->cols(), 3);
    EXPECT_EQ(weights[1]->rows(), 4);
    EXPECT_EQ(weights[1]->cols(), 1);
    EXPECT_EQ(weights[2]->rows(), 3);
    EXPECT_EQ(weights[2]->cols(), 5);
}

TEST(RandomizedWeightTest, RandomizeEmptyVector) {
    std::vector<std::shared_ptr<kmlcpplib::Parameter>> empty_weights;
    EXPECT_NO_THROW(kmlcpplib::randomize_weights(empty_weights));
}
