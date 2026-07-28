#include <gtest/gtest.h>
#include <KMLCPPLib/Parameter.hpp>

TEST(ParameterTest, Initialization) {
    kmlcpplib::Parameter param(3, 4);
    EXPECT_EQ(param.rows(), 3);
    EXPECT_EQ(param.cols(), 4);
    EXPECT_EQ(param.value.rows(), 3);
    EXPECT_EQ(param.value.cols(), 4);
    EXPECT_EQ(param.gradient.rows(), 3);
    EXPECT_EQ(param.gradient.cols(), 4);
    
    EXPECT_TRUE(param.value.isZero());
    EXPECT_TRUE(param.gradient.isZero());
}

TEST(ParameterTest, ZeroGrad) {
    kmlcpplib::Parameter param(2, 2);
    param.gradient.setConstant(5.0);
    EXPECT_FALSE(param.gradient.isZero());
    
    param.zero_grad();
    EXPECT_TRUE(param.gradient.isZero());
}
