#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include <KMLCPPLib/Optimizers/GradientDescentOptimizer.hpp>
#include <KMLCPPLib/Parameter.hpp>

TEST(GradDescentOptimizerTest, CheckParamUpdates) {
    std::vector<std::shared_ptr<kmlcpplib::Parameter>> params;
    params.push_back(std::make_shared<kmlcpplib::Parameter>(3, 3));
    params.push_back(std::make_shared<kmlcpplib::Parameter>(1, 3));
    params.push_back(std::make_shared<kmlcpplib::Parameter>(3, 1));

    params[0]->value << 1, 1, 1,
        1, 1, 1,
        1, 1, 1;

    // Add values to the other parameters in the params variable
    params[1]->value << 2, 2, 2;
    params[2]->value << 3,
                        3,
                        3;

    // Set non-zero gradients for the parameter updates
    params[0]->gradient << 0.1, 0.2, 0.3,
                           0.4, 0.5, 0.6,
                           0.7, 0.8, 0.9;
    params[1]->gradient << 0.1, 0.2, 0.3;
    params[2]->gradient << 0.1,
                           0.2,
                           0.3;

    // Initialize optimizer
    kmlcpplib::GradientDescentOptimizer optimizer;
    optimizer.set_params(params);
    optimizer.set_learning_rate(0.1);

    // Run optimization step
    optimizer.step();

    // Verify parameter updates: value = value - learning_rate * gradient
    // params[0] expected values (3x3)
    EXPECT_NEAR(params[0]->value(0, 0), 0.99, 1e-6);
    EXPECT_NEAR(params[0]->value(0, 1), 0.98, 1e-6);
    EXPECT_NEAR(params[0]->value(0, 2), 0.97, 1e-6);
    EXPECT_NEAR(params[0]->value(1, 0), 0.96, 1e-6);
    EXPECT_NEAR(params[0]->value(1, 1), 0.95, 1e-6);
    EXPECT_NEAR(params[0]->value(1, 2), 0.94, 1e-6);
    EXPECT_NEAR(params[0]->value(2, 0), 0.93, 1e-6);
    EXPECT_NEAR(params[0]->value(2, 1), 0.92, 1e-6);
    EXPECT_NEAR(params[0]->value(2, 2), 0.91, 1e-6);

    // params[1] expected values (1x3)
    EXPECT_NEAR(params[1]->value(0, 0), 1.99, 1e-6);
    EXPECT_NEAR(params[1]->value(0, 1), 1.98, 1e-6);
    EXPECT_NEAR(params[1]->value(0, 2), 1.97, 1e-6);

    // params[2] expected values (3x1)
    EXPECT_NEAR(params[2]->value(0, 0), 2.99, 1e-6);
    EXPECT_NEAR(params[2]->value(1, 0), 2.98, 1e-6);
    EXPECT_NEAR(params[2]->value(2, 0), 2.97, 1e-6);
}
