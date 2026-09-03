
#include "KMLCPPLib/Layers/SequenceLayer.hpp"
#include "KMLCPPLib/Layers/SoftmaxLayer.hpp"
#include "KMLCPPLib/Optimizers/GradientDescentOptimizer.hpp"
#include "KMLCPPLib/Parsers/MNIST/MNISTReader.hpp"
#include "KMLCPPLib/WeightInitializers/RandomizedWeight.hpp"
#include <KMLCPPLib/Layers/LinearLayer.hpp>
#include <KMLCPPLib/Layers/SigmoidLayer.hpp>
#include <KMLCPPLib/LossFunctions/CategoricalCrossEntropy.hpp>

#include <iostream>
#include <memory>

int main() {
    using namespace kmlcpplib;

    fs::path datasetDir = "datasets/mnist";
    if (!fs::exists(datasetDir)) {
        datasetDir = "../datasets/mnist";
    }
    if (!fs::exists(datasetDir)) {
        datasetDir = "../../datasets/mnist";
    }

    fs::path trainImagesPath = datasetDir / "train-images.idx3-ubyte";
    fs::path trainLabelsPath = datasetDir / "train-labels.idx1-ubyte";
    fs::path testImagesPath = datasetDir / "t10k-images.idx3-ubyte";
    fs::path testLabelsPath = datasetDir / "t10k-labels.idx1-ubyte";

    try {
        kmlcpplib::MNISTReader reader(trainImagesPath, trainLabelsPath,
                                      testImagesPath, testLabelsPath);

        auto data = reader.loadData();

        int row = data.train.image.rows;
        int col = data.train.image.cols;
        int flatten = row * col;

        int firstLayer = 30;
        int secondLayer = 30;
        int outputLayer = 10;
        double learning_rate = 2e-2;

        CategoricalCrossEntropy loss(outputLayer);

        SequenceLayer model(
            {std::make_shared<LinearLayer>(flatten, firstLayer),
             std::make_shared<SigmoidLayer>(firstLayer),
             std::make_shared<LinearLayer>(firstLayer, secondLayer),
             std::make_shared<SigmoidLayer>(secondLayer),
             std::make_shared<LinearLayer>(secondLayer, outputLayer),
             std::make_shared<SoftmaxLayer>(outputLayer)});

        int epoch_max = 10;

        assert(data.train.image.n == data.train.label.n);
        GradientDescentOptimizer optimizer;
        optimizer.set_learning_rate(learning_rate);
        optimizer.set_params(model.get_params());

        randomize_weights(model.get_params());

        for(int epoch = 1;epoch <= epoch_max;epoch++) {
            double loss_avg = 0.0;
            for(int i = 0; i < data.train.image.n;i++) {
                auto in = data.train.image.data[i].reshaped();
                auto label = Eigen::VectorXd::Unit(outputLayer, data.train.label.data[i]);
                auto scaled_in = in.array() / 255.0;

                auto model_forward = model.forward(scaled_in);
                double loss_amount = loss.forward(model_forward, label);
                loss_avg += loss_amount / data.train.image.n;

                auto loss_backward = loss.backward();
                auto model_backward = model.backward(loss_backward);

                optimizer.step();

                if( i< 10) {
                    Eigen::Index pred_idx;
                    double max_val = model_forward.maxCoeff(&pred_idx);
                    std::cout << "label : " <<  (int)data.train.label.data[i] << ' ' << pred_idx << std::endl;
                }


            }

            std::cout << "[epoch " << epoch << "] : " << loss_avg << "\n";
        }

        int total_correct = 0;

        for(int i=0 ;i < data.test.image.n;i++) {
            auto in = data.test.image.data[i].reshaped().array() / 255.0;
            auto model_forward = model.forward(in);
            Eigen::Index pred_idx;
            model_forward.maxCoeff(&pred_idx);
            if(pred_idx == static_cast<int>(data.test.label.data[i])) {
                total_correct++;
            }
        }
        std::cout << "total data : " << data.test.image.n << std::endl;
        std::cout << "total correct : " << total_correct << std::endl;
        std::cout << "total wrong : " << data.test.image.n - total_correct << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
