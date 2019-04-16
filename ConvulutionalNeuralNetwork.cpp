//
// Created by talba on 2018-10-03.
//

#include "ConvulutionalNeuralNetwork.h"

ConvulutionalNeuralNetwork::ConvulutionalNeuralNetwork() {
    srand(time(NULL));
    filterCount = 4;
    learningRate = 0.7;
    epoch = 1000;

    imageCount = 1;     // user will only be passing in images one at a time for testing; there will be at most 1
}

ConvulutionalNeuralNetwork::ConvulutionalNeuralNetwork(vector<vector<double> > anImage, double expected) {
    srand(time(NULL));
    filterCount = 4;
    imageCount = 1;
    learningRate = 0.7;
    epoch = 1000;

    expectedValues.push_back(expected);

    images.push_back(anImage);

    initializeWeights();
}

ConvulutionalNeuralNetwork::ConvulutionalNeuralNetwork(vector<vector<vector<double> > > seriesImage, vector<double> seriesExpected) {
    srand(time(NULL));
    filterCount = 4;
    imageCount = seriesImage.size();
    learningRate = 0.7;
    epoch = 1000;

    expectedValues = seriesExpected;

    images = seriesImage;

    initializeWeights();
}

void ConvulutionalNeuralNetwork::initializeWeights() {
    //std::default_random_engine generator;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    //std::random_device rd{};
    std::mt19937 gen{seed};
    std::normal_distribution<double> distribution(0.0,2.0); // mean at 0, standard deviation is 2

    int n = images[0].size() - 1;
    initialWeights.resize(n);
    for (int i=0; i<n; i++) {
        initialWeights[i] = vector<double>(n);
        for (int j=0; j<n; j++) {
            //row[j] = randDouble(-6, 6);
            initialWeights[i][j] = distribution(gen);     // randomly sample values from above normal distribution
        }
    }

    int m = 4;
    finalWeights.resize(m);
    for (int i=0; i<m; i++) {
        finalWeights[i] = distribution(gen);    // randomly sample values from above normal distribution
    }
    //finalWeights = flatten2D(rotateVector(initialWeights));
}

void ConvulutionalNeuralNetwork::adaptWeights(vector<vector<double> > grad1, vector<double> grad2) {
    int initialY = initialWeights.size();
    int initialX = initialWeights[0].size();
    for (int i=0; i<initialY; i++) {    // iterate across the n x n vector of initial weights
        for (int j=0; j<initialX; j++) {
            double tmp = initialWeights[i][j];
            initialWeights[i][j] = tmp - grad1[i][j] * learningRate;
        }
    }

    int finalLength = finalWeights.size();
    for (int i=0; i<finalLength; i++) {     // iterate across the 1D vector of final weights
        double tmp = finalWeights[i];
        finalWeights[i] = tmp - grad2[i] * learningRate;
    }
}

void ConvulutionalNeuralNetwork::train(int cores) {
    vector<vector<double> > filter1 = read2DVector("identity");
    vector<vector<double> > filter2 = read2DVector("edge1");
    vector<vector<double> > filter3 = read2DVector("edge2");
    vector<vector<double> > filter4 = read2DVector("edge3");
    vector<vector<double> > filter5 = read2DVector("sharpen");
    vector<vector<double> > filter6 = read2DVector("guassBlur");

    for (int i = 0; i < epoch; i++) {
        for (int k=0; k<imageCount; k++) {
            // Applies convolution to the image with the filter
            vector<vector<double> > image = convolve2Dpad(images[k], filter1, cores);
            image = convolve2Dpad(image, filter5, cores);

            // Applies the function (sigmoid tanh) onto the neurons
            vector<vector<double> > layer1 = convolve2D(image, initialWeights, cores);
            vector<double> layer1Vec = flatten2D(layer1);
            vector<double> layer1ActVec = funcOnVector(tanh, layer1Vec);

            // Dot product the clamped Layer1 values with the finalWeights.
            // Then, applying the Logistic Sigmoid function gives the "output".
            double layer2 = dotProduct(layer1ActVec, finalWeights);
            double layer2Act = logarithm(layer2);

            // First, calculate the difference of the output from the expected value.
            // Apply this difference as an x onto the Logistic Sigmoid Function, and then dot product that with
            // the clamped values from Layer1 to get the derivative.
            double gradient2a = layer2Act - expectedValues[k];
            double gradient2b = dlog(gradient2a);
            vector<double> gradient2c = layer1ActVec;
            vector<double> gradient2 = dotProduct(gradient2c, gradient2a * gradient2b);


            vector<double> gradient1a = dotProduct(finalWeights, gradient2a * gradient2b);
            vector<vector<double> > gradient1b = funcOnVector2D(dtanh, layer1);
            vector<vector<double> > gradient1c = image;

            vector<vector<double> > gradient1Tmp = multiplyVectors(gradient1b, gradient1a);
            vector<vector<double> > gradient1 = rotateVector(
                    convolve2D(gradient1c, rotateVector(gradient1Tmp), cores));

            // We have the derivatives and have finished 1 step. Adjust the weights and then repeat.
            adaptWeights(gradient1, gradient2);
        }
    }
}

vector<double> ConvulutionalNeuralNetwork::outputLayer() {
    vector<double> output (imageCount);

    for (int i=0; i<imageCount; i++) {
        vector<vector<double> > layer1 = convolve2D(images[i], initialWeights, 4);
        vector<double> layer1Vec = flatten2D(layer1);
        vector<double> layer1ActVec = funcOnVector(tanh, layer1Vec);

        double layer2 = dotProduct(layer1ActVec, finalWeights);
        double layer2Act = logarithm(layer2);

        output[i] = layer2Act;
    }
    return output;
}

double ConvulutionalNeuralNetwork::testAnImage(vector<vector<double> > anImage) {
    vector<vector<double> > layer1 = convolve2D(anImage, initialWeights, 4);
    vector<double> layer1Vec = flatten2D(layer1);
    vector<double> layer1ActVec = funcOnVector(tanh, layer1Vec);

    double layer2 = dotProduct(layer1ActVec, finalWeights);
    double layer2Act = logarithm(layer2);

    return layer2Act;
}

void ConvulutionalNeuralNetwork::saveWeights(char *preFileName, char *postFileName) {
    write2DVector(initialWeights, preFileName);
    write1DVector(finalWeights, postFileName);
}

void ConvulutionalNeuralNetwork::loadWeights(char *preFileName, char *postFileName) {
    initialWeights = read2DVector(preFileName);
    finalWeights = read1DVector(postFileName);
}

void ConvulutionalNeuralNetwork::setLearningRate(double newRate) {
    learningRate = newRate;
}