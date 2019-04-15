//
// Created by talba on 2018-10-03.
//

#ifndef CNN_CNN_H
#define CNN_CNN_H

#include "utility.h"
#include "utilityIO.h"

#include <iostream>
#include <random>       // normal distribution random
#include <chrono>
#include <stdio.h>      // null, and I/O
#include <stdlib.h>     // for srand, rand
#include <time.h>
#include <cmath>
#include <vector>

using namespace std;

class ConvulutionalNeuralNetwork {
private:
    int filterCount;
    int imageCount;

    vector<vector<vector<double> > > filters;

    vector<vector<double> > initialWeights;      // weights between Input Layer and Hidden Layer
    vector<double> finalWeights;                // weights between Hidden Layer and Output Layer. Has same element count as the prior

    double learningRate;    // rate at which the weights will be adjusted during the learning process

    int epoch;              // essentially the amounts of runs

    vector<vector<vector<double> > > images;      // vector containing all the representation of images
    vector<double> expectedValues;              // what values to expect for each image. Follows a Sigmoid function, with 1.0 as very expected.
public:
    /*
     * Constructor for when a user is not specifying any images or expected values for training.
     * This should be used if no training is to be done weights are to be read in from prior training.
     */
    ConvulutionalNeuralNetwork() {
        srand(time(NULL));
        filterCount = 4;
        learningRate = 0.7;
        epoch = 1000;

        imageCount = 1;     // user will only be passing in images one at a time for testing; there will be at most 1
    }

    /*
     * Constructor for when a user is passing in only one image and expected value to train.
     * Expected values should follow a Sigmoid function, with nearing 1.0 as very expected.
     */
    ConvulutionalNeuralNetwork(vector<vector<double> > anImage, double expected) {
        srand(time(NULL));
        filterCount = 4;
        imageCount = 1;
        learningRate = 0.7;
        epoch = 1000;

        expectedValues.push_back(expected);

        images.push_back(anImage);

        initializeWeights();
    }

    /*
     * Constructor for when multiple images and their respective expected values are passed in for training.
     * Expected values should follow a Sigmoid function, with nearing 1.0 as very expected.
     */
    ConvulutionalNeuralNetwork(vector<vector<vector<double> > > seriesImage, vector<double> seriesExpected) {
        srand(time(NULL));
        filterCount = 4;
        imageCount = seriesImage.size();
        learningRate = 0.7;
        epoch = 1000;

        expectedValues = seriesExpected;

        images = seriesImage;

        initializeWeights();
    }

    ~ConvulutionalNeuralNetwork() {

    }

    /*
     * This method is for the beginning of training. It sets the weights before the Hidden Layer to a certain pattern
     * and the ones after the Hidden Layer to a random pattern.
     */
    void initializeWeights() {
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

    /*
     * Given the derivatives over initial and final weights from the learning process, adjust the weights
     * as dictated by the learning rate.
     * NewWeight = OldWeight - Derivative * LearningRate
     */
    void adaptWeights(vector<vector<double> > grad1, vector<double> grad2) {
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

    void train(int cores) {
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
                        convolve2D(gradient1c, rotateVector(gradient1Tmp), cores));f

                // We have the derivatives and have finished 1 step. Adjust the weights and then repeat.
                adaptWeights(gradient1, gradient2);
            }
        }
    }

    vector<double> outputLayer() {
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

    /*
     * Gives an expected floating point value (0,1) for a given 2D vector representing an image.
     */
    double testAnImage(vector<vector<double> > anImage) {
        vector<vector<double> > layer1 = convolve2D(anImage, initialWeights, 4);
        vector<double> layer1Vec = flatten2D(layer1);
        vector<double> layer1ActVec = funcOnVector(tanh, layer1Vec);

        double layer2 = dotProduct(layer1ActVec, finalWeights);
        double layer2Act = logarithm(layer2);

        return layer2Act;
    }

    /*
     * Saves the trained weights before and after the Hidden Layer to 2 specified file names (one for each).
     */
    void saveWeights(char *preFileName, char *postFileName) {
        write2DVector(initialWeights, preFileName);
        write1DVector(finalWeights, postFileName);
    }

    /*
     * Loads in the prior trained weights from 2 different files.
     */
    void loadWeights(char *preFileName, char *postFileName) {
        initialWeights = read2DVector(preFileName);
        finalWeights = read1DVector(postFileName);
    }

    /*
     * For when a user wants to adjust the learning rate.
     */
    void setLearningRate(double newRate) {
        learningRate = newRate;
    }
};


#endif //CNN_CNN_H
