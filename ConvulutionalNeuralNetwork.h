//
// Created by talba on 2018-10-03.
//

#ifndef CNN_CNN_H
#define CNN_CNN_H

#include "utility.h"
#include "utilityIO.h"

#include <iostream>
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

        initializeFilter();
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

        initializeFilter();
    }

    ~ConvulutionalNeuralNetwork() {

    }

    /*
     * This method is for the beginning of training. It sets the weights before the Hidden Layer to a certain pattern
     * and the ones after the Hidden Layer to a random pattern.
     */
    void initializeFilter() {
        vector<vector<double> > aFilter(3);
        for (int i=0; i<3; i++) {
            vector<double> row(3);
            for (int j=0; j<3; j++) {
                row[j] = randDouble(-6, 6);
            }
            aFilter[i] = row;
        }
        initialWeights = aFilter;

        vector<double> secondFilter(3*3);
        for (int i=0; i<3*3; i++) {
            secondFilter[i] = randDouble(-6, 6);
        }
        finalWeights = secondFilter;
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
        vector<vector<double> > newInitial(initialWeights.size());
        for (int i=0; i<initialY; i++) {    // iterate across the n x n vector of initial weights
            vector<double> row(initialX);
            for (int j=0; j<initialX; j++) {
                row[j] = initialWeights[i][j] - grad1[i][j] * learningRate;
            }
            newInitial[i] = row;
        }
        initialWeights = newInitial;

        int finalLength = finalWeights.size();
        vector<double> newFinal(finalLength);
        for (int i=0; i<finalLength; i++) {     // iterate across the 1D vector of final weights
            newFinal[i] = finalWeights[i] - grad2[i] * learningRate;
        }
        finalWeights = newFinal;
    }

    void train() {
        for (int k=0; k<imageCount; k++) {
            for (int i = 0; i < epoch; i++) {
                // Applies convolution to the image with the filter, then clamps the values between (-1,1) with tanh
                vector<vector<double> > layer1 = convolve2D(images[k], initialWeights);
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
                vector<vector<double> > gradient1c = images[k];

                vector<vector<double> > gradient1Tmp = multiplyVectors(gradient1b, gradient1a);
                vector<vector<double> > gradient1 = rotateVector(
                        convolve2D(gradient1c, rotateVector(gradient1Tmp)));

                // We have the derivatives and have finished 1 step. Adjust the weights and then repeat.
                adaptWeights(gradient1, gradient2);
            }
        }
    }

    vector<double> outputLayer() {
        vector<double> output (imageCount);

        for (int i=0; i<imageCount; i++) {
            vector<vector<double> > layer1 = convolve2D(images[i], initialWeights);
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
        vector<vector<double> > layer1 = convolve2D(anImage, initialWeights);
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
