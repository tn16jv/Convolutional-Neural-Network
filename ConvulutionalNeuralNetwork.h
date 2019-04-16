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
    ConvulutionalNeuralNetwork();

    /*
     * Constructor for when a user is passing in only one image and expected value to train.
     * Expected values should follow a Sigmoid function, with nearing 1.0 as very expected.
     */
    ConvulutionalNeuralNetwork(vector<vector<double> > anImage, double expected);

    /*
     * Constructor for when multiple images and their respective expected values are passed in for training.
     * Expected values should follow a Sigmoid function, with nearing 1.0 as very expected.
     */
    ConvulutionalNeuralNetwork(vector<vector<vector<double> > > seriesImage, vector<double> seriesExpected);

    ~ConvulutionalNeuralNetwork() {

    }

    /*
     * This method is for the beginning of training. It sets the weights before the Hidden Layer to a certain pattern
     * and the ones after the Hidden Layer to a random pattern.
     */
    void initializeWeights();

    /*
     * Given the derivatives over initial and final weights from the learning process, adjust the weights
     * as dictated by the learning rate.
     * NewWeight = OldWeight - Derivative * LearningRate
     */
    void adaptWeights(vector<vector<double> > grad1, vector<double> grad2);

    void train(int cores);

    virtual vector<double> outputLayer();

    /*
     * Gives an expected floating point value (0,1) for a given 2D vector representing an image.
     */
    double testAnImage(vector<vector<double> > anImage);

    /*
     * Saves the trained weights before and after the Hidden Layer to 2 specified file names (one for each).
     */
    void saveWeights(char *preFileName, char *postFileName);

    /*
     * Loads in the prior trained weights from 2 different files.
     */
    void loadWeights(char *preFileName, char *postFileName);

    /*
     * For when a user wants to adjust the learning rate.
     */
    void setLearningRate(double newRate);
};


#endif //CNN_CNN_H
