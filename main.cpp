#include <iostream>
#include <assert.h>
#include <chrono>
#include <vector>
#include <cmath>
#include "utility.h"
#include "utilityIO.h"
#include "ConvulutionalNeuralNetwork.h"

using namespace std;
using namespace std::chrono;

int main() {
    std::chrono::time_point<std::chrono::system_clock> t1;
    std::chrono::time_point<std::chrono::system_clock> t2;
    int duration;
    /*
     * This example will train the neural network to try and recognize images of X. This will be like a grayscale image
     * where 0 is white, and 1 and above is colored in.
     */
    vector<vector<vector<double> > > images (3);
    images[0] = read2DVector("four.txt");       // the actual X image
    images[1] = read2DVector("badFour.txt");      // half of the X (bad image)
    images[2] = read2DVector("empty.txt");     // empty image

    vector<double> expects (3);
    expects[0] = 1.0;
    expects[1] = 0.7;
    expects[2] = 0.5;
    ConvulutionalNeuralNetwork neural = ConvulutionalNeuralNetwork(images, expects);

    t1 = std::chrono::system_clock::now();
    neural.train();
    t2 = std::chrono::system_clock::now();
    duration = duration_cast<microseconds>(t2 - t1).count();
    cout<<"Full training was "<<duration<<" microseconds"<<endl;

    vector<vector<double> > picture = read2DVector("bigExample2");
    vector<vector<double> > filter = read2DVector("identity");
    t1 = std::chrono::system_clock::now();
    convolve2D(picture, filter);
    t2 = std::chrono::system_clock::now();
    duration = duration_cast<microseconds>(t2 - t1).count();
    cout<<"Sequential convolution was "<<duration<<" microseconds"<<endl;

    t1 = std::chrono::system_clock::now();
    convolve2Dparallel(picture, filter);
    t2 = std::chrono::system_clock::now();
    duration = duration_cast<microseconds>(t2 - t1).count();
    cout<<"Parallel convolution was "<<duration<<" microseconds"<<endl;

    cout<<"Output layer at the end of training: "<<endl;
    vector<double> output = neural.outputLayer();
    for (int i=0; i<output.size(); i++) {
        std::cout<<output[i]<<" ";
    }
    std::cout<<endl;

/*
    cout<<"Testing an X image that is slightly off:"<<endl;
    vector<vector<double> > anotherImage = read2DVector("XtestAfter.txt");
    cout<<neural.testAnImage(anotherImage)<<endl;
    neural.saveWeights("preHiddenWeights.txt", "postHiddenWeights.txt");

    vector<vector<double> > picture = read2DVector("four.txt");
    vector<vector<double> > filter = read2DVector("identity");
    vector<vector<double> > result1 = convolve2D(picture, filter);
    vector<vector<double> > result2 = convolve2Dpad(picture, filter);
    result2 = convolve2Dpad(picture, filter);
    */
/*
    for (int i=0; i<picture.size(); i++) {
        for (int j=0; j<picture[0].size(); j++) {
            cout<<picture[i][j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;

    for (int i=0; i<filter.size(); i++) {
        for (int j=0; j<filter[0].size(); j++) {
            cout<<filter[i][j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;

    for (int i=0; i<result1.size(); i++) {
        for (int j=0; j<result1[0].size(); j++) {
            std::cout<<result1[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    cout<<endl<<endl;
    for (int i=0; i<result2.size(); i++) {
        for (int j=0; j<result2[0].size(); j++) {
            std::cout<<result2[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    */
    return 0;
}