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
    int cores = 4;
    /*
     * This is a simplified example meant to test the power of parallelization. This example uses a single 100x100
     * greyscale image to train on and may take up to 10 seconds to finish.
     * The CNN at the end should recognize very well if the image is similar to it.
     */
    cout<<"Running with "<<cores<< " cores..."<<endl;
    vector<vector<vector<double> > > images (1);
    images[0] = read2DVector("bigExample");

    vector<double> expects (1);
    expects[0] = 1.0;
    ConvulutionalNeuralNetwork neural = ConvulutionalNeuralNetwork(images, expects);

    // This times and outputs long the training takes
    t1 = std::chrono::system_clock::now();
    neural.train(cores);
    t2 = std::chrono::system_clock::now();
    duration = duration_cast<microseconds>(t2 - t1).count();
    cout<<"Full training was: "<<duration<<" microseconds"<<endl;

    // This block times and outputs how long the parallel (with specified cores) convolution takes on a 400x400 image
    vector<vector<double> > picture = read2DVector("bigExample2");
    vector<vector<double> > filter = read2DVector("identity");
    t1 = std::chrono::system_clock::now();
    convolve2D(picture, filter, cores);
    t2 = std::chrono::system_clock::now();
    duration = duration_cast<microseconds>(t2 - t1).count();
    cout<<"Parallel convolution was: "<<duration<<" microseconds"<<endl;

    // This block times and outputs how long sequential convolution takes on the same 400x400 image
    t1 = std::chrono::system_clock::now();
    convolve2D(picture, filter, 1);         // running with 1 cores will be sequential
    t2 = std::chrono::system_clock::now();
    duration = duration_cast<microseconds>(t2 - t1).count();
    cout<<"Sequential convolution was: "<<duration<<" microseconds"<<endl;

    cout<<"Output layer at the end of training: "<<endl;
    vector<double> output = neural.outputLayer();
    for (int i=0; i<output.size(); i++) {
        std::cout<<output[i]<<" ";
    }
    std::cout<<endl;

    return 0;
}