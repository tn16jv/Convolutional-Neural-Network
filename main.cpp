#include <iostream>
#include <assert.h>
#include <vector>
#include <cmath>
#include "utility.h"
#include "utilityIO.h"
#include "ConvulutionalNeuralNetwork.h"

using namespace std;

int main() {
    /*
     * This example will train the neural network to try and recognize images of X. This will be like a grayscale image
     * where 0 is white, and 1 and above is colored in.
     */
    vector<vector<vector<double> > > images (3);
    images[0] = read2DVector("Xpic.txt");       // the actual X image
    images[1] = read2DVector("Xpic2.txt");      // half of the X (bad image)
    images[2] = read2DVector("emptyX.txt");     // empty image

    vector<double> expects (3);
    expects[0] = 1.0;
    expects[1] = 0.7;
    expects[2] = 0.5;
    ConvulutionalNeuralNetwork neural = ConvulutionalNeuralNetwork(images, expects);
    neural.train();

    cout<<"Output layer at the end of training: "<<endl;
    vector<double> output = neural.outputLayer();
    for (int i=0; i<output.size(); i++) {
        std::cout<<output[i]<<" ";
    }
    std::cout<<endl;


    cout<<"Testing an X image that is slightly off:"<<endl;
    vector<vector<double> > anotherImage = read2DVector("XtestAfter.txt");
    cout<<neural.testAnImage(anotherImage)<<endl;
    return 0;
}