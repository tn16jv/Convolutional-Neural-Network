#include <numeric>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>      // null, and I/O
#include <stdlib.h>     // for srand, rand
#include <omp.h>        // for OpenMP parallelization

using namespace std;

/*
 * Returns a dot product of 2 2D vectors.
 */
double dotProduct(vector<double> a, vector<double> b) {
    int length = a.size();
    double product = 0;
    for (int i=0; i<length; i++) {
        product += a[i] * b[i];
    }
    return product;
}

/*
 * dotProduct(vector<double> a, double b) returns a dotProduct (1D array vector) of a 1D array vector and a point
 */
vector<double> dotProduct(vector<double> a, double b) {
    int length = a.size();
    vector<double> product(length);
    for (int i=0; i<length; i++) {
        product[i] = a[i] * b;
    }
    return product;
}

int dotProduct2D(vector<vector<int> > a, vector<vector<int> > b) {
    int yLength = a.size();
    int xLength = a[0].size();
    int product = 0;
    for (int i=0; i<yLength; i++) {
        for (int j=0; j<xLength; j++) {
            product += a[i][j] * b[i][j];
        }
    }
    return product;
}

/*
 * Returns the multiplication a 2D vector and 1D vector with the same amount of elements.
 */
vector<vector<double> > multiplyVectors(vector<vector<double> > twoDim, vector<double> oneDim) {
    int yLength = twoDim.size();
    int xLength = twoDim[0].size();
    vector<vector<double> > result(yLength);
    for (int i=0; i<yLength; i++) {
        vector<double> row(xLength);
        for (int j=0; j<xLength; j++) {
            row[j] = twoDim[i][j] * oneDim[i*yLength + j];
        }
        result[i] = row;
    }
    return result;
}

/*
 * Rotates the elements of a passed in 2D vector by 180 degrees and returns it.
 * i.e. the first element is now the old last element, the second element of first row is second last element of last row...
 */
vector<vector<double> > rotateVector(vector<vector<double> > matrix) {
    int yLength = matrix.size();
    int xLength = matrix[0].size();
    vector<vector<double> > result(yLength);
    for (int i=0; i<yLength; i++) {
        vector<double> row(xLength);
        result[i] = row;
    }
    for (int i=0; i<yLength; i++) {
        for (int j=0; j<xLength; j++) {
            result[i][j] = matrix[yLength - 1 - i][xLength - 1 - j];
        }
    }
    return result;
}

/*
 * Returns a random floating point number in a given interval between min and max.
 */
double randDouble(int min, int max) {
    double randNum = (double)rand() / RAND_MAX;
    return min + randNum * (max - min);
}

/*
 * Returns derivative of the tanh (a hyperbolic) function. This can be used as a Sigmoid function.
 */
double dtanh(double x) {
    return 1 - pow(tanh(x), 2);
}

/*
 * Returns the logarithmic Sigmoid function.
 */
double logarithm(double x) {
    return 1 / (1 + exp(-1 * x));
}

/*
 * Returns derivative the logarithm function.
 */
double dlog(double x) {
    return logarithm(x) * (1 - logarithm(x));
}

/*
 * Applies a given function that returns a double onto a given vector of doubles.
 */
vector<double> funcOnVector(double (*f)(double), vector<double> elements) {
    int length = elements.size();
    vector<double> result(length);

    for (int i=0; i<length; i++) {
        result[i] = f(elements[i]);
    }
    return result;
}

/*
 * Applies a given function that returns a double onto a given 2D vector of doubles. This vector must be well defined.
 */
vector<vector<double> > funcOnVector2D(double (*f)(double), vector<vector<double> > elements) {
    int yLength = elements.size();
    int xLength = elements[0].size();
    vector<vector<double> > result(yLength);     // must declare vector's size to be usable

    for (int i=0; i<yLength; i++) {
        vector<double> row(xLength);
        for (int j=0; j<xLength; j++) {
            row[j] = f(elements[i][j]);
        }
        result[i] = row;
    }
    return result;
}

/*
 * Takes an inputted 2D vector and transforms it into a corresponding 1D vector with the same values.
 */
vector<double> flatten2D (vector<vector<double> > elements) {
    int yLength = elements.size();
    int xLength = elements[0].size();
    vector<double> result(yLength * xLength);

    for (int i=0; i<yLength; i++) {
        for (int j=0; j<xLength; j++) {
            result[i*yLength + j] = elements[i][j];     // copied to 1D with formula (rowIndex * length + columnIndex)
        }
    }
    return result;
}

/*
 * Does a convolution of one 2D vector onto another 2D vector. The 2D vector representing the filter must be of
 * n x n sized dimension and n<=image_width and n<= image_length.
 * More specifically, a n x n portion of the image is taken and multiplied with the filter and summed up, and then
 * put into the result matrix. This happens across the entire image with the same filter.
 */
vector<vector<double> > convolve2D(vector<vector<double> > image, vector<vector<double> > filter) {
    int imageY = image.size();
    int imageX = image[0].size();
    int filterY = filter.size();
    int filterX = filter[0].size();

    int resultY = imageY - (filterY - 1);   // convoluted result is shorter by 1 minus filter length
    int resultX = imageX - (filterX - 1);
    double result[resultY][resultX];

    double product;
    int i, j;
    int threads = 4;
    #pragma omp parallel num_threads(threads) shared(resultY, resultX, result)
    {
        # pragma omp for private(i, j, product) schedule(dynamic, resultX)
        for (int a = 0; a < resultY * resultX; a++) {     // loop that builds the result rows and columns
            i = a / resultX;
            j = a % resultX;
            product = 0;
            for (int l = 0; l < filterY; l++) {     // apply filter to area
                for (int k = 0; k < filterX; k++) {
                    product += image[i + l][j + k] * filter[l][k];
                }
            }
            result[i][j] = product;
        }
    }
    vector<vector<double> > toReturn(resultY);       // empty vector created to copy the primitive 2D result array
    for (int i=0; i<resultY; i++) {
        double* thing = result[i];      // pointer to a row from the primitive 2D result array
        vector<double> something(thing, thing+resultX);
        toReturn[i] = something;
    }
    return toReturn;
}

vector<vector<double> > convolve2Dsequential(vector<vector<double> > image, vector<vector<double> > filter) {
    int imageY = image.size();
    int imageX = image[0].size();
    int filterY = filter.size();
    int filterX = filter[0].size();

    int resultY = imageY - (filterY - 1);   // convoluted result is shorter by 1 minus filter length
    int resultX = imageX - (filterX - 1);
    double result[resultY][resultX];

    double product;
    for (int i=0; i<resultY; i++) {     // loop that builds the result rows
        for (int j=0; j<resultX; j++) {     // loop that builds the result columns
            product = 0;
            for (int l=0; l<filterY; l++) {     // these 2 nested loops is the multiplication of the filter on the image
                for (int k=0; k<filterX; k++) {
                    product += image[i+l][j+k] * filter[l][k];
                }
            }
            result[i][j] = product;
        }
    }

    vector<vector<double> > toReturn(resultY);       // empty vector created to copy the primitive 2D result array
    for (int i=0; i<resultY; i++) {
        double* thing = result[i];      // pointer to a row from the primitive 2D result array
        vector<double> something(thing, thing+resultX);
        toReturn[i] = something;
    }
    return toReturn;
}

/*
 * Acts like convolve2D() but the image is not reduced in size. Instead, it is padded by zeros.
 * Was made into a separate function for clarity purposes.
 */
vector<vector<double> > convolve2Dpad(vector<vector<double> > image, vector<vector<double> > filter) {
    int imageY = image.size();
    int imageX = image[0].size();
    int filterY = filter.size();
    int filterX = filter[0].size();

    int padY = (filterY - 1) / 2;
    int padX = (filterX - 1) / 2;
    double result[imageY][imageX];

    double product;
    for (int i=0; i<imageY; i++) {
        for (int j=0; j<imageX; j++) {
            product = 0;
            bool toPad = i < padY || i >= (imageY - padY) || j < padX || j >=(imageX - padX);
            // If we are not in the borders to pad with 0, we perform convolution normally.
            if (!toPad) {
                for (int l = 0; l < filterY; l++) {     // these loops are the multiplication of the filter on the image
                    for (int k = 0; k < filterX; k++) {
                        product += image[i + l - padY][j + k - padX] * filter[l][k];
                    }
                }
            }
            result[i][j] = product;
        }
    }
    vector<vector<double> > toReturn(imageY);       // empty vector created to copy the primitive 2D result array
    for (int i=0; i<imageY; i++) {
        double* thing = result[i];      // pointer to a row from the primitive 2D result array
        vector<double> something(thing, thing+imageY);
        toReturn[i] = something;
    }
    return toReturn;
}

