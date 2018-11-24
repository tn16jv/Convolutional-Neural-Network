//
// Created by talba on 2018-10-02.
//

#ifndef CNN_UTILITY_H
#define CNN_UTILITY_H

#endif //CNN_UTILITY_H
#include <vector>

using namespace std;

double dotProduct(vector<double> a, vector<double> b);

vector<double> dotProduct(vector<double> a, double b);

int dotProduct2D(vector<vector<int> > a, vector<vector<int> > b);

vector<vector<double> > multiplyVectors(vector<vector<double> > twoDim, vector<double> oneDim);

vector<vector<double> > rotateVector(vector<vector<double> > matrix);

double randDouble(int min, int max);

double dtanh(double x);

double logarithm(double x);

double dlog(double x);

vector<double> funcOnVector(double (*f)(double), vector<double> elements);

vector<vector<double> > funcOnVector2D(double (*f)(double), vector<vector<double> > elements);

vector<double> flatten2D (vector<vector<double> > elements);

vector<vector<double> > convolve2D(vector<vector<double> > image, vector<vector<double> > filter);
