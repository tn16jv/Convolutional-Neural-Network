//
// Created by talba on 2018-10-12.
//

#ifndef CNN_UTILITYIO_H
#define CNN_UTILITYIO_H

#endif //CNN_UTILITYIO_H

#include <vector>

using namespace std;

void write2DVector(vector<vector<double> > vect, char *fileName);

void write1DVector(vector<double> vect, char *fileName);

vector<vector<double> > read2DVector(char *fileName);

vector<double> read1DVector(char *fileName);
