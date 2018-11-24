#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

void write2DVector(vector<vector<double> > vect, char *fileName){
	//opening up file stream
	ofstream file;
	file.open(fileName,ios::out);
	
	file<<vect.size()<<endl;
	file<<vect[0].size()<<endl;
	
	//putting data from vector into file
	for(int i=0;i<vect[0].size();i++){
		for(int j=0;j<vect.size();j++){
			file<<vect[i][j];    //adding individual element to file
			file<<endl;          //delimeter to seperate elements
		}//for j loop
	}//for i loop
	
	//finishing up
	file.close();
}//write2DVector


void write1DVector(vector<double> vect, char *fileName){
	//opening up file stream
	ofstream file;
	file.open(fileName,ios::out);
	
	file<<vect.size()<<endl;
	
	//putting data from vector into file
	for(int i=0;i<vect.size();i++){
		file<<vect[i];       //adding individual element to file
		file<<endl;          //delimeter to seperate elements
	}//for i loop
	
	//finishing up
	file.close();
}//write1DVector

vector<vector<double> > read2DVector(char *fileName){
	int temp1, temp2;
	//opening up file stream
	ifstream file;
	file.open(fileName,ios::in);
	
	//initializing vector to be returned
	vector<vector<double> > result;
	file>>temp1;
	file>>temp2;
	result.resize(temp1,vector<double>(temp2));
	
	//loading vector with data from file
	for(int i=0;i<result[0].size();i++){
		for(int j=0;j<result.size();j++){
			file>>result[i][j];
		}//for j loop
	}//for i loop
	
	//finishing up
	file.close();
	return result;
}//read2DVector

vector<double> read1DVector(char *fileName){
	int temp1;
	//opening up file stream
	ifstream file;
	file.open(fileName,ios::in);
	
	//initializing vector to be returned
	vector<double> result;
	file>>temp1;
	result.resize(temp1);
	
	//loading vector with data from file
	for(int i=0;i<result.size();i++){
		file>>result[i];
	}//for i loop
	
	//finishing up
	file.close();
	return result;
}//read1DVector