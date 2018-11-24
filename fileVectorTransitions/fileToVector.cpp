#include <iostream>
#include <fstream>
#include <vector>

int PICTUREWIDTH = 10;
int PICTUREHEIGHT = 10;

using namespace std;

vector<vector<int> > fileToVector(char *name){
	//opening up file stream
	ifstream file;
	file.open(name,ios::in);
	
	//initializing vector to be returned
	vector<vector<int> > result;
	result.resize(PICTUREWIDTH,vector<int>(PICTUREHEIGHT));
	
	//loading vector with data from file
	for(int i=0;i<result[0].size();i++){
		for(int j=0;j<result.size();j++){
			file>>result[j][i];
		}//for j loop
	}//for i loop
	
	//finishing up
	file.close();
	return result;
}//fileToVector