#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

void vectorToFile(vector<vector<int> > vect,char *name){
	//opening up file stream
	ofstream file;
	file.open(name,ios::out);
	
	//putting data from vector into file
	for(int i=0;i<vect[0].size();i++){
		for(int j=0;j<vect.size();j++){
			file<<vect[j][i];    //adding individual element to file
			file<<endl;          //delimeter to seperate elements
		}//for j loop
	}//for i loop
	
	//finishing up
	file.close();
}//vectorToFile