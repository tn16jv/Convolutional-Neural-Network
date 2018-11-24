#include "fileToVector.cpp"
#include <vector>

using namespace std;

char name[7] = {'o','u','t','p','u','t','1'};

int main(int argv, char *argc[]){
	vector<vector<int> > picture = fileToVector(name);
	
	for(int i=0;i<picture[0].size();i++){
		for(int j=0;j<picture.size();j++){
			std::cout<<picture[j][i]<<endl;
		}//for j loop
	}//for i loop
}//main