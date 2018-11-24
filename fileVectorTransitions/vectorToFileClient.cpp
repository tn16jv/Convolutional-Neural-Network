#include "vectorToFile.cpp"
#include <vector>

using namespace std;

int  NUMPICS = 7;
char name[7] = {'o','u','t','p','u','t','1'};

int main(int argv, char *argc[]){
	std::vector<vector<int> > picture;
	picture.resize(10,std::vector<int>(10));
	
	for(int i=0;i<picture[0].size();i++){
		for(int j=0;j<picture.size();j++){
			picture[j][i] = i+j;
		}//for j loop
	}//for i loop
	
	vectorToFile(picture,name);
}//main