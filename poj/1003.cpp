#include <iostream>
using namespace std;
int main(){
    float oh;
	while(cin>>oh && oh != 0.0) {
		float s = 0.0;
        int j = 1;
		for(; s < oh; j++){
			s = s + 1.0/(j + 1);
		}
		cout<<j - 1<<" card(s)"<<endl;
	}
	return 0;
}
