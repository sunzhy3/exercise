#include <iostream>
using namespace std;
enum tower {A = 'A', B = 'B', C = 'C'};
void TowersOfHanoi(int n, tower x, tower y, tower z) {
	if(n){
		TowersOfHanoi(n - 1, x, z, y);
		cout<<"move top disk from tower "<< char(x)<< " to top of tower "<<char(y)<<endl;
		TowersOfHanoi(n - 1, z, y, x);
	}
}
int main(){
    tower src = A;
    tower dst = B;
    tower tmp = C;
	TowersOfHanoi(5, src, dst, tmp);
	return 0;
}
