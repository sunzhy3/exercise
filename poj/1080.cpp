#include <iostream>
#include <string>
using namespace std;
#define MIN_INT -100
const int SA[5][5]={5,-1,-2,-1,-3, -1,5,-3,-2,-4, -2,-3, 5,-2,-2, -1,-2,-2,5,-1, -3,-4,-2,-1, MIN_INT};
int index(char gene) {
    switch(gene){
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        deafult: return 4;
    }
}
int max(int a, int b) {
    return a> b? a:b;
}
int main(){
	int numSamples;
    cin>> numSamples;
	for(int i = 0; i < numSamples; i++) {
        int len1, len2;
        string s1, s2;
        cin>> len1>> s1>> len2>> s2;
		int score[110][110];
        for(int i = 0; i <= len1; i++) {
            for(int j = 0; j <= len2; j++) {
                if(i == 0 && j == 0) {
                    score[i][j] = 0;
                    continue;
                }
                int tmp = MIN_INT;
                if(i > 0) tmp = max(tmp, score[i - 1][j] + SA[index(s1[i - 1])][4]);
                if(j > 0) tmp = max(tmp, score[i][j - 1] + SA[4][index(s2[j - 1])]);
                if(i > 0 && j > 0) tmp = max(tmp, score[i - 1][j - 1] + SA[index(s1[i - 1])][index(s2[j - 1])]);
                score[i][j] = tmp;
            }
        }
        cout<< score[len1][len2]<<endl;
	}
    return 0;
}
