// Example program
#include <iostream>
#include <string>
#include <vector>

class polygon {
  public:
    polygon(std::vector<int>&& aa) {
        // a.clear();
        ss = std::move(aa);
        // ss = aa;
    }
    
    std::vector<int> ss;
};

int main() {
    std::vector<int> a({1, 2, 3});
    polygon tp(std::move(a));
    // a[0] = 4;
    std::cout << "a" << std::endl;
    for (const auto& x : a) {
        std::cout << x << " " << std::endl;
    }
    std::cout << "ss" << std::endl;
    for (const auto& x : tp.ss) {
        std::cout << x << " " << std::endl;
    }
}