#include <iostream>
#include <vector>
#include <assert.h>
using namespace std;
class LinearLayer{
private:
    vector<vector<double>> weights;
    vector<double> output_vals;
    vector<double> input_vals;
    int output_dim;
    int input_dim;
    double eta;
public:
    LinearLayer(){}
    LinearLayer(int input_size, int output_size, double lr);
    vector<double> feedForward(const vector<double> &input);
    vector<double> backPropagate(const vector<double> &grad);
};
LinearLayer::LinearLayer(int input_size, int output_size, double lr){
    assert(input_size > 0);
    assert(output_size > 0);
    output_dim = output_size;
    input_dim = input_size;
    eta = lr;

    //generate random weights
    for(int out = 0; out < output_size; out++){
        weights.push_back(vector<double>());
        for(int input = 0; input < input_size + 1; input++){ //we create an extra weight (one more than input_size) for our bias
            weights.back().push_back((double)rand() / RAND_MAX); //random value between 0 and 1
        }
    }
}
vector<double> LinearLayer::feedForward(const vector<double> &input){
    assert(input.size() == input_dim);
    output_vals = vector<double>();
    input_vals = input; //store the input vector

    //perform matrix multiplication
    for(int out = 0; out < output_dim; out++){
        double sum = 0.0;
        for(int w = 0; w < input_dim; w++){
            sum += weights[out][w] * input[w];
        }
        sum += weights[out][input_dim]; //account for the bias
        output_vals.push_back(sum);
    }
    return output_vals;
}
vector<double> LinearLayer::backPropagate(const vector<double> &grad){
    assert(grad.size() == output_dim);
    vector<double> prev_layer_grad;

    //calculate partial derivatives with respect to input values
    for(int input = 0; input < input_dim; input++){
        double g = 0.0;
        for(int out = 0; out < output_dim; out++){
            g += (grad[out] * weights[out][input]);
        }
        prev_layer_grad.push_back(g);
    }

    //change weights using gradient
    for(int out = 0; out < output_dim; out++){
        for(int input = 0; input < input_dim; input++){
            weights[out][input] -= (eta * grad[out] * input_vals[input]);
        }
        weights[out][input_dim] -= eta * grad[out];
    }
    
    //return computed partial derivatives to be passed to preceding layer
    return prev_layer_grad;
}