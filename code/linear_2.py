import numpy as np

def initialize_parameters(layers_dims):
    """
    layers_dims: (L)
    """
    np.random.seed(1)               
    parameters = {}
    L = len(layers_dims)            

    for l in range(1, L):           
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))

        assert parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1])
        assert parameters["b" + str(l)].shape == (layers_dims[l], 1)
    
    return parameters

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, A

def tanh(Z):
    A = np.tanh(Z)
    return A, A

def relu(Z):
    A = np.maximum(0, Z)
    return A, A

def leaky_relu(Z):
    A = np.maximum(0.1 * Z, Z)
    return A, A

# Define helper functions that will be used in L-model forward prop
def linear_forward(X, W, b):
    """
    X: (N, D)
    W: (D, B)
    b: (B,)
    Z: (N, B)
    """
    Z = np.dot(X, W) + b
    cache = (X, W, b)
    return Z, cache

def linear_activation_forward(X, W, b, activation_fn):
    """
    X: (N, D)
    W: (D, B)
    b: (B,)
    """
    Z, linear_cache = linear_forward(X, W, b)

    if activation_fn == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation_fn == "tanh":
        A, activation_cache = tanh(Z)
    elif activation_fn == "relu":
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters, hidden_layers_activation_fn="relu"):
    A = X                           
    caches = []                     
    L = len(parameters) // 2        

    for l in range(1, L):
        A, cache = linear_activation_forward(
            A, 
            parameters["W" + str(l)], 
            parameters["b" + str(l)],
            activation_fn=hidden_layers_activation_fn
        )
        caches.append(cache)

    AL, cache = linear_activation_forward(
        A, 
        parameters["W" + str(L)], 
        parameters["b" + str(L)],
        activation_fn="sigmoid"
    )
    caches.append(cache)

    return AL, caches

# Compute cross-entropy cost
def compute_cost(AL, y):
    """
    AL: (N, C)
    y: (N, C)
    """
    N = y.shape[0]
    cost = - (1 / N) * np.sum(np.multiply(y, np.log(AL)) + np.multiply(1 - y, np.log(1 - AL)))
    return cost

def sigmoid_gradient(dA, A):
    dZ = A * (1 - A)
    return dA * dZ

def tanh_gradient(dA, A):
    dZ = 1 - np.square(A)
    return dA * dZ

def relu_gradient(dA, A):
    dZ = np.int64(A > 0)
    return dA * dZ

# define helper functions that will be used in L-model back-prop
def linear_backword(dZ, cache):
    """
    dZ: (N, B)
    W: (D, B)
    X: (N, D)
    b: (B)
    """
    X, W, b = cache
    N = X.shape[0]

    dW = (1 / N) * np.dot(X.T, dZ)
    db = (1 / N) * np.sum(dZ, axis=0)
    dX = np.dot(dZ, W.T)

    assert dX.shape == X.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape

    return dX, dW, db

def linear_activation_backward(dA, cache, activation_fn):
    linear_cache, activation_cache = cache

    if activation_fn == "sigmoid":
        dZ = sigmoid_gradient(dA, activation_cache)
    elif activation_fn == "tanh":
        dZ = tanh_gradient(dA, activation_cache)
    elif activation_fn == "relu":
        dZ = relu_gradient(dA, activation_cache)
    
    dX, dW, db = linear_backword(dZ, linear_cache)

    return dX, dW, db

def L_model_backward(AL, y, caches, hidden_layers_activation_fn="relu"):
    y = y.reshape(AL.shape)
    L = len(caches)
    grads = {}

    dAL = np.divide(AL - y, np.multiply(AL, 1 - AL))

    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(
        dAL, 
        caches[L - 1], 
        "sigmoid"
    )

    for l in range(L - 1, 0, -1):
        current_cache = caches[l - 1]
        grads["dA" + str(l - 1)], grads["dW" + str(l)], grads["db" + str(l)] = linear_activation_backward(
            grads["dA" + str(l)], 
            current_cache,
            hidden_layers_activation_fn
        )

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters

# Define the multi-layer model using all the helper functions we wrote before
def L_layer_model(
        X,
        y,
        layers_dims,
        learning_rate=0.01,
        num_iterations=3000,
        print_cost=True,
        hidden_layers_activation_fn="relu"):
    np.random.seed(1)

    # initialize parameters
    parameters = initialize_parameters(layers_dims)

    # intialize cost list
    cost_list = []

    # iterate over num_iterations
    for i in range(num_iterations):
        # iterate over L-layers to get the final output and the cache
        AL, caches = L_model_forward(
            X, parameters, hidden_layers_activation_fn)

        # compute cost to plot it
        cost = compute_cost(AL, y)

        # iterate over L-layers backward to get gradients
        grads = L_model_backward(AL, y, caches, hidden_layers_activation_fn)

        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        # append each 100th cost to the cost list
        if (i + 1) % 100 == 0 and print_cost:
            print(f"The cost after {i + 1} iterations is: {cost:.4f}")

        if i % 100 == 0:
            cost_list.append(cost)

    return parameters

def accuracy(X, parameters, y, activation_fn="relu"):
    probs, caches = L_model_forward(X, parameters, activation_fn)
    labels = (probs >= 0.5) * 1
    accuracy = np.mean(labels == y) * 100
    return f"The accuracy rate is: {accuracy:.2f}%."

if __name__ == "__main__":
    X_train = np.random.rand(209, 64 * 64 * 3)
    y_train = np.random.rand(209, 1)
    X_test = np.random.rand(50, 64 * 64 * 3)
    y_test = np.random.rand(50, 1)
    # Setting layers dims
    layers_dims = [X_train.shape[1], 5, 5, 1]
    # NN with tanh activation fn
    parameters_tanh = L_layer_model(
        X_train, 
        y_train, 
        layers_dims, 
        learning_rate=0.03, 
        num_iterations=3000, 
        hidden_layers_activation_fn="tanh")
    
    # Print the accuracy
    accuracy(X_test, parameters_tanh, y_test, activation_fn="tanh")
