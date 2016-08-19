import theano
from theano import tensor as T

a = T.scalar()
b = T.scalar()

y = a * b
multiply = theano.function(inputs = [a, b], output = y)

print multiply(3, 2)
print multiply(4, 5)
