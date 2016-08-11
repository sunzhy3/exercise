from pylab import *
import math, random

def throwDarts(numDarts, shouldPlot):
    inCircle = 0;
    estimates = []
    for darts in xrange(1, numDarts - 1, 1):
        x = random.random()
        y = random.random()
        if math.sqrt(x*x + y*y) <= 1.0:
            inCircle += 1
        if shouldPlot:
            piGuess = 4 * (inCircle / float(darts))
            estimates.append(piGuess)
        if(darts % 1000000 == 0):
            piGuess = 4 * (inCircle / float(darts))
            print "Estimation with ", darts, " darts ", piGuess
    if shouldPlot:
        xAxis = xrange(1, len(estimates) + 1)
        semilogx(xAxis, estimates)
        titleString = str(piGuess)
        title(titleString)
        xlabel("number of estimations")
        ylabel("the estimate value of pi")
        axhline(3.14159)
    return 4 * (inCircle / float(darts))

def findPI(numDarts, shouldPlot = False):
    piGuess = throwDarts(numDarts, shouldPlot)
    print "Estimation of pi with ", numDarts, " darts: ", piGuess
if __name__ == '__main__':
    print("start to run")
    findPI(1000000, True)
    show()
