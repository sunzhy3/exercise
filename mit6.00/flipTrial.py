from pylab import *
import random, math

def flipTrial(numTrials):
    heads, tails = 0, 0
    for i in xrange(0, numTrials):
        coin = random.randint(0, 1)
        if coin == 0: heads += 1
        else: tails += 1
    return heads, tails    

def simFilps(numFlips, numTrials):
    diffs = []
    for i in xrange(0, numTrials):
        heads, tails = flipTrial(numFlips)
        diffs.append(abs(heads - tails))
    diffs = array(diffs)
    diffMean = sum(diffs) / len(diffs)
    diffPercent = (diffs / float(numFlips)) * 100
    PercentMean = sum(diffPercent) / len(diffPercent)
    #plot the histogram
    hist(diffs)
    axvline(diffMean, label = 'Mean')
    legend()
    titleString = str(numFlips) + ' Flips ' + str(numTrials) + ' Trials'
    title(titleString)
    xlabel('Difference between heads and tails')
    ylabel('Number of Trials')
    #plot the different percent
    figure()
    plot(diffPercent)
    axhline(PercentMean, label = 'Mean')
    legend()
    title(titleString)
    xlabel('Trial Number')
    ylabel('Percent Difference between heads and tails')

if __name__ == '__main__':
    simFilps(100, 100)
    show()
