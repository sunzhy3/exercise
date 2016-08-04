import dice, pylab

fair = [1, 2, 3, 4, 5, 6]

def craps(die1, die2):
    d1, d2 = dice.throwDice(die1, die2)
    tot = d1 + d2
    if tot in [7, 11]: return True
    if tot in [2, 3, 12]: return False
    point = tot
    while True:
        d1, d2 = dice.throwDice(die1, die2)
        tot = d1 + d2
        if tot == 7: return False
        if tot == point: return True

def simCraps(numBets, die1, die2):
    wins, losses = 0, 0
    for i in xrange(numBets):
        if craps(die1, die2): wins += 1
        else: losses += 1
    houseWin = float(losses) / float(numBets)
    print wins, losses
    print houseWin
    print "the casino win for {0} percent".format(str(houseWin * 100))
    print "for {0} times craps the casino will get {1} win".format(numBets, losses - wins)

if __name__ == '__main__':
    print "start simulation" 
    numThrows = 100000
    simCraps(numThrows, fair, fair)
    weighted = [1, 2, 3, 4, 5, 5, 6]
    throws = dice.conductTrials(numThrows, fair, weighted)
    sums = pylab.array([0] * 14)
    for val in xrange(2, 13):
        sums[val] = throws.count(val)
    probs = sums[2: 13] / float(numThrows)
    xVals = pylab.arange(2, 13)
    pylab.plot(xVals, probs, label = 'Weighted Dice')
    pylab.legend()
    simCraps(numThrows, fair, weighted)
    pylab.show()
