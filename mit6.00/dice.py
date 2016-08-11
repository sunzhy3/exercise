import pylab, random
fair = [1, 2, 3, 4, 5, 6]

def throwDice(vals1, vals2):
    d1 = random.choice(vals1)
    d2 = random.choice(vals2)
    return d1, d2

def conductTrials(numThrows, die1, die2):
    throws = []
    for i in xrange(numThrows):
        d1, d2 = throwDice(die1, die2)
        throws.append(d1 + d2)
    return throws

if __name__ == '__main__':
    numThrows = 10000
    throws = conductTrials(numThrows,  fair, fair)
    pylab.hist(throws, 11)
    pylab.xticks(range(2, 13), ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    pylab.title("Distribution of Values")
    pylab.xlabel("Sum of Two Dice")
    pylab.ylabel("Number of Trials")

    pylab.show()
