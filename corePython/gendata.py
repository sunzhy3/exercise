#!usr/bin/env python

from random import randrange, choice
from string import ascii_lowercase as lc
#from sys import maxint
from time import ctime

tlds = ('com', 'edu', 'net', 'org', 'gov')

for i in xrange(randrange(5, 11)):
    dtint = randrange(2**31)    #pick date
    dtstr = ctime(dtint)        #date string
    llen = randrange(4, 8)
    login = ''.join(choice(lc) for j in range(llen))
    dlen = randrange(llen, 13)
    dom = ''.join(choice(lc) for j in range(dlen))
    if i == 0:
        with open('redata.txt', 'w') as f:
            f.write('%s::%s@%s.%s::%d-%d-%d\n'
                    %(dtstr, login, dom,choice(tlds), dtint, llen, dlen))
    else:
        with open('redata.txt', 'a') as f:
            f.write('%s::%s@%s.%s::%d-%d-%d\n' 
                    %(dtstr, login, dom, choice(tlds), dtint, llen, dlen))
