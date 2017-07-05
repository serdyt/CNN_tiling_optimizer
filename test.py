# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 17:21:20 2017

@author: serdyt
"""
from math import ceil, floor
from LoopType import RowCol
   
def ALUperm(depth, MAC, maxMAC, maxlist, minlist):
    if depth == 1:
        if (MAC < minlist[0]):
            pass
        else:
            yield [min(MAC, maxlist[0])]
    else:
        for i in divisorsFloor(maxMAC, minlist[0], maxlist[0]):
            for perm in ALUperm(depth-1, int(floor(MAC/float(i))), maxMAC, maxlist[1:], minlist[1:]):
                yield [i] + perm

def divisorsCeil(mini,maxi):
    res = [mini]
    for i in xrange(maxi, 0, -1):
        t = ceil(maxi/float(i))
        if t != res[-1] and t >= mini:
            res.append(int(t))
    return res
    
def nextDivisor(curr, maxi):
    if type(curr) == RowCol:
        return RowCol(curr.row, nextDivisor(curr.col, maxi.col))
    else:
        for i in xrange(maxi, 0, -1):
            t = ceil(maxi/float(i))
            if t > curr:
                return int(t)

def divisorsFloor(x, mini, maxi):
    res = [maxi]
    for i in xrange(1, x+1):
        t = floor(x/float(i))
        if (t >= mini and t <= maxi):
            if len(res) == 0:
                res.append(int(t))
            else:
                if t != res[-1]:
                    res.append(int(t))
            
    return res

print divisorsCeil(1,48)

#print divisorsFloor(256, 5,5)

#for ii in divisorsCeil(4,48):
#    for i in divisorsCeil(4,ii):
#        print i, ii
#
#i = 0
#tot = 0
#for div in d:
#    i += 1
#    tot += len(divisors(div))
#    
#print tot

#for i in ALUperm(4,256,256,[5,5,48,25],[5,5,1,1]):
#    print i, reduce(lambda x,y:x*y, i)