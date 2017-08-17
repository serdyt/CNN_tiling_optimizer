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
                
def ALUperm2(depth, MAC, maxMAC, maxlist, minlist):
    if depth == 1:
        if (MAC < minlist[-1]):
            pass
        else:
            yield [min(MAC, maxlist[-1])]
    else:
        for i in divisorsFloor(maxMAC, minlist[len(minlist) - depth], maxlist[len(maxlist) - depth]):
            for perm in ALUperm2(depth-1, int(floor(MAC/float(i))), maxMAC, maxlist, minlist):
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

#print divisorsCeil(1,48)

def testFiles():
    a = open('CNP3000_level_update')
    b = open('CNP3000_usedData_update')
    
    i = 0
    for la, lb in zip(a,b):
        if la != lb:
            print la
            print lb
            print i
            break
        i += 1
    a.close()        
    b.close()
    print i
    
#testFiles()
    
def csvReduce():
    import csv
    out = open('CNP30000_reduced', 'wb')
    spamwriter = csv.writer(out, delimiter='\t', quotechar='|')
    
    i = 0
    with open('CNP30000', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
#            print row
            if float(row[-1]) < 12000:
                spamwriter.writerow(row)
                i+=1
                
        csvfile.close()
                
    print i
    
    out.close()
   
def _process(s):
    for i in xrange(100000000):
        j = i * i
       
from  multiprocessing.pool import ThreadPool
import time
import os

#thread_pool_size = multiprocessing.cp.pu_count()
pool = ThreadPool(2)
for single_string in range(10):
    pool.apply_async(_process, [single_string ])
pool.close()
pool.join()


    
    # Do staff, pure python string manipulation

#    # evaluate "f(20)" asynchronously
#    res = pool.apply_async(f, (20,))      # runs in *only* one process
#    print res.get(timeout=1)              # prints "400"
#
#    # evaluate "os.getpid()" asynchronously
#    res = pool.apply_async(os.getpid, ()) # runs in *only* one process
#    print res.get(timeout=1)              # prints the PID of that process
#
#    # launching multiple evaluations asynchronously *may* use more processes
#    multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
#    print [res.get(timeout=1) for res in multiple_results]
#
#    # make a single worker sleep for 10 secs
#    res = pool.apply_async(time.sleep, (10,))
#    try:
#        print res.get(timeout=1)
#    except TimeoutError:
#        print "We lacked patience and got a multiprocessing.TimeoutError"
        
        
          
          
#csvReduce()
      
    

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

#a = []
#for i in ALUperm(2,256,256,[48,64],[2,1]):
##    print i, reduce(lambda x,y:x*y, i)
#    a.append(i)

#for i in ALUperm2(3,256,256,[5,55,256],[5,1,1]):
#    print i, reduce(lambda x,y:x*y, i)    

