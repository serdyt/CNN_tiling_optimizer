from math import ceil
from operator import add, mul

from LoopType import LoopType
from Model import *

class Buffers(object):
    # [RF, localBuffer, DRAM]
    
    def __init__(self):
#        self.levels = 3
        self.Bin = [0,0]
        self.Bkern = [0,0]
        self.Bout = [0,0]

        self.RRin = [1,1,1]
        self.RRkern = [1,1,1]
        self.RRout = [1,1,1]
        
#        self.DRin = [0,0,0]
#        self.DRkern = [0,0,0]
#        self.DRout = [0,0,0]
#                
    def totalSpace(self):
        return sum(self.Bin) + sum(self.Bkern) + sum(self.Bout)
                    
    def calcUsedData(self, left,loop):
        #scan all the data to the left to find the amount of each data type being used
        usedData = [1]*len(LoopType)
        for i in LoopType:
            rawData = [ x for x in left if x.type == i ]
            usedData[i.value] = 1 if len(rawData) == 0 else rawData[-1].size
        return usedData
        
    def concatBuff(self, buff):
        self.Bin = map(max, self.Bin, buff.Bin)
        self.Bout = map(max, self.Bout, buff.Bout)
        self.Bkern = map(max, self.Bkern, buff.Bkern)
        
        self.RRin = map(mul, self.RRin, buff.RRin)
        self.RRout = map(mul, self.RRout, buff.RRout)
        self.RRkern = map(mul, self.RRkern, buff.RRkern)
            
    def calcBuffSizeRR(self, left, loop, arch):
        # TODO: Move RRout*2 to Buffers (it is in calcEnergy now)
    
        # avoid unnececary computations
        if loop.size == 1:
            return
    
        level = len([x.type for x in left if x.type == loop.type ])
        if level < 0 and level > 2:
            print "WTF", level, loop
            raise Exception('Too many loop levels') 

#        print left, loop, arch
        usedData = self.calcUsedData(arch+left, loop)

        # avoid unnececary computations
        # this would generate a bufer with RR=1
        if loop.size == usedData[loop.type.value]:
            return

        # calc buffer size only for local buffers
        if level < 2:
            if loop.type == LoopType.kern:
                self.Bin[level] = kernBuff( usedData[LoopType.row.value], 
                                            usedData[LoopType.dx.value], 
                                            usedData[LoopType.col.value], 
                                            usedData[LoopType.dy.value], 
                                            usedData[LoopType.fm.value])
                                            
            elif loop.type == LoopType.fm:
                self.Bout[level] = fmBuff(  usedData[LoopType.row.value],
                                            usedData[LoopType.col.value],
                                            usedData[LoopType.kern.value])
                                            
            elif (loop.type == LoopType.row or loop.type == LoopType.col):
                self.Bkern[level] = rowcolBuff( usedData[LoopType.kern.value],
                                                usedData[LoopType.fm.value],
                                                usedData[LoopType.dx.value],
                                                usedData[LoopType.dy.value])
                                                
            elif loop.type == LoopType.dx or loop.type == LoopType.dy:
                self.Bin[level], self.Bout[level] = \
                                   dxdyBuff(usedData[LoopType.col.value],
                                            usedData[LoopType.dx.value],
                                            usedData[LoopType.row.value],
                                            usedData[LoopType.dy.value],                                
                                            usedData[LoopType.fm.value],
                                            usedData[LoopType.kern.value] )

            # calc RR
            if loop.type == LoopType.kern:
                self.RRin[level] *= kernRR( loop.size,
                                            usedData[LoopType.kern.value])
    
            elif loop.type == LoopType.fm:
                self.RRout[level] *= fmRR(  loop.size,
                                            usedData[LoopType.fm.value])
    
            elif loop.type == LoopType.row:
                self.RRkern[level] *= rowRR(loop.size,
                                            usedData[LoopType.row.value])
    
            elif loop.type == LoopType.col:
                self.RRkern[level] *= colRR(loop.size,
                                            usedData[LoopType.col.value])
                           
            #TODO: check levels of dx dy
            elif loop.type == LoopType.dx:
                rrin, rrout = dxRR(loop.size,
                                    usedData[LoopType.dx.value],
                                    usedData[LoopType.col.value])
                self.RRin[level] *= rrin
                self.RRout[level] *= rrout                                                 
        
            elif loop.type == LoopType.dy:
                rrin, rrout = dyRR(loop.size,
                                   usedData[LoopType.dy.value],
                                   usedData[LoopType.row.value])
                self.RRin[level] *= rrin
                self.RRout[level] *= rrout

#    def mergeBuff(self, hwRestrictions):
#        for b,rr in zip([self.Bin, self.RRin, self.Bkern], [self.RRkern, self.Bout, self.RRout]):
#            if 0 in b:
                
          
#    def shiftLeft(self):
#        for x in [self.Bin, self.RRin, self.Bkern, self.RRkern, self.Bout, self.RRout]:
#            x.sort(key=lambda v: v!= 0)            
           
    def __str__ (self):
        return "Bin " + str(self.Bin) + str(self.RRin) + " \n" + \
               "Bkern " + str(self.Bkern) + str(self.RRkern) + " \n" + \
               "Bout " + str(self.Bout) + str(self.RRout)  #+ "\n" + \
#               "DRin " + str(self.DRout) + str([b*r for b,r in zip(self.Bin,self.RRin)]) + "\n" + \
#               "DRkern " + str(self.DRkern) + str([b*r for b,r in zip(self.Bkern,self.RRkern)]) + "\n" + \
#               "DRout "+ str(self.DRin) + str([b*r for b,r in zip(self.Bout,self.RRout)]) + "\n"
