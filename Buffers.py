from math import ceil
from operator import add, mul

from LoopType import LoopType, RowCol
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
        
        self.usedData = []
#                
    def totalSpace(self):
        return sum(self.Bin) + sum(self.Bkern) + sum(self.Bout)
                    
    def initUsedData(self, left,loop):
        #scan all the data to the left to find the amount of each data type being used
        self.usedData = [1]*len(LoopType)
        for i in LoopType:
            if (i == LoopType.rowcol):
                self.usedData[i.value] = RowCol(1,1)
        
        for i in LoopType:
            rawData = [ x for x in left if x.type == i ]

            if len(rawData) > 0:
                if (i == LoopType.rowcol):
                    self.usedData[i.value].row = rawData[-1].size.row    
                    self.usedData[i.value].col = rawData[-1].size.col
                else:
                    self.usedData[i.value] = rawData[-1].size

        return self.usedData
        
    def concatBuff(self, buff):
        self.Bin = map(max, self.Bin, buff.Bin)
        self.Bout = map(max, self.Bout, buff.Bout)
        self.Bkern = map(max, self.Bkern, buff.Bkern)
        
        self.RRin = map(mul, self.RRin, buff.RRin)
        self.RRout = map(mul, self.RRout, buff.RRout)
        self.RRkern = map(mul, self.RRkern, buff.RRkern)
            
    def calcBuffSizeRR(self, left, loop, archU, archP, skip):
        # TODO: Move RRout*2 to Buffers (it is in calcEnergy now)
    
        if self.usedData == []:
            self.initUsedData(archU+archP+left, loop)
        else:
            if (left[-1].type == LoopType.rowcol):
                self.usedData[left[-1].type.value].row = left[-1].size.row
                self.usedData[left[-1].type.value].col = left[-1].size.col
            else:                    
                self.usedData[left[-1].type.value] = left[-1].size
                    
        # avoid unnececary computations
        if loop.size == 1:
            return
            
        # avoid unnececary computations
        # this would generate a bufer with RR=1
        if loop.size == self.usedData[loop.type.value]:
            return
    
        level = len([x.type for x in skip + archP + left if x.type == loop.type ])
        if level < 0 and level > 2:
            print "WTF", level, loop
            raise Exception('Too many loop levels') 
        
        # calc buffer size only for local buffers
        if level < 2:
            if loop.type == LoopType.kern:
                self.Bin[level] = kernBuff( self.usedData[LoopType.rowcol.value].row, 
                                            self.usedData[LoopType.dx.value], 
                                            self.usedData[LoopType.rowcol.value].col, 
                                            self.usedData[LoopType.dy.value], 
                                            self.usedData[LoopType.fm.value])
                                            
            elif loop.type == LoopType.fm:
                self.Bout[level] = fmBuff(  self.usedData[LoopType.rowcol.value].row,
                                            self.usedData[LoopType.rowcol.value].col,
                                            self.usedData[LoopType.kern.value])
                                            
            elif loop.type == LoopType.rowcol:
                self.Bkern[level] = rowcolBuff( self.usedData[LoopType.kern.value],
                                                self.usedData[LoopType.fm.value],
                                                self.usedData[LoopType.dx.value],
                                                self.usedData[LoopType.dy.value])
                                                
            elif loop.type == LoopType.dx or loop.type == LoopType.dy:
                self.Bin[level], self.Bout[level] = \
                                   dxdyBuff(self.usedData[LoopType.rowcol.value].col,
                                            self.usedData[LoopType.dx.value],
                                            self.usedData[LoopType.rowcol.value].row,
                                            self.usedData[LoopType.dy.value],                                
                                            self.usedData[LoopType.fm.value],
                                            self.usedData[LoopType.kern.value] )

        # calc RR
        if loop.type == LoopType.kern:
            self.RRin[level] *= kernRR( loop.size,
                                        self.usedData[LoopType.kern.value])

        elif loop.type == LoopType.fm:
            self.RRout[level] *= fmRR(  loop.size,
                                        self.usedData[LoopType.fm.value])

        elif loop.type == LoopType.rowcol:
            self.RRkern[level] *= rowRR(loop.size.row,
                                        self.usedData[LoopType.rowcol.value].row)

            self.RRkern[level] *= colRR(loop.size.col,
                                        self.usedData[LoopType.rowcol.value].col)
                       
        #TODO: check levels of dx dy
#        elif loop.type == LoopType.dx:
#            rrin, rrout = dxRR(loop.size,
#                                self.usedData[LoopType.dx.value],
#                                self.usedData[LoopType.rowcol.value].col)
#            self.RRin[level] *= rrin
#            self.RRout[level] *= rrout                                                 
#    
#        elif loop.type == LoopType.dy:
#            rrin, rrout = dyRR(loop.size,
#                               self.usedData[LoopType.dy.value],
#                               self.usedData[LoopType.rowcol.value].row)
#            self.RRin[level] *= rrin
#            self.RRout[level] *= rrout
           
    def __str__ (self):
        return "Bin " + str(self.Bin) + str(self.RRin) + str(reduce(mul, self.RRin)) + " \n" + \
               "Bkern " + str(self.Bkern) + str(self.RRkern) + str(reduce(mul, self.RRkern)) + " \n" + \
               "Bout " + str(self.Bout) + str(self.RRout) + str(reduce(mul, self.RRout))  #+ "\n" + \
#               "DRin " + str(self.DRout) + str([b*r for b,r in zip(self.Bin,self.RRin)]) + "\n" + \
#               "DRkern " + str(self.DRkern) + str([b*r for b,r in zip(self.Bkern,self.RRkern)]) + "\n" + \
#               "DRout "+ str(self.DRin) + str([b*r for b,r in zip(self.Bout,self.RRout)]) + "\n"
