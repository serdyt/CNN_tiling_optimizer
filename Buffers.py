from math import ceil
from operator import add, mul

from LoopType import LoopType

class Buffers:
    # [RF, localBuffer, DRAM]
    
    def __init__(self):
#        self.levels = 3
        self.Bin = [0,0,0]
        self.Bkern = [0,0,0]
        self.Bout = [0,0,0]

        self.RRin = [1,1,1]
        self.RRkern = [1,1,1]
        self.RRout = [1,1,1]
                
    def totalSpace(self):
        return sum(self.Bin[:-1]) + sum(self.Bkern[:-1]) + sum(self.Bout[:-1])
                    
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
        # the level of current buffer / loop
            
        usedData = self.calcUsedData(arch+left, loop)
    
        #TODO correct levels
        # count the amount of the same loop types to the left (ommiting the HW core, which has 'u' in the 3rd element)
        # TODO: design normal data structures
        # there should be access to self.levels
    
        level = len([x.type for x in left if x.type == loop.type ])
        if level < 0 or level > 2:
            print "WTF", level, loop

        #TODO: add buffer merging rules    
        if loop.type == LoopType.kern:
            self.Bin[level] = ((usedData[LoopType.row.value] + usedData[LoopType.dx.value] - 1) *
                            (usedData[LoopType.col.value] + usedData[LoopType.dy.value] - 1) * usedData[LoopType.fm.value])
            self.RRin[level] = int(ceil( (float(loop.size) / usedData[LoopType.kern.value]) * 
                                    (usedData[LoopType.row.value] + usedData[LoopType.dx.value] - 1) *
                                    (usedData[LoopType.col.value] + usedData[LoopType.dy.value] - 1) /
                                    (usedData[LoopType.row.value] * usedData[LoopType.col.value])                                
                                    ))
        elif loop.type == LoopType.fm:
            self.Bout[level] = usedData[LoopType.row.value]*usedData[LoopType.col.value]*usedData[LoopType.kern.value]
            self.RRout[level] = int(ceil(2.0 * loop.size / usedData[LoopType.fm.value]))
        #TODO: add rules to merge row/col buffer when they follow each other
        elif (loop.type == LoopType.row or loop.type == LoopType.col):
            self.Bkern[level] = usedData[LoopType.kern.value]*usedData[LoopType.fm.value]*usedData[LoopType.dx.value]*usedData[LoopType.dy.value]
            # TODO: recheck this
            self.RRkern[level] =  int(ceil(loop.size*usedData[LoopType.col.value if loop.type==LoopType.row else LoopType.row.value] / 
                                    float(usedData[LoopType.row.value]*usedData[LoopType.col.value]) ))
        #TODO: check dx, dy
#        elif (loop[0] == LoopType.dx or loop[0] == LoopType.dy):
#            self.Bin[level] = ((usedData[LoopType.row.value] + usedData[LoopType.dx.value] - 1) *
#                            (usedData[LoopType.col.value] + usedData[LoopType.dy.value] - 1) * usedData[LoopType.fm.value])
#            self.RRin[level] += loop[1] * int(ceil( float(usedData[LoopType.row.value] + usedData[LoopType.dx.value] - 1) *
#                                    (usedData[LoopType.col.value] + usedData[LoopType.dy.value] - 1) /
#                                    (usedData[LoopType.row.value] * usedData[LoopType.col.value])                                
#                                    ))
#                                    
#            self.Bout[level] = usedData[LoopType.row.value]*usedData[LoopType.col.value]*usedData[LoopType.kern.value]
#            self.RRout[level] += int(ceil(2.0 * loop[1])) #something is wrong here
    
    def permBuff():
#        TODO: add possible shifts
          
#    def shiftLeft(self):
#        for x in [self.Bin, self.RRin, self.Bkern, self.RRkern, self.Bout, self.RRout]:
#            x.sort(key=lambda v: v!= 0)            
           
    def __str__ (self):
        return "Bin " + str(self.Bin) + str(self.RRin) + " \n" + \
               "Bkern " + str(self.Bkern) + str(self.RRkern) + " \n" + \
               "Bout " + str(self.Bout) + str(self.RRout)
