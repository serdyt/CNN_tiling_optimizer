from math import ceil
from operator import add, mul

from LoopType import LoopType

class Buffers:
    # [RF, localBuffer, DRAM]
    
    def __init__(self):
#        self.levels = 3
        self.Bin = [0,0]
        self.Bkern = [0,0]
        self.Bout = [0,0]

        self.RRin = [1,1,1]
        self.RRkern = [1,1,1]
        self.RRout = [1,1,1]
        
        self.DRin = [0,0,0]
        self.DRkern = [0,0,0]
        self.DRout = [0,0,0]
                
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
        if level == 2:
            return
        if level < 0 and level > 2:
            print "WTF", level, loop
            raise Exception('Too many loop levels') 

        #TODO: add buffer merging rules    

        # calc buffer size only for local buffers
        if level < 2:
            if loop.type == LoopType.kern:
                self.Bin[level] = ((usedData[LoopType.row.value] + usedData[LoopType.dx.value] - 1) *
                            (usedData[LoopType.col.value] + usedData[LoopType.dy.value] - 1) * usedData[LoopType.fm.value])
            elif loop.type == LoopType.fm:
                self.Bout[level] = usedData[LoopType.row.value]*usedData[LoopType.col.value]*usedData[LoopType.kern.value]
            elif (loop.type == LoopType.row or loop.type == LoopType.col):
                self.Bkern[level] = usedData[LoopType.kern.value]*usedData[LoopType.fm.value]*usedData[LoopType.dx.value]*usedData[LoopType.dy.value]            
            elif loop.type == LoopType.dx:
                self.Bin[level] = ((usedData[LoopType.row.value] + usedData[LoopType.dx.value] - 1) *
                                (usedData[LoopType.col.value] + usedData[LoopType.dy.value] - 1) * usedData[LoopType.fm.value])
                self.Bout[level] = usedData[LoopType.row.value]*usedData[LoopType.col.value]*usedData[LoopType.kern.value]
            elif loop.type == LoopType.dy:
                self.Bin[level] = ((usedData[LoopType.row.value] + usedData[LoopType.dx.value] - 1) *
                                (usedData[LoopType.col.value] + usedData[LoopType.dy.value] - 1) * usedData[LoopType.fm.value])
                self.Bout[level] = usedData[LoopType.row.value]*usedData[LoopType.col.value]*usedData[LoopType.kern.value]
            
            
        # calc RR and DR
        if loop.type == LoopType.kern:
            tmp = int(ceil( (float(loop.size) / usedData[LoopType.kern.value]) ))
            self.RRin[level] *= tmp
            self.DRin[level] += tmp * self.Bin[level]
        elif loop.type == LoopType.fm:
            tmp = int(ceil( float(loop.size) / usedData[LoopType.fm.value]))
            self.RRout[level] *= tmp
            self.DRout[level] += 2 * tmp * self.Bout[level]
        elif (loop.type == LoopType.row or loop.type == LoopType.col):
            tmp = int(ceil(loop.size*usedData[loop.type.value] / 
                                    float(usedData[LoopType.row.value]*usedData[LoopType.col.value]) ))
            self.RRkern[level] *= tmp
            self.DRkern[level] += tmp * self.Bin[level]
                       
        #TODO: check levels of dx dy
        elif loop.type == LoopType.dx:
            tmp = ceil(float(loop.size)/ usedData[loop.type.value]) / (
                                        (usedData[LoopType.row.value] + loop.size - 1) * 
                                        (usedData[LoopType.col.value] + usedData[LoopType.dy.value] - 1) /
                                        (usedData[LoopType.row.value] * usedData[LoopType.col.value]))
            self.RRin[level] *= tmp
            self.DRin[level] += tmp * self.Bin[level]
            
            tmp = int(ceil(float(loop.size)/ usedData[loop.type.value]))
            self.RRout[level] *= tmp
            self.DRout[level] += 2 * tmp * self.Bout[level]
    
        elif loop.type == LoopType.dy:
            tmp = ceil(float(loop.size)/ usedData[loop.type.value]) / (
                                        (usedData[LoopType.row.value] + usedData[LoopType.dx.value] - 1) * 
                                        (usedData[LoopType.col.value] + loop.size - 1) /
                                        (usedData[LoopType.row.value] * usedData[LoopType.col.value]))
            self.RRin[level] *= tmp
            self.DRin[level] += tmp * self.Bin[level]
            
            tmp = int(ceil(float(loop.size)/ usedData[loop.type.value]))
            self.RRout[level] *= tmp
            self.DRout[level] += 2 * tmp * self.Bout[level]


#    def mergeBuff(self, hwRestrictions):
#        for b,rr in zip([self.Bin, self.RRin, self.Bkern], [self.RRkern, self.Bout, self.RRout]):
#            if 0 in b:
                
          
#    def shiftLeft(self):
#        for x in [self.Bin, self.RRin, self.Bkern, self.RRkern, self.Bout, self.RRout]:
#            x.sort(key=lambda v: v!= 0)            
           
    def __str__ (self):
        return "Bin " + str(self.Bin) + str(self.RRin) + " \n" + \
               "Bkern " + str(self.Bkern) + str(self.RRkern) + " \n" + \
               "Bout " + str(self.Bout) + str(self.RRout)  + "\n" + \
               "DRin " + str(self.DRout) + str([b*r for b,r in zip(self.Bin,self.RRin)]) + "\n" + \
               "DRkern " + str(self.DRkern) + str([b*r for b,r in zip(self.Bkern,self.RRkern)]) + "\n" + \
               "DRout "+ str(self.DRin) + str([b*r for b,r in zip(self.Bout,self.RRout)]) + "\n"
