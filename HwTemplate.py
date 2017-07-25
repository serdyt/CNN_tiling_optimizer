from enum import Enum

from LoopType import LoopType, Loop, RowCol
from Buffers import Buffers
from math import floor, ceil
from operator import mul

class Pragma(Enum):
    n = 0 # normal
    u = 1 # unroll
    p = 2 # pipeline

class HwTemplate(object):
    def __init__(self, name, archU=[], archP=[]):
        self.archU = archU
        self.archP = archP
        self.name = name
        
        self.numU = len(self.archU)
        
        self.minArchUvalues = []
        self.maxArchUvalues = []
        
    def calcALUamount(self):
        return reduce(lambda x, y: x*y, self.archU)
                        
    def setALU(self, ALUlist):
        for i,MAC in enumerate(ALUlist):
            self.archU[i].size = MAC
            
    def MACRestrictions(self, layer):
        self.maxArchUvalues = [layer.getMaxLoopSize(x.type) for x in self.archU]
        self.minArchUvalues = [layer.getMaxLoopSize(x.type) if x.type == LoopType.dx or x.type == LoopType.dy else 1 for x in self.archU]
        
        for i,l in enumerate(self.archU):
            if l.type == LoopType.rowcol:
                self.maxArchUvalues[i] = reduce(mul, self.maxArchUvalues[i])
        
    def ALUperm(self, depth, MAC, maxMAC):
        if depth == 1:
            if (MAC < self.minArchUvalues[-1]):
                pass
            else:
                yield [min(MAC, self.maxArchUvalues[-1])]
        else:
            for i in self.divisorsFloor(maxMAC, self.minArchUvalues[len(self.minArchUvalues) - depth], self.maxArchUvalues[len(self.maxArchUvalues) - depth]):
                for perm in self.ALUperm(depth-1, int(floor(MAC/float(i))), maxMAC):
                    yield [i] + perm
                    
    def divisorsFloor(self, x, mini, maxi):
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
        
    def concatBuff(self, buff):
        self.Bin = map(max, self.Bin, buff.Bin)
        self.Bout = map(max, self.Bout, buff.Bout)
        self.Bkern = map(max, self.Bkern, buff.Bkern)
        
        self.RRin = map(mul, self.RRin, buff.RRin)
        self.RRout = map(mul, self.RRout, buff.RRout)
        self.RRkern = map(mul, self.RRkern, buff.RRkern)
        
    def calcBuffers(self, tiledTemplate, layer):
        buff = Buffers()
        for index in xrange(len(tiledTemplate)):
            loop = tiledTemplate[index]
            left = tiledTemplate[:index]
            buff.calcBuffSizeRR(left, loop, self.archU, self.archP)
                    
        return buff
        
    def calcEnergy(self, buff, energyModel):
        res = 0        
            
        for rr in [buff.RRin, buff.RRkern]:
            a = rr[2]
            b = rr[1]
            c = rr[0]
            
            res += (2*a - 1) * energyModel.DRAM
            if rr[1] != 1:
                res += a*(2*b - 1)*energyModel.buff 
            if rr[0] != 1:
                res += a*b*(2*c - 1)*energyModel.RF
                
        a = buff.RRout[2]
        b = buff.RRout[1]
        c = buff.RRout[0]
        
        res += a * energyModel.DRAM
        if rr[1] != 1:
            res += a * b * energyModel.buff 
        if rr[0] != 1:
            res += a*b*c*energyModel.RF
        
        return res
    
'''
Example tempate classes
'''    
        
class DiaNNao(HwTemplate):
    def __init__(self):
        super(DiaNNao, self).__init__("DiaNNao", [Loop(LoopType.fm, 16, Pragma.u), Loop(LoopType.kern, 16, Pragma.u)],
              [Loop(LoopType.fm, 1, Pragma.p)]
              )


## TODO: there should be RR and energy computations here
#class DNNweaver(HwTemplate):
#    def __init__(self):
##        include fm?
#        super(DNNweaver, self).__init__("DNNweaver", [Loop(LoopType.col, 1, Pragma.u), Loop(LoopType.kern, 1, Pragma.u)],
#                                     [Loop(LoopType.dx, 1, Pragma.p), Loop(LoopType.col, 1, Pragma.p)])
#                                     
#    def calcBuffers(self):
#        buff = Buffers()
##       col1*kern0
#        buff.Bout[0] = self.archP[1].size * self.archU[1].size
##       2 * dx - each output element is reused dx times
#        buff.RRout[0] = 2 * self.archP[0].size
##       col0 for each kernel
#        buff.Bin[0] = self.archU[0].size * self.archU[1].size
#        return buff

#d= DiaNNao()
#d = DNNweaver()
#DiaNNao = HwTemplate("DianNao", [Loop(LoopType.fm, 16, Pragma.u), Loop(LoopType.kern, 16, Pragma.u)])

class CNP (HwTemplate):
    def __init__(self):
        super(CNP, self).__init__(  "CNP",
                                    [Loop(LoopType.dx, 1, Pragma.u), Loop(LoopType.dy, 1, Pragma.u)], 
                                    [Loop(LoopType.rowcol, RowCol(1,1), Pragma.p)]
                                 )
                                 
    def calcBuffers(self, tiledTemplate, layer):
        superBuff = super(CNP, self).calcBuffers(tiledTemplate, layer)
         
        buff = Buffers()
        
        buff.Bin[0] = 1
        buff.RRin[0] = 1
        
        # Bkern[0] = dx * dy
        buff.Bkern[0] = self.archU[0].size * self.archU[1].size
        # RRkern[0] = row*(col+dx-1)
        buff.RRkern[0] = self.archP[0].size.row * self.archP[0].size.col
        
        # Bout[0] = dx * dy
        buff.Bout[0] = self.archU[0].size * self.archU[1].size
        # RRout[0] = dx
        buff.RRout[0] = self.archU[0].size
        
        # Bout[1] = col * (dy-1)
        buff.Bout[1] = self.archP[0].size.col * (self.archU[1].size - 1)
        # RRout[1] = dy
        buff.RRout[1] = self.archU[1].size 
        
        buff.concatBuff(superBuff)        
        
        return buff
        
class Eyeriss (HwTemplate):
    def __init__(self):
        super(Eyeriss, self).__init__(  "Eyeriss",
                                    [Loop(LoopType.dy, 1, Pragma.u), Loop(LoopType.rowcol, RowCol(1,1), Pragma.u), Loop(LoopType.kern, 1, Pragma.u)], 
                                    [Loop(LoopType.rowcol, RowCol(1,1), Pragma.p), Loop(LoopType.dx, 1, Pragma.p), Loop(LoopType.fm, 1, Pragma.p)],
                                 )
    def calcBuffers(self, tiledTemplate, layer):
        superBuff = super(Eyeriss, self).calcBuffers(tiledTemplate, layer)
        
        buff = Buffers()        
        
        #Bin[0] = dx*fm * dy*(col+dx-1) * kern
        buff.Bin[0] = self.archP[1].size * self.archP[2].size * self.archU[0].size * (self.archU[1].size.col + self.archP[1].size - 1) * self.archU[2].size
        #RRin[0] = dx
        buff.RRin[0] = self.archP[1].size
        
#        #Bkern[0] = dx*fm * dy*(col+dx-1) * kern
        buff.Bkern[0] = self.archP[1].size * self.archP[2].size * self.archU[0].size * (self.archU[1].size.col + self.archP[1].size - 1) * self.archU[2].size
#        #RRkern[0] = col + dx - 1
#        buff.RRkern[0] = self.archU[1].size.col + self.archP[1].size - 1
        #RRkern[0] = col[0]
        buff.RRkern[0] = self.archP[0].size.col * int(ceil(self.archP[0].size.row / float(self.archU[1].size.row)))
        
        #Bout[0] = dy * kern * (col + dx - 1)
        buff.Bout[0] = self.archU[0].size * self.archU[2].size * (self.archU[1].size.col + self.archP[1].size - 1)
        #RRout[0] = dy * (dx * 2 - 1) * fm
        buff.RRout[0] = self.archU[0].size * self.archP[2].size * (self.archP[1].size * 2 - 1) + self.archU[0].size * 2
        
        
        #Bout[1] = row*col*kern
        buff.Bout[1] = self.archP[0].size.row * self.archP[0].size.col * self.archU[2].size
#        
#        print buff
#        print superBuff

        buff.concatBuff(superBuff) 

        return buff
        
    def MACRestrictions(self, layer):
        self.minArchUvalues = [layer.dy, 1, 1]
        self.maxArchUvalues = [layer.dy, layer.X, layer.kern]

    def setALU(self, ALUlist):
        for i,MAC in enumerate(ALUlist):
            if self.archU[i].type == LoopType.rowcol:
                self.archU[i].size.col = 1
                self.archU[i].size.row = MAC
            else:
                self.archU[i].size = MAC
                
    def calcEnergy(self, buff, energyModel):
        res = 0        
            
        for rr in [buff.RRin, buff.RRkern]:
            a = rr[2]
            b = rr[1]
            c = rr[0]
            
            res += (2*a - 1) * energyModel.DRAM
            if rr[1] != 1:
                res += a*(2*b - 1)*energyModel.buff 
            if rr[0] != 1:
#           2*c-1 is already in RRin[0]
                res += a*b*c*energyModel.RF
                
        a = buff.RRout[2]
        b = buff.RRout[1]
        c = buff.RRout[0]
        
        res += a * energyModel.DRAM
        if rr[1] != 1:
            res += a * b * energyModel.buff 
        if rr[0] != 1:
            res += a*b*c*energyModel.RF
        
        return res
        
        
        
#class Origami (HwTemplate):
#    def __init__(self):
#        super(Origami, self).__init__(  "Origami",
#                                    [   Loop(LoopType.fm, 1, Pragma.u), 
#                                        Loop(LoopType.dx, 1, Pragma.u), 
#                                        Loop(LoopType.dy, 1, Pragma.u),
#                                        Loop(LoopType.kern, 1, Pragma.u)], 
#                                    [Loop(LoopType.rowcol, RowCol(1,1), Pragma.p)]
#                                 )
#                                 
#    def calcBuffers(self):
#        buff = Buffers()
#        
#        # Bkern[0] = dx*dy*kern
#        buff.Bkern[0] = self.archU[1].size * self.archU[2].size * self.archU[3].size 
#        # RRkern[0] = row*col
#        buff.RRkern[0] = self.archP[0].size.row * self.archP[0].size.col
#        
#        # Binp[0] = fm*dx*dy
#        buff.Binp[0] = self.archU[0].size * self.archU[1].size * self.archU[2].size 
#        # RRinp[0] = dy
#        buff.Binp[0] = self.archU[2].size 
#        
#        # Binp[1] = fm*row*dx
#        buff.Bin[1] = self.archU[0].size * self.archP[0].size.row *  self.archU[1].size 
#        # RRinp[1] = dx
#        buff.RRin[1] =  self.archU[1].size 
#        
#        return buff