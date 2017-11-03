from LoopType import LoopType, Loop, RowCol, Pragma
from Buffers import Buffers
from math import floor, ceil
from operator import mul



class HwTemplate(object):
    def __init__(self, name, archU=[], archP=[], skip = []):
        self.archU = archU
        self.archP = archP
        self.skipLoop = skip
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
            buff.calcBuffSizeRR(left, loop, self.archU, self.archP, self.skipLoop)
                    
        return buff
        
    def calcEnergy(self, buff, energyModel, layer):
        res = 0
            
        for rr, data in zip([buff.RRin, buff.RRkern], [layer.dataIn, layer.dataKern]):
            a = rr[2]
            b = rr[1]
            c = rr[0]

            res += a * energyModel.DRAM * data
            if rr[1] != 1:
                res += a * b * energyModel.buff * data
            res += a*b*c*energyModel.RF * data      
                
        a = buff.RRout[2]
        b = buff.RRout[1]
        c = buff.RRout[0]
        data = layer.dataOut        
        
        res += (2*a - 1) * energyModel.DRAM * data
        if rr[1] != 1:
            res += a*(2*b - 1)*energyModel.buff * data
        res += a*b*(2*c - 1)*energyModel.RF * data
            
        # the amount of MAC operations
        res += layer.MAC * energyModel.ALU
        
        return res
            
'''
Example tempate classes
'''    

#Make a variation without kern in archP
class DiaNNao(HwTemplate):
    def __init__(self):
        super(DiaNNao, self).__init__("DiaNNao", [Loop(LoopType.fm, 16, Pragma.u), Loop(LoopType.kern, 16, Pragma.u)],
              [Loop(LoopType.fm, 1, Pragma.p), Loop(LoopType.kern, 1, Pragma.p), Loop(LoopType.dx, 1, Pragma.p), Loop(LoopType.dy, 1, Pragma.p)]
              )
    
    # This is not correct at all       
    def calcBuffers(self, tiledTemplate, layer):
        superBuff = super(DiaNNao, self).calcBuffers(tiledTemplate, layer)
        
        buff = Buffers()
        
        #Bout[0] = fm
        buff.Bout[0] = self.archU[0]
        #RRout[0] = fm1 / fm0
        buff.RRout[0] = ceil(self.archP[0] / float(self.archU[0]))
        
        #Binp[0] = fm1
        buff.Bin[0] = self.archP[0]
        #RRinp[0] = kern1 / kern0
        buff.RRin[0] = ceil(self.archP[1] / float(self.archU[1]))
        
        #Bkern[0] = kern1 * fm1
        buff.Bkern[0] = self.archP[1] * self.archP[0]
        #RRkern[0] = 1
        
        #Bout[1] = kern1
        buff.RRout[1] = self.archP[1]
        #RRout[1] = dx * dy
        buff.RRout[1] = self.archP[2] * self.archP[3]
        
        if (superBuff.Bin[1] == 0):
            buff.RRin[2] = 1
        
        
        #Bin[1] = fm1 * dx * dy
        buff.Bin[1] = self.archP[2] * self.archP[3] * self.archP[0]
        #RRin[1] = 
        
        buff.concatBuff(superBuff)        
        
        return buff


## TODO: make a version with unique Bin for all kernels (PUs)
class DNNweaver(HwTemplate):
    def __init__(self):
#        include fm?
        super(DNNweaver, self).__init__("DNNweaver", [Loop(LoopType.rowcol, RowCol(1,1), Pragma.u), Loop(LoopType.kern, 1, Pragma.u)],
                                     [Loop(LoopType.dx, 1, Pragma.p), Loop(LoopType.dy, 1, Pragma.p), Loop(LoopType.rowcol, RowCol(1,1), Pragma.p)],
                                     skip = [Loop(LoopType.kern, 1, Pragma.s), Loop(LoopType.fm, 1, Pragma.s)])
                                     
    def MACRestrictions(self, layer):
        self.minArchUvalues = [layer.dx, 1]
        self.maxArchUvalues = [layer.X, layer.kern]  

    def setALU(self, ALUlist):
        for i,MAC in enumerate(ALUlist):
            if self.archU[i].type == LoopType.rowcol:
                self.archU[i].size.col = MAC
                self.archU[i].size.row = 1
            else:
                self.archU[i].size = MAC                                     
                
    def getLoopSizeByType(self, Ltype ,tiledTemplate):
        for t in tiledTemplate:
            if Ltype == t.type:
                return t.size
        
        raise Exception("No such loop type for some reason, check DNNweawer template")
                                     
    def calcBuffers(self, tiledTemplate, layer):
        superBuff = super(DNNweaver, self).calcBuffers(tiledTemplate, layer)
        
        buff = Buffers()
        
#        col1 = self.getLoopSizeByType(LoopType.rowcol, tiledTemplate).col
#        row1 = self.getLoopSizeByType(LoopType.rowcol, tiledTemplate).row
        col1 = self.archP[2].size.col
        
#       Bout[0] = ceil(col1/col0) * kern0
        buff.Bout[0] = ceil(col1 / float(self.archU[0].size.col)) * self.archU[1].size
#       dx*dy
        buff.RRout[0] = self.archP[0].size * self.archP[1].size
        
#       Bin[0] = col0 * kernel
        buff.Bin[0] = col1 * self.archU[1].size
#       RRin[0] = dx #dx (not reused between kernels according to the architecture)
        buff.RRin[0] = self.archP[0].size
        
#       RRin[1] = dy
        buff.RRin[1] = self.archP[1].size
        
#       Bkern[0] = kern0 * dx * dy
        buff.Bkern[0] = self.archU[1].size * self.archP[0].size * self.archP[1].size
#       RRkern[0] = ceil(col1 / col0) * row0
        buff.RRkern[0] = int(ceil(col1 / float(self.archU[0].size.col))) * self.archP[2].size.row    
        
        buff.concatBuff(superBuff)
        
        return buff


class CNP (HwTemplate):
    def __init__(self):
        super(CNP, self).__init__(  "CNP",
                                    [Loop(LoopType.dx, 1, Pragma.u), Loop(LoopType.dy, 1, Pragma.u)], 
                                    [Loop(LoopType.rowcol, RowCol(1,1), Pragma.p)],
                                    skip = [Loop(LoopType.fm, 1, Pragma.s)]
                                 )

    def calcBuffers(self, tiledTemplate, layer):
        superBuff = super(CNP, self).calcBuffers(tiledTemplate, layer)
         
        buff = Buffers()
        
        # Bkern[0] = dx * dy
        buff.Bkern[0] = self.archU[0].size * self.archU[1].size
        # RRkern[0] = row*(col+dx-1)
        buff.RRkern[0] = self.archP[0].size.row * self.archP[0].size.col
        
        # Bout[0] = dx * dy
        buff.Bout[0] = self.archU[0].size * self.archU[1].size
        # RRout[0] = dx
        buff.RRout[0] = self.archU[0].size
        
        # Bout[1] = col * (dy-1)
        buff.Bout[0] += self.archP[0].size.col * (self.archU[1].size - 1)
        # RRout[1] = dy
        buff.RRout[0] *= self.archU[1].size 
                       
        buff.concatBuff(superBuff)        
        
        return buff


class CNPfm (HwTemplate):
    def __init__(self):
        super(CNPfm, self).__init__(  "CNPfm",
                                    [Loop(LoopType.dx, 1, Pragma.u), Loop(LoopType.dy, 1, Pragma.u), Loop(LoopType.fm, 1, Pragma.u)], 
                                    [Loop(LoopType.rowcol, RowCol(1,1), Pragma.p)],
                                     skip = [Loop(LoopType.fm, 1, Pragma.s)]
                                 )
                                                                  
    def calcBuffers(self, tiledTemplate, layer):
        superBuff = super(CNPfm, self).calcBuffers(tiledTemplate, layer)

        buff = Buffers()
        
        buff.Bin[0] = 1
        buff.RRin[0] = 1
        
        # Bkern[0] = dx * dy * fm
        buff.Bkern[0] = self.archU[0].size * self.archU[1].size * self.archU[2].size
        # RRkern[0] = row*(col+dx-1)
        buff.RRkern[0] = self.archP[0].size.row * self.archP[0].size.col
        
        # Bout[0] = dx * dy
        buff.Bout[0] = self.archU[0].size * self.archU[1].size
        # RRout[0] = dx 
        buff.RRout[0] = self.archU[0].size
        
        # Bout[1] = col * (dy-1)
        buff.Bout[0] += self.archP[0].size.col * (self.archU[1].size - 1)
        # RRout[1] = dy
        buff.RRout[0] *= self.archU[1].size 
        
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
                
    def calcEnergy(self, buff, energyModel, layer):
        res = 0        
       
        for rr, data in zip([buff.RRin, buff.RRkern], [layer.dataIn, layer.dataKern]):
            a = rr[2]
            b = rr[1]
            c = rr[0]
            
            res += a * energyModel.DRAM * data
            if rr[1] != 1:
                res += a * b * energyModel.buff * data
            res += a*b*c*energyModel.RF * data
                                
        a = buff.RRout[2]
        b = buff.RRout[1]
        c = buff.RRout[0]
        data = layer.dataOut
        
        res += (2*a - 1) * energyModel.DRAM * data
        if rr[1] != 1:
            res += a*(2*b - 1)*energyModel.buff * data
#       2*c-1 is already in RRin[0]
        res += a*b*c*energyModel.RF * data
            
        # the amount of MAC operations
        res += layer.MAC * energyModel.ALU
        
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