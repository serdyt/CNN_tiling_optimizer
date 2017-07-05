from enum import Enum

from LoopType import LoopType, Loop, RowCol
from Buffers import Buffers

class Pragma(Enum):
    n = 0 # normal
    u = 1 # unroll
    p = 2 # pipeline

class HwTemplate(object):
    def __init__(self, name, archU=[], archP=[]):
        self.archU = archU
        self.archP = archP
        self.name = name
#        self.buff = self.calcBufferSize()
        
        self.numU = len(self.archU)

        
    def calcALUamount(self):
        return reduce(lambda x, y: x*y, self.archU)
                
    def calcBufferSize(self):
        return Buffers()
        
    def setALU(self, ALUlist):
        for i,MAC in enumerate(ALUlist):
            self.archU[i].size = MAC
            
    
'''
Example tempate classes
'''    
        
class DiaNNao(HwTemplate):
    def __init__(self):
        super(DiaNNao, self).__init__("DiaNNao", [Loop(LoopType.fm, 16, Pragma.u), Loop(LoopType.kern, 16, Pragma.u)],
              [Loop(LoopType.fm, 1, Pragma.p)]
              )


# TODO: there should be RR and energy computations here
class DNNweaver(HwTemplate):
    def __init__(self):
#        include fm?
        super(DNNweaver, self).__init__("DNNweaver", [Loop(LoopType.col, 1, Pragma.u), Loop(LoopType.kern, 1, Pragma.u)],
                                     [Loop(LoopType.dx, 1, Pragma.p), Loop(LoopType.col, 1, Pragma.p)])
                                     
    def calcBufferSize(self):
        buff = Buffers()
#       col1*kern0
        buff.Bout[0] = self.archP[1].size * self.archU[1].size
#       2 * dx - each output element is reused dx times
        buff.RRout[0] = 2 * self.archP[0].size
#       col0 for each kernel
        buff.Bin[0] = self.archU[0].size * self.archU[1].size
        return buff

#d= DiaNNao()
#d = DNNweaver()
#DiaNNao = HwTemplate("DianNao", [Loop(LoopType.fm, 16, Pragma.u), Loop(LoopType.kern, 16, Pragma.u)])

class CNP (HwTemplate):
    def __init__(self):
        super(CNP, self).__init__(  "CNP",
                                    [Loop(LoopType.dx, 1, Pragma.u), Loop(LoopType.dy, 1, Pragma.u)], 
                                    [Loop(LoopType.rowcol, RowCol(1,1), Pragma.p)]
                                 )
                                 
    def calcBufferSize(self):
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
        
        return buff
        
        
class Origami (HwTemplate):
    def __init__(self):
        super(Origami, self).__init__(  "Origami",
                                    [   Loop(LoopType.fm, 1, Pragma.u), 
                                        Loop(LoopType.dx, 1, Pragma.u), 
                                        Loop(LoopType.dy, 1, Pragma.u),
                                        Loop(LoopType.kern, 1, Pragma.u)], 
                                    [Loop(LoopType.rowcol, RowCol(1,1), Pragma.p)]
                                 )
                                 
    def calcBufferSize(self):
        buff = Buffers()
        
        # Bkern[0] = dx*dy*kern
        buff.Bkern[0] = self.archU[1].size * self.archU[2].size * self.archU[3].size 
        # RRkern[0] = row*col
        buff.RRkern[0] = self.archP[0].size.row * self.archP[0].size.col
        
        # Binp[0] = fm*dx*dy
        buff.Binp[0] = self.archU[0].size * self.archU[1].size * self.archU[2].size 
        # RRinp[0] = dy
        buff.Binp[0] = self.archU[2].size 
        
        # Binp[1] = fm*row*dx
        buff.Bin[1] = self.archU[0].size * self.archP[0].size.row *  self.archU[1].size 
        # RRinp[1] = dx
        buff.RRin[1] =  self.archU[1].size 
        
        return buff