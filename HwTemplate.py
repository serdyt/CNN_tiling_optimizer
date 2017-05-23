from enum import Enum

from LoopType import LoopType, Loop
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
        self.buff = self.calcBufferSize()
        
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
