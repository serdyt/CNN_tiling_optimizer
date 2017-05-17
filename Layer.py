from LoopType import LoopType

class Layer:
    
    def __init__ (self, name="AlexNet1", X=227, Y=227, fm=3, kern=46, dx=3, dy=3, stride=1):
        self.dx = dx
        self.dy = dy
        self.name = name
        self.X = X
        self.Y = Y
        self.fm = fm
        self.kern = kern
        self.stride = stride
        
        self.dataIn = self.X * self.Y * self.fm
        self.dataKern = self.dx * self.dy * self.kern * self.fm
        self.dataOut = (self.X - self.dx + 1) * (self.Y - self.dy + 1) * self.fm

    def getMaxLoopSize(self, loopType):
        return {LoopType.fm:self.fm,
                LoopType.kern:self.kern,
                LoopType.col:self.X,
                LoopType.row:self.Y,
                LoopType.dx:self.dx,
                LoopType.dy:self.dy                
                }[loopType]

    def getMaxLoopSizeByIndex(self, index):
        return {0:self.fm,
                1:self.kern,
                2:self.X,
                3:self.Y,
                4:self.dx,
                5:self.dy                
                }[index]