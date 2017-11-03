from LoopType import LoopType, RowCol

class Layer(object):
    
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
        
        self.MAC = self.dx * self.dy * self.X * self.Y * self.fm * self.kern

    def getMaxLoopSize(self, loopType):
        return {LoopType.fm:self.fm,
                LoopType.kern:self.kern,
                LoopType.rowcol:RowCol(self.Y, self.X),
                LoopType.dx:self.dx,
                LoopType.dy:self.dy                
                }[loopType]

    def getMaxLoopSizeByIndex(self, index):
        return {0:RowCol(self.Y, self.X),
                1:self.kern,
                2:self.fm,
                3:self.dx,
                4:self.dy                
                }[index]
                
class AlexNet1(Layer):
    def __init__(self): super(AlexNet1, self).__init__(  name="AlexNet1", X=227, Y=227, fm=3, kern=46, dx=11, dy=11, stride=1)
        
class AlexNet2(Layer):
    def __init__(self): super(AlexNet2, self).__init__(  name="AlexNet2", X=55, Y=55, fm=48, kern=256, dx=5, dy=5, stride=1)
        
class AlexNet3(Layer):
        def __init__(self): super(AlexNet3, self).__init__(  name="AlexNet3", X=27, Y=27, fm=128, kern=384, dx=3, dy=3, stride=1)