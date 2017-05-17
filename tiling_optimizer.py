#import caffe_pb2
#from google.protobuf.text_format import Merge
#
#net = caffe_pb2.NetParameter()
#Merge((open("Alexnet.prototxt",'r').read()), net)

import sys
from enum import Enum
import csv
import itertools
import math
from operator import mul

#netFileName = sys.argv[1]

# Arguments list:
# Caffe CNN description file name
# Log file name


class Loop(Enum):
    fm = 0
    kern = 1
    col = 2
    row = 3
    dx = 4
    dy = 5

class IncTileResult(Enum):
    LoopIncremented = 1
    LoopOverflow = 2

class Layer:
#    InputX = 227
#    InputY = 227
#    InputFM = 3
#    Kernels = 46
#    Dx = 3
#    Dy = 3
#    stride = 1
#    name = "AlexNet1"
    
    def __init__ (self, name="AlexNet1", X=227, Y=227, fm=3, kern=46, dx=3, dy=3, stride=1):
        self.dx = dx
        self.dy = dy
        self.name = name
        self.X = X
        self.Y = Y
        self.fm = fm
        self.kern = kern
        self.stride = stride

    def getByLoop(self, index):
        return {0:self.fm,
                1:self.kern,
                2:self.X,
                3:self.Y,
                4:self.dx,
                5:self.dy                
                }[index]

class Energy:
    def __init__(self):
        self.DRAM = 200
        self.buff = 6
        self.RF = 1
        self.ALU = 1

class Template:
                                        # ([loop name][value])
    microArchitecture = []   # microArchitecture is a fixed part of a string
    tiles = []           # tiles is the rest of the loop string - it will be permuted
#    size = []
    levels = 3                          # maximum number of tiling of one loop
    name = ""
        
    def __init__(self, name, microArchitecture):
        self.microArchitecture = microArchitecture
        self.name = name

class HWrestrictions:
# Zynq Z-7007S
#    memory = (36000, 50)                # (size of a block, number of blocks)
    
    def __init__(self, memory=((36000, 50)) , MAC = 66):
        self.memory = memory
        self.MAC = MAC
        
def printTile (tile, i=2):
    if i == 1:
        print  [(elm[0].name, elm[1]) for elm in tile]
    elif i == 2:
        print  [elm[0].name for elm in tile]
    else:
        print tile
    
def printBuff (buff):
    print "Bin", buff.Bin, buff.RRin
    print "Bkern", buff.Bkern, buff.RRkern
    print "Bout", buff.Bout, buff.RRout

class Buffers:
    # [RF, localBuffer, DRAM]
    
    def __init__(self):
#        self.levels = 3
        self.Bin = [0,0,0]
        self.Bkern = [0,0,0]
        self.Bout = [0,0,0]

        self.RRin = [0,0,0]
        self.RRkern = [0,0,0]
        self.RRout = [0,0,0]
        
    def totalSpace(self):
        return sum(self.Bin[:-1]) + sum(self.Bkern[:-1]) + sum(self.Bout[:-1])
        
    def buffFitMem(self, hardware):
        if self.totalSpace() > hardware.memory[0]*hardware.memory[1]:
            return False
        else:
            return True

def maxLoop(a,b):
    if a.value >= b.value:
        return a
    else:
        return b

def calcUsedData(left,loop):
    #scan all the data to the left to find the amount of each data type being used
    usedData = [1]*len(Loop)
    for i in Loop:
        rawData = [ x for x in left if x[0] == i ]
        usedData[i.value] = 1 if len(rawData) == 0 else sum([x[1] for x in rawData])
    return usedData
    
def formulas(left, loop, right, buff):
    # the level of current buffer / loop

    usedData = calcUsedData(left, loop)

    #TODO correct levels
    level = len([ x for x in left if x[0] == loop[0] ])
    #TODO: add buffer merging rules    
    if loop[0] == Loop.kern:
        buff.Bin[level] = ((usedData[Loop.row.value] + usedData[Loop.dx.value] - 1) *
                        (usedData[Loop.col.value] + usedData[Loop.dy.value] - 1) * usedData[Loop.fm.value])
        buff.RRin[level] += int(math.ceil( (float(loop[1]) / usedData[Loop.kern.value]) * 
                                (usedData[Loop.row.value] + usedData[Loop.dx.value] - 1) *
                                (usedData[Loop.col.value] + usedData[Loop.dy.value] - 1) /
                                (usedData[Loop.row.value] * usedData[Loop.col.value])                                
                                ))
    elif loop[0] == Loop.fm:
        buff.Bout[level] = usedData[Loop.row.value]*usedData[Loop.col.value]*usedData[Loop.kern.value]
        buff.RRout[level] += int(math.ceil(2.0 * loop[1] / usedData[Loop.fm.value]))
    #TODO: add rules to merge row/col buffer when they follow each other
    elif (loop[0] == Loop.row or loop[0] == Loop.col):
        buff.Bkern[level] = usedData[Loop.kern.value]*usedData[Loop.fm.value]*usedData[Loop.dx.value]*usedData[Loop.dy.value]
        # TODO: recheck this
        buff.RRkern[level] +=  int(math.ceil(loop[1]*usedData[Loop.col.value if loop[0]==Loop.row else Loop.row.value] / 
                                float(usedData[Loop.row.value]*usedData[Loop.col.value]) ))
    #TODO: check dx, dy
    elif (loop[0] == Loop.dx or loop[0] == Loop.dy):
        buff.Bin[level] = ((usedData[Loop.row.value] + usedData[Loop.dx.value] - 1) *
                        (usedData[Loop.col.value] + usedData[Loop.dy.value] - 1) * usedData[Loop.fm.value])
        buff.RRin[level] += loop[1] * int(math.ceil( float(usedData[Loop.row.value] + usedData[Loop.dx.value] - 1) *
                                (usedData[Loop.col.value] + usedData[Loop.dy.value] - 1) /
                                (usedData[Loop.row.value] * usedData[Loop.col.value])                                
                                ))
                                
        buff.Bout[level] = usedData[Loop.row.value]*usedData[Loop.col.value]*usedData[Loop.kern.value]
        buff.RRout[level] += int(math.ceil(2.0 * loop[1])) #something is wrong here
  
    return buff


def calcBuffers(tiledTemplate):
    buff = Buffers()
    for index in xrange(len(tiledTemplate)):
        left = tiledTemplate[:index]
        loop = tiledTemplate[index]
        right = tiledTemplate[index:]
        
        buff = formulas(left, loop, right, buff)
    
    return buff

def ALUFitMAC(template, MAC):
    return not(reduce(mul, [x[1] for x in template if len(x) == 3], 1) > MAC)
    
def checkRestrictions(template, buff, restrictions):
#    if buff.doesBuffFits(restrictions):
#        return False
#    # product of all unrolled loop = MACs
#    if reduce(mul, [x[1] for x in template if len(x) == 3], 1) > restrictions.MAC:
#        return False
#    return True
    
    return (buff.buffFitMem(restrictions), ALUFitMAC(template, restrictions.MAC))
        
        
def incrementTiling(tiledTemplate, lsliceIndex, maxLoopValue):
    # if current group of loops has iterated through all the variants - reset them to 1,1,max
        # and try to increment the next loop
    if tiledTemplate[lsliceIndex[0]][1] == maxLoopValue:
        tiledTemplate[lsliceIndex[-1]][1] = maxLoopValue
        for index in lsliceIndex[:-1]:
            tiledTemplate[index][1] = 1
        return IncTileResult.LoopOverflow
    
    # increment the inner loop value until it reaches outer loop`s value
    # then increment outer loop and reset all the inner loops
    for index in xrange(len(lsliceIndex)-1):
        if tiledTemplate[lsliceIndex[index]][1] < tiledTemplate[lsliceIndex[index+1]][1]:
            tiledTemplate[lsliceIndex[index]][1] += 1 
            return IncTileResult.LoopIncremented
        else:
            for i in xrange(index):
                tiledTemplate[lsliceIndex[i]][1] = 1
            continue

'''
Assumptions
1) buffer size is at least size of RF
'''    

def OptimizeTile (tiledTemplate, layer, hardware):    
    print 'starting buffer calc'
    buff = calcBuffers(tiledTemplate)

    if not checkRestrictions(tiledTemplate, buff, hardware):
        return
        
    # bruteforce core
    it = 0
    # cycle exclude dx,dy
    loopSlice = [x for x in Loop if x!=Loop.dx and x!=Loop.dy]

    loopIndexes = []    
    for loop in Loop:
        loopIndexes.append([i for i,x in enumerate(tiledTemplate) if x[0] == loop])
        
    # indexes of unrolled loops
    templateUIndex = [i for i,x in enumerate(tiledTemplate) if len(x) == 3]
#    templateULoops = {loop[0] for loop in tiledTemplate if len(loop) == 3}
    
    rightLoop = Loop.fm
    bruteForceStep = True
    while bruteForceStep:
        it += 1
       
        for loop in loopSlice:
            rightLoop = maxLoop(rightLoop, loop)
            lsliceIndex = loopIndexes[loop.value]
            
            res = incrementTiling(tiledTemplate, lsliceIndex, layer.getByLoop(loop.value))
              
            if res == IncTileResult.LoopIncremented:
                printTile(tiledTemplate, 1)
                buff = calcBuffers(tiledTemplate)
                passRestr = checkRestrictions(tiledTemplate, buff, hardware)
                printBuff(buff)
                print passRestr
                if not passRestr[0]: # too large buffers
                    # when the latest loop group fills the buffer, we may stop bruteforce
                    if loop == Loop.row:
                        bruteForceStep = False
                        break
                    # if tiling does not satisfy restrictions reset all loop groups up to rightLoop
                    # and increase the following
                    
                    l = Loop(0)  
                    while l <= loop:
                        
                        qwdqwdqw~~~!!!111
                    
#                    for l in [x for x in loopSlice if x.value < rightLoop.value]:
#                        tiledTemplate[loopIndexes[l.value][-1][1]] = layer.getByLoop(l.value)
#                        for index in loopIndexes[l.value][:-1]:
#                            tiledTemplate[index][1] = 1
                    
                    
#                        tiledTemplate[lsliceIndex[-1]][1] = layer.getByLoop(loop.value)
#                        for index in lsliceIndex[:-1]:
#                            tiledTemplate[index][1] = 1
                    break
                elif not passRestr[1]: #too many ALUs
                    # max the loops with ALUs to reset them on the next iteration
                    for index in templateUIndex:
                       tiledTemplate[index][1] = layer.getByLoop(tiledTemplate[index][0].value)
                    break
                else:
                    pass # TODO: energy calculation   
                    pass # TODO: store optimal result
                    break
            elif res == IncTileResult.LoopOverflow:
                if loop == Loop.row:
                    bruteForceStep = False
                    break
                else:
                    continue
            else:
                print "Welp, something is wrong"
                
    print " ".join(("Total iterations", str(it)))                
    printTile(tiledTemplate,2)
        
#    printBuff(buff)
            
def Optimize (template, layer, hardware, writer):
    
    tilingDebth = [0]*6
    
    for item in template.microArchitecture:
        for loop in Loop:
            if (loop == item[0]):
                tilingDebth[loop.value] += 1
                    
    tiling = [ [loop,0] for loop in Loop for i in xrange(tilingDebth[loop.value],template.levels) 
                                                            # do not tile dx and dy
                                                            if (loop != Loop.dx and loop != Loop.dy) ]
                                                                
    #check whether dx and dy are already in, add them if not  
    if Loop.dx not in {g for g,h in tiling}:
        tiling = [[Loop.dx,0]] + tiling
    if Loop.dy not in {g for g,h in tiling}:
        tiling = [[Loop.dy,0]] + tiling
        
    prev = []
          
    for tile in itertools.permutations(tiling):
        
        # compress the string when nearby loops are the same
        tiledTemplate = template.microArchitecture + list(tile)       
        tiledTemplate = [x for x,y in zip(tiledTemplate, tiledTemplate[1:]) if x[0]!=y[0]]
        tiledTemplate.append(tile[-1])

# TODO: filter out repetitions 
        # init with all data to level DRAM
        #check if our loop is the right most of its kind
        for index in xrange(len(tiledTemplate)):
            right = tiledTemplate[index+1:]
            loop = tiledTemplate[index]
            if len([ x for x in right if x[0] == loop[0] ]) == 0:
                tiledTemplate[index][1] = layer.getByLoop(loop[0].value)
            else:
                tiledTemplate[index][1] = 1
        
        if prev == tiledTemplate:
            continue
        else:
            prev = tiledTemplate
        
        OptimizeTile(tiledTemplate, layer, hardware)
        
        
#    writer.writerow()

###############################################################################

AlexNet1 = Layer("AlexNet1")
DiaNNao = Template("DianNao", [[Loop.fm, 16, "u"], [Loop.kern, 16, "u"]])
zinq=HWrestrictions((3000,1), 32)

f = open(sys.argv[2], 'wb')
writer = csv.writer(f, delimiter=',', quotechar=',')
writer.writerow(('Template','Layer'))

Optimize (DiaNNao, AlexNet1, zinq, writer)

f.close()