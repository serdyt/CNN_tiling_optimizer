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
import time

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
    def __init__(self, DRAM=200, buff=6, RF=1, ALU=1):
        self.DRAM = DRAM
        self.buff = buff
        self.RF = RF
        self.ALU = ALU

class Buffers:
    # [RF, localBuffer, DRAM]
    
    def __init__(self, layer=None):
#        self.levels = 3
        self.Bin = [0,0,0]
        self.Bkern = [0,0,0]
        self.Bout = [0,0,0]

        self.RRin = [0,0,0]
        self.RRkern = [0,0,0]
        self.RRout = [0,0,0]
        
        if layer != None:
            self.dataIn = layer.X * layer.Y * layer.fm
            self.dataKern = layer.dx * layer.dy * layer.kern * layer.fm
            self.dataOut = (layer.X - layer.dx + 1) * (layer.Y - layer.dy + 1) * layer.fm
        else:
            self.dataIn = 0
            self.dataKern = 0
            self.dataOut = 0
        
    def totalSpace(self):
        return sum(self.Bin[:-1]) + sum(self.Bkern[:-1]) + sum(self.Bout[:-1])
        
    def buffFitMem(self, hardware):
        if self.totalSpace() > hardware.memory[0]*hardware.memory[1]:
            return False
        else:
            return True
            
    def calcUsedData(self, left,loop):
        #scan all the data to the left to find the amount of each data type being used
        usedData = [1]*len(Loop)
        for i in Loop:
            rawData = [ x for x in left if x[0] == i ]
            usedData[i.value] = 1 if len(rawData) == 0 else rawData[-1][1]
        return usedData
            
    def calcBuffSizeRR(self, left, loop, right):
        # the level of current buffer / loop
            
        if len(loop) < 3:
            usedData = self.calcUsedData(left, loop)
        
            #TODO correct levels
            # count the amount of the same loop types to the left (ommiting the HW core, which has 'u' in the 3rd element)
            # TODO: design normal data structures
            # there should be access to self.levels
        
            level = 2 - len([x[0] for x in right if x[0] == loop[0] ])
            if level < 0 or level > 2:
                print "WTF", level, loop, right

            #TODO: add buffer merging rules    
            if loop[0] == Loop.kern:
                self.Bin[level] = ((usedData[Loop.row.value] + usedData[Loop.dx.value] - 1) *
                                (usedData[Loop.col.value] + usedData[Loop.dy.value] - 1) * usedData[Loop.fm.value])
                self.RRin[level] = int(math.ceil( (float(loop[1]) / usedData[Loop.kern.value]) * 
                                        (usedData[Loop.row.value] + usedData[Loop.dx.value] - 1) *
                                        (usedData[Loop.col.value] + usedData[Loop.dy.value] - 1) /
                                        (usedData[Loop.row.value] * usedData[Loop.col.value])                                
                                        ))
            elif loop[0] == Loop.fm:
                self.Bout[level] = usedData[Loop.row.value]*usedData[Loop.col.value]*usedData[Loop.kern.value]
                self.RRout[level] = int(math.ceil(2.0 * loop[1] / usedData[Loop.fm.value]))
            #TODO: add rules to merge row/col buffer when they follow each other
            elif (loop[0] == Loop.row or loop[0] == Loop.col):
                self.Bkern[level] = usedData[Loop.kern.value]*usedData[Loop.fm.value]*usedData[Loop.dx.value]*usedData[Loop.dy.value]
                # TODO: recheck this
                self.RRkern[level] =  int(math.ceil(loop[1]*usedData[Loop.col.value if loop[0]==Loop.row else Loop.row.value] / 
                                        float(usedData[Loop.row.value]*usedData[Loop.col.value]) ))
            #TODO: check dx, dy
    #        elif (loop[0] == Loop.dx or loop[0] == Loop.dy):
    #            self.Bin[level] = ((usedData[Loop.row.value] + usedData[Loop.dx.value] - 1) *
    #                            (usedData[Loop.col.value] + usedData[Loop.dy.value] - 1) * usedData[Loop.fm.value])
    #            self.RRin[level] += loop[1] * int(math.ceil( float(usedData[Loop.row.value] + usedData[Loop.dx.value] - 1) *
    #                                    (usedData[Loop.col.value] + usedData[Loop.dy.value] - 1) /
    #                                    (usedData[Loop.row.value] * usedData[Loop.col.value])                                
    #                                    ))
    #                                    
    #            self.Bout[level] = usedData[Loop.row.value]*usedData[Loop.col.value]*usedData[Loop.kern.value]
    #            self.RRout[level] += int(math.ceil(2.0 * loop[1])) #something is wrong here
          
     
    def calcEnergy(self, energy):
        res = 0        
        for rr, data in zip([self.RRin, self.RRkern, self.RRout], [self.dataIn, self.dataKern, self.dataOut]):
            res += sum(r*e for r,e in zip(rr,[energy.RF, energy.buff, energy.DRAM])) * data
            #TODO: recheck this
        return res        

        
#    def shiftBuffers(self):
#        for x in [self.Bin, self.RRin, self.Bkern, self.RRkern, self.Bout, self.RRout]:
#            x.sort(key=lambda v: v!= 0)
            
    def __str__ (self):
        return "Bin " + str(self.Bin) + str(self.RRin) + " \n" + \
               "Bkern " + str(self.Bkern) + str(self.RRkern) + " \n" + \
               "Bout " + str(self.Bout) + str(self.RRout)

class Template:

    microArchitecture = []   # microArchitecture is a fixed part of a string
#    size = []
    levels = 3                          # maximum number of tiling of one loop
    name = ""
    buffBest = Buffers()
    energyModel = Energy(200, 6 ,1 ,1)
    energyBest = -1
    tileBest = []
        
    def __init__(self, name, microArchitecture):
        self.microArchitecture = microArchitecture
        self.name = name
        
    def Optimize (self, layer, hardware, writer):
        
#        tilingDebth = [0]*6
        
#        for item in self.microArchitecture:
#            for loop in Loop:
#                if (loop == item[0]):
#                    tilingDebth[loop.value] += 1
#                        
#        tiling = [ [loop,0] for loop in Loop for i in xrange(tilingDebth[loop.value],self.levels+1) 
#                                                                # do not tile dx and dy
#                                                                if (loop != Loop.dx and loop != Loop.dy) ]
        
        tiling = []
        for i in xrange(self.levels):
            tiling += [[x,0] for x in Loop if (x != Loop.dx and x != Loop.dy)]
                                                                 
        #check whether dx and dy are already in, add them if not  
        if Loop.dx not in [g for g,h in tiling]:
            tiling = [[Loop.dx,0]] + tiling
        if Loop.dy not in [g for g,h in tiling]:
            tiling = [[Loop.dy,0]] + tiling
            
#        prev = []
        
#        allTiles = []
#        allCount = 0
#        totalIt = 0
        
#        That is DiaNNao template
#        tiledTemplate = [[Loop.fm, 16, "u"], [Loop.kern, 16, "u"], [Loop.fm,1],[Loop.dx,1],[Loop.dy,1],[Loop.kern,1],
#                         [Loop.row,1],[Loop.col,1],[Loop.kern,1],[Loop.row,1],[Loop.col,1] ]
 
        
        for tile in itertools.permutations(tiling):
            if any([x[0] == y[0] for x,y in zip(tile, tile[1:])]):
                continue            
            
            # compress the string when nearby loops are the same
            tiledTemplate = self.microArchitecture + list(tile)       

            # init by putting all data to DRAM
            # check if our loop is the right most of its kind
            for index in xrange(len(tiledTemplate)):
                loop = tiledTemplate[index][0]
                right = {g[0] for g in tiledTemplate[index+1:]}
                if loop not in right:
                    tiledTemplate[index][1] = layer.getByLoop(loop.value)
                else:
                    tiledTemplate[index][1] = 1

            
#            if prev == tiledTemplate:
#                continue
#            else:
#                prev = tiledTemplate
                
#            if prev in allTiles:
#                allCount += 1                    
#            else:
#                allTiles.append(tiledTemplate)
#            totalIt += 1                  
#            print allCount, totalIt, len(allTiles)
            
            t1 = time.clock()
            self.OptimizeTile(tiledTemplate, layer, hardware)
            print "Time ", time.clock() - t1

    def OptimizeTile (self, tiledTemplate, layer, hardware):    

        # TODO: need to save template
        print 'starting buffer calc'
        printTile(tiledTemplate,2)
        self.buffBest = self.calcBuffers(tiledTemplate, layer)  
        if not self.buffBest.buffFitMem(hardware):
            
            print self.buffBest.calcEnergy(self.energyModel)
            print self.buffBest
            print "The layer cannot fit hardware!"
            return        
                      
        self.energyBest = self.buffBest.calcEnergy(self.energyModel)

        self.tileBest = tiledTemplate        
        
        itCounter = 0
    
        loopIndexByType = []    
        for loop in Loop:
            loopIndexByType.append([i for i,x in enumerate(tiledTemplate) if x[0] == loop])
            
        # indexes of unrolled loops
        templateUIndex = [i for i,x in enumerate(tiledTemplate) if len(x) == 3]
        # type of unrolled loops
    #    templateULoops = {loop[0] for loop in tiledTemplate if len(loop) == 3}
        

# TODO: Iterate ALU related loops separately
# So that ALU is always equal to hardware MACs


        currloopGroup = 0
        # Exit condition: all loops reach maximum, i.e. the last group overflows
        while currloopGroup < len(Loop):
            currLoopSliceIndexes = loopIndexByType[currloopGroup]
            
            res, currLoopIndex = self.incrementTiling(tiledTemplate, currLoopSliceIndexes, layer.getByLoop(currloopGroup))

            printTile(tiledTemplate, 1)
                       
            if res == IncTileResult.LoopOverflow:
                currloopGroup += 1
                continue
                
            elif res == IncTileResult.LoopIncremented:
                itCounter += 1
#                printTile(tiledTemplate, 1)

                if not ALUFitMAC(tiledTemplate, hardware.MAC):
                    self.updateTilingOnMACOverflow(tiledTemplate, loopIndexByType, templateUIndex)
#                    print 'ALU overflow'
                    currloopGroup = 0
                    continue

                currBuff = self.calcBuffers(tiledTemplate, layer)
#                print 'Buff size ', currBuff.totalSpace()
                if not currBuff.buffFitMem(hardware):
                    self.updateTilingOnBuffOverflow(tiledTemplate, loopIndexByType[0:currloopGroup+1], currLoopIndex)
#                    print 'BUFF overflow'
                    currloopGroup = 0
                    continue
                
                currEnergy = currBuff.calcEnergy(self.energyModel)
                if currEnergy < self.energyBest:
                    self.energyBest = currEnergy
                    del self.buffBest
                    self.buffBest = currBuff
                    self.tileBest = tiledTemplate
                    print ' new best! '
                    printTile(tiledTemplate,1)
                    print self.energyBest
                    print self.buffBest
                else:
                    del currBuff
                
                currloopGroup = 0
                continue
                    
        print "Total iterations", itCounter, self.energyBest
        print self.buffBest
        printTile(self.tileBest,1)
            
            
    def calcBuffers(self, tiledTemplate, layer):
        buff = Buffers(layer)
        for index in xrange(len(tiledTemplate)):
            loop = tiledTemplate[index]
            if len(loop) > 2:
                continue
            
            left = tiledTemplate[:index]
            right = tiledTemplate[index+1:]
            
            buff.calcBuffSizeRR(left, loop, right)
        
        for x in [buff.RRin, buff.RRkern, buff.RRout]:
            if x[-1] == 0:
                x[-1] = 1
                
#        buff.shiftBuffers()
        return buff
        
    def incrementTiling(self, tiledTemplate, loopSliceIndexes, maxLoopValue):
    # if current group of loops has iterated through all the variants - reset them to 1,1,max
        # and try to increment the next loop
        if tiledTemplate[loopSliceIndexes[0]][1] == maxLoopValue:
            tiledTemplate[loopSliceIndexes[-1]][1] = maxLoopValue
            for i in loopSliceIndexes[:-1]:
                tiledTemplate[i][1] = 1
            return IncTileResult.LoopOverflow, 0
        
        # increment the inner loop value until it reaches outer loop`s value
        # then increment outer loop and reset all the inner loops
        for i in xrange(len(loopSliceIndexes)-1):
            if tiledTemplate[loopSliceIndexes[i]][1] < tiledTemplate[loopSliceIndexes[i+1]][1]:
                tiledTemplate[loopSliceIndexes[i]][1] += 1 
                return IncTileResult.LoopIncremented, i
            else:
                for ii in xrange(i):
                    tiledTemplate[loopSliceIndexes[ii]][1] = 1
                continue
            
    def updateTilingOnBuffOverflow(self, tiledTemplate, currGroupSlice, currLoopIndex):
        # in all the groups <= to current group        
        # set all values to max
        for loopGroup in currGroupSlice:
            for x,y, in zip(loopGroup[-2::-1], loopGroup[::-1]):
                if x == currLoopIndex:
                    break
                else:
                    tiledTemplate[x][1] = tiledTemplate[y][1]         
            
    def updateTilingOnMACOverflow(self, tiledTemplate, loopIndexByType, templateUIndex):
        # set all MAC related loops to the value of the following loop of the same type
        # that will cause an increment in the current groop of loops at the next loop after the ALU related ones
        for loopSliceIndexes in loopIndexByType:
            for x,y in zip(loopSliceIndexes[-2::-1], loopSliceIndexes[::-1]):
                if x in templateUIndex:
                    tiledTemplate[x][1] = tiledTemplate[y][1]
        

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
    
def maxLoop(a,b):
    if a.value >= b.value:
        return a
    else:
        return b

def ALUFitMAC(template, MAC):
    return not(reduce(mul, [x[1] for x in template if len(x) == 3], 1) > MAC)
    
        
'''
Assumptions
1) buffer size is at least size of RF
'''    

        
#    printBuff(buff)
            

#    writer.writerow()

###############################################################################

#AlexNet1 = Layer("AlexNet1")
AlexNet1 = Layer("AlexNet2", X=55, Y=55, fm=48, kern=256, dx=5, dy=5, stride=1)
DiaNNao = Template("DianNao", [[Loop.fm, 16, "u"], [Loop.kern, 16, "u"]])
zinq=HWrestrictions((3000,1), 256)

#f = open(sys.argv[2], 'wb')
#writer = csv.writer(f, delimiter=',', quotechar=',')
#writer.writerow(('Template','Layer'))

DiaNNao.Optimize(AlexNet1, zinq, writer='')

#f.close()