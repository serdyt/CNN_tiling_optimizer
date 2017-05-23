#import caffe_pb2
#from google.protobuf.text_format import Merge
#
#net = caffe_pb2.NetParameter()
#Merge((open("Alexnet.prototxt",'r').read()), net)

import sys
from enum import Enum
import csv
import itertools
from operator import mul, itemgetter
import time

# my classes
from EenrgyModel import EnergyModel
from LoopType import LoopType, Loop
from Buffers import Buffers
from Layer import Layer
from HwTemplate import DiaNNao, Pragma


#netFileName = sys.argv[1]

# Arguments list:
# Caffe CNN description file name
# Log file name




class IncTileResult(Enum):
    LoopIncremented = 1
    LoopOverflow = 2

class TemplateOptimizer:

    levels = 3
    
    def __init__(self, template, layer, hwRestrictions):
        self.buffBest = Buffers()
        self.energyModel = EnergyModel(200, 6 ,1 ,1)
        self.tilingBest = []
        self.energyBest = 0
        
        self.hwTemplate = template
        self.layer = layer
        self.hwRestrictions = hwRestrictions
        
        
    def Optimize (self):
     # TODO: make two loops so that the number of tiling loops gradualy increases
    # and filter out tilings with neigbouring loops of the same kind
    
        # add loop missing from archU
        tiling = [Loop(x,1,Pragma.n) for x in LoopType if (x != LoopType.dx and x != LoopType.dy and \
                x not in [y.type for y in self.hwTemplate.archP])]

        for i in xrange(self.levels-1):
            tiling += [Loop(x,1,Pragma.n) for x in LoopType if (x != LoopType.dx and x != LoopType.dy)]
                   
        #check whether dx and dy are already in, add them if not  
        if LoopType.dx not in [x.type for x in self.hwTemplate.archU + self.hwTemplate.archP + tiling]:
            tiling = [Loop(LoopType.dx,self.layer.dx,Pragma.n)] + tiling
        if LoopType.dy not in [x.type for x in self.hwTemplate.archU + self.hwTemplate.archP + tiling]:
            tiling = [Loop(LoopType.dy,self.layer.dy,Pragma.n)] + tiling
            
        
#        allTiles = []
#        allCount = 0
#        totalIt = 0
        
#        That is DiaNNao template
#        tiledTemplate = [[LoopType.fm, 16, "u"], [LoopType.kern, 16, "u"], [LoopType.fm,1],[LoopType.dx,1],[LoopType.dy,1],[LoopType.kern,1],
#                         [LoopType.row,1],[LoopType.col,1],[LoopType.kern,1],[LoopType.row,1],[LoopType.col,1] ]
 
        
        for tile in itertools.permutations(tiling):
#            skip permutations where two neighbours have the same loop type
            if any([x.type == y.type for x,y in zip(tile, tile[1:])]):
                continue            
            
#            tiledTemplate = self.hwTemplate.architecture + list(tile)
            
            tile = list(tile)

            # init by putting all data to DRAM
            for index in xrange(len(tile)):
                loopType = tile[index].type
                right = [g.type for g in tile[index+1:]]
                if loopType not in right:
                    tile[index].size = self.layer.getMaxLoopSize(loopType)
                else:
                    tile[index].size = 1


#            if prev in allTiles:
#                allCount += 1                    
#            else:
#                allTiles.append(tiledTemplate)
#            totalIt += 1                  
#            print allCount, totalIt, len(allTiles)
            
            print 'starting buffer calc'
            printTile(self.hwTemplate.archU + self.hwTemplate.archP + list(tiling),2)
            t1 = time.clock()
            self.OptimizeTile(tile)
            print "Time ", time.clock() - t1
            sys.exit()
            
    def OptimizeTile (self, tiling):   
        
        self.buffBest = self.calcBuffers(self.hwTemplate.archP + tiling)  
        self.buffBest.concatBuff(self.hwTemplate.buff)
        
        if not self.buffFitMem(self.buffBest):
            
            print self.calcEnergy(self.buffBest)
            print self.buffBest
            print "The layer cannot fit hardware!"
            return        
                      
        self.energyBest = self.calcEnergy(self.buffBest)
        
        self.tilingBest = self.hwTemplate.archU + self.hwTemplate.archP + list(tiling)
           
        tiledArch = self.hwTemplate.archP + list(tiling)
        
        loopIndexByType = []
        for loop in LoopType:
            loopIndexByType.append([i for i,x in enumerate(tiledArch) if x.type == loop])
            
        # indexes of unrolled loops
#        templateUIndex = [i for i,x in enumerate(tiledTemplate) if len(x) == 3]
        # type of unrolled loops
    #    templateULoops = {loop[0] for loop in tiledTemplate if len(loop) == 3}
        
        itCounter = 0
        t2 = time.clock()
        
        # Pragma.u loops alwasy occupy all possible MACs
        for ALUlist in self.ALUperm(self.hwTemplate.numU, self.hwRestrictions.MAC, self.hwRestrictions.MAC):
            self.hwTemplate.setALU(ALUlist)
            
            minLoopValues = [1]*len(LoopType)
            for loop in self.hwTemplate.archU:
                minLoopValues[loop.type.value] = loop.size
                for i, archLoop in enumerate(tiledArch):
                    if archLoop.type == loop.type:
                        if archLoop.type in [x.type for x in tiledArch[i+1:]]:
                            tiledArch[i].size = loop.size
                        
            print "ALUperm", itCounter, " took: ", time.clock() - t2
            t2 = time.clock()
            currloopGroup = 0
            # Main loop
            # Exit condition: all loops reach maximum, i.e. the last group overflows
            while currloopGroup < len(LoopType):
                               
                currLoopSliceIndexes = loopIndexByType[currloopGroup]
                res, currLoopIndex = self.incrementTiling(tiledArch, currLoopSliceIndexes, self.layer.getMaxLoopSizeByIndex(currloopGroup), minLoopValues[currloopGroup])

                if res == IncTileResult.LoopOverflow:
                    currloopGroup += 1
                    continue
                    
                if res == IncTileResult.LoopIncremented:
#                    printTile(tiledArch, 1)
                    itCounter += 1
                    print itCounter
                    if itCounter > 50000:
                        sys.exit()
    
                    currBuff = self.calcBuffers(tiledArch)
                    currBuff.concatBuff(self.hwTemplate.buff)

                    if not self.buffFitMem(currBuff):
                        self.updateTilingOnBuffOverflow(tiledArch, loopIndexByType[0:currloopGroup+1], currLoopIndex)
                        currloopGroup = 0
#                        print 'buff overflow'
                        continue
                    
                    currEnergy = self.calcEnergy(currBuff)
#                    print currEnergy
                    if currEnergy < self.energyBest:
                        self.energyBest = currEnergy
                        del self.buffBest
                        self.buffBest = currBuff
                        self.tilingBest = tiledArch
#                        print ' new best! '
#                        printTile(self.hwTemplate.archU + tiledArch,1)
#                        print self.energyBest
#                        print self.buffBest
                    else:
#                        diff = self.hwRestrictions.memory[0]*self.hwRestrictions.memory[1] - currBuff.totalSpace() 
                        # increment currlooop so that buffer is maximised
                        del currBuff
                
                    currloopGroup = 0
                    continue
                    
        print "Total iterations", itCounter, self.energyBest
        print self.buffBest
        printTile(self.tilingBest,1)
 
    def buffFitMem(self, buff):
        if buff.totalSpace() > self.hwRestrictions.memory[0]*self.hwRestrictions.memory[1]:
            return False
        else:
            return True
            
            
    def calcEnergy(self, buff):
        res = 0        
            
        for rr in [buff.RRin, buff.RRkern, buff.RRout]:
            res += rr[2]*self.energyModel.DRAM + rr[2]*rr[1]*self.energyModel.buff + rr[2]*rr[1]*rr[0]*self.energyModel.RF
    
        return res 
            
            
    def calcBuffers(self, tiledTemplate):
        buff = Buffers()
        for index in xrange(len(tiledTemplate)):
            loop = tiledTemplate[index]          
            left = tiledTemplate[:index]
           
#            print left, loop, self.hwTemplate.archU
            buff.calcBuffSizeRR(left, loop, self.hwTemplate.archU)
        
    
    #    for some reason this erases everything to 0
    #    for rr,b in zip([buff.RRin, buff.RRkern, buff.RRout],[buff.Bin, buff.Bkern, buff.Bout]):
    #        b[0] = b[0] if int(rr[0]) == 1 else 0
    
    #   if RR==1 we do not need the buffer
    #   normal code does not want to do it, my mind is blowing
        for i in [0,1]:
            if int(buff.RRin[i]) == 1:
                buff.RRin[i] = 0
                buff.Bin[i] = 0
    
            if int(buff.RRkern[i]) == 1:
                buff.RRkern[i] = 0
                buff.Bkern[i] = 0
        
            if int(buff.RRout[i]) == 1:
                buff.RRout[i] = 0
                buff.Bout[i] = 0    
            else:
                buff.RRout[i] *= 2
        
    #    for x in [buff.RRin, buff.RRkern, buff.RRout]:
    #        if x[-1] == 0:
    #            x[-1] = 1
            return buff
        
#   divides MACs for ALUs with reminder = 0       
#   TODO: consider non-full allocation of MACS, i.e. for 17 MACs
    def ALUperm(self, depth, MAC, maxMAC):
        if depth == 1:
            if MAC ==0:
                pass
            else:
                yield [MAC]
        else:
            for i in xrange(1,maxMAC+1):
                if MAC%i == 0:
                    for perm in self.ALUperm(depth-1, MAC/i, maxMAC):
                        yield [i] + perm
        
    def incrementTiling(self, tiledTemplate, loopSliceIndexes, maxLoopValue, minLoopValue):
    # if current group of loops has iterated through all the variants - reset them to 1,1,max
        # and try to increment the next loop
        if tiledTemplate[loopSliceIndexes[0]].size == maxLoopValue:
#           TODO: this is probably not needed - [-1] is always maxLoopValue
            tiledTemplate[loopSliceIndexes[-1]].size = maxLoopValue
            for i in loopSliceIndexes[:-1]:
                tiledTemplate[i].size = minLoopValue
            return IncTileResult.LoopOverflow, 0
        
        # increment the inner loop value until it reaches outer loop`s value
        # then increment outer loop and reset all the inner loops
        for i in xrange(len(loopSliceIndexes)-1):
            if tiledTemplate[loopSliceIndexes[i]].size < tiledTemplate[loopSliceIndexes[i+1]].size:
                tiledTemplate[loopSliceIndexes[i]].size += 1 
                return IncTileResult.LoopIncremented, i
            else:
                for ii in xrange(i+1):
                    tiledTemplate[loopSliceIndexes[ii]].size = minLoopValue
                continue
            
    def updateTilingOnBuffOverflow(self, tiledTemplate, currGroupSlice, currLoopIndex):
        # in all the groups <= to current group        
        # set all values to max
        for loopGroup in currGroupSlice:
            for x,y, in zip(loopGroup[-2::-1], loopGroup[::-1]):
                if x == currLoopIndex:
                    break
                else:
                    tiledTemplate[x].size = tiledTemplate[y].size      


class HWrestrictions:
# Zynq Z-7007S
#    memory = (36000, 50)                # (size of a block, number of blocks)
    
    def __init__(self, memory=((36000, 50)) , MAC = 66):
        self.memory = memory
        self.MAC = MAC
        
def printTile (tile, i=2):
    if i == 1:
        print  [(elm.type.name, elm.size) for elm in tile]
    elif i == 2:
        print  [elm.type.name for elm in tile]
    else:
        print tile
    
def maxLoop(a,b):
    if a.value >= b.value:
        return a
    else:
        return b
    
        
'''
Assumptions
1) buffer size is at least size of RF
'''    

        
#    printBuff(buff)
            

#    writer.writerow()

###############################################################################

#AlexNet1 = Layer("AlexNet1")
AlexNet2 = Layer("AlexNet2", X=55, Y=55, fm=48, kern=256, dx=5, dy=5, stride=1)

zinq=HWrestrictions((3000,1), 256)

#f = open(sys.argv[2], 'wb')
#writer = csv.writer(f, delimiter=',', quotechar=',')
#writer.writerow(('Template','Layer'))

optimizer = TemplateOptimizer(DiaNNao(), AlexNet2, zinq)
optimizer.Optimize()


#def calcBuffers(tiledTemplate, arch):
#    buff = Buffers()
#    for index in xrange(len(tiledTemplate)):
#        loop = tiledTemplate[index]          
#        left = tiledTemplate[:index]
#       
#        buff.calcBuffSizeRR(left, loop, arch)
#    
#
##    for some reason this erases everything to 0
##    for rr,b in zip([buff.RRin, buff.RRkern, buff.RRout],[buff.Bin, buff.Bkern, buff.Bout]):
##        b[0] = b[0] if int(rr[0]) == 1 else 0
#
##   if RR==1 we do not need the buffer
##   normal code does not want to do it, my mind is blowing
#    for i in [0,1]:
#        if int(buff.RRin[i]) == 1:
#            buff.RRin[i] = 0
#            buff.Bin[i] = 0
#
#        if int(buff.RRkern[i]) == 1:
#            buff.RRkern[i] = 0
#            buff.Bkern[i] = 0
#    
#        if int(buff.RRout[i]) == 1:
#            buff.RRout[i] = 0
#            buff.Bout[i] = 0    
#        else:
#            buff.RRout[i] *= 2
#    
##    for x in [buff.RRin, buff.RRkern, buff.RRout]:
##        if x[-1] == 0:
##            x[-1] = 1
#               
#        
#    for i,d in enumerate(zip(buff.Bkern, buff.RRkern)):
#        if int(d[1]) == 1:        
#            buff.Bkern[i] = 0
#            
#    for i,d in enumerate(zip(buff.Bin, buff.RRin)):
#        if int(d[1]) == 1:        
#            buff.Bin[i] = 0
#            
#    for i,d in enumerate(zip(buff.Bout, buff.RRout)):
#        if int(d[1]) == 1:        
#            buff.Bout[i] = 0
#    
##    buff.Bin = [b if rr != 1 else 0 for b,rr in zip(buff.Bin, buff.RRin)]
##    buff.Bkern = [b if rr != 1 else 0 for b,rr in zip(buff.Bkern, buff.RRkern)]
##    buff.Bout = [b if rr != 1 else 0 for b,rr in zip(buff.Bout, buff.RRout)]
#    
#    return buff
#
#
#def calcEnergy(buff, layer, energyModel):
#    res = 0        
#        
#    for rr in [buff.RRin, buff.RRkern, buff.RRout]:
#        res += rr[2]*energyModel.DRAM + rr[2]*rr[1]*energyModel.buff + rr[2]*rr[1]*rr[0]*energyModel.RF
#
#    return res

#dian = DiaNNao()
#tiling = [Loop(LoopType.fm, 48, Pragma.n),
#          Loop(LoopType.dx, 5, Pragma.n),
#          Loop(LoopType.dy, 5, Pragma.n),
#          Loop(LoopType.kern, 16, Pragma.n),
#          Loop(LoopType.row, 1, Pragma.n),
#          Loop(LoopType.col, 55, Pragma.n),
#          Loop(LoopType.kern, 256, Pragma.n),
#          Loop(LoopType.row, 55, Pragma.n),
#          Loop(LoopType.col, 55, Pragma.n),
#          ]

#tiling = [Loop(LoopType.fm, 48, Pragma.n),
#          Loop(LoopType.dx, 5, Pragma.n),
#          Loop(LoopType.dy, 5, Pragma.n),
#          Loop(LoopType.kern, 16, Pragma.n),
#          Loop(LoopType.row, 1, Pragma.n),
#          Loop(LoopType.col, 55, Pragma.n),
#
#Loop(LoopType.kern, 16, Pragma.n),
#Loop(LoopType.fm, 48, Pragma.n),
#Loop(LoopType.fm, 48, Pragma.n),
#Loop(LoopType.row, 1, Pragma.n),
#Loop(LoopType.col, 55, Pragma.n),
#
#          Loop(LoopType.kern, 256, Pragma.n),
#          Loop(LoopType.row, 55, Pragma.n),
#          Loop(LoopType.col, 55, Pragma.n),
#
#          ]

#tiling = [Loop(LoopType.fm, 48, Pragma.n),
##          Loop(LoopType.row, 4, Pragma.n),
#          Loop(LoopType.dx, 5, Pragma.n),
#          Loop(LoopType.dy, 5, Pragma.n),
#        Loop(LoopType.row, 28, Pragma.n),
#          Loop(LoopType.kern, 16, Pragma.n),
#          Loop(LoopType.col, 55, Pragma.n),
#
#Loop(LoopType.kern, 48, Pragma.n),
#Loop(LoopType.fm, 48, Pragma.n),
#Loop(LoopType.fm, 48, Pragma.n),
#Loop(LoopType.col, 55, Pragma.n),
#Loop(LoopType.row, 55, Pragma.n),
#          Loop(LoopType.kern, 256, Pragma.n),
#          Loop(LoopType.row, 55, Pragma.n),
#          Loop(LoopType.col, 55, Pragma.n),
#
#          ]
#
#buff = calcBuffers(tiling, dian.archU + dian.archP)  
#buff.concatBuff(dian.buff)
#
#currEnergy = calcEnergy(buff, AlexNet2, EnergyModel())
#print buff
#print currEnergy


#f.close()