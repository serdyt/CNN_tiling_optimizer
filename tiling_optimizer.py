#import caffe_pb2
#from google.protobuf.text_format import Merge
#
#net = caffe_pb2.NetParameter()
#Merge((open("Alexnet.prototxt",'r').read()), net)

import sys
from enum import Enum
#import csv
import itertools
from operator import mul
import time
from math import ceil, floor

# my classes
from EenrgyModel import EnergyModel
from LoopType import LoopType, Loop, RowCol
from Buffers import Buffers
from Layer import Layer
from HwTemplate import *
from Model import *

#netFileName = sys.argv[1]

# Arguments list:
# Caffe CNN description file name
# Log file name




class IncTileResult(Enum):
    LoopIncremented = 1
    LoopOverflow = 2

class TemplateOptimizer(object):

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
    
        #TODO: update for rowcol
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
 
#       CNP template
        self.hwTemplate.archU = [Loop(LoopType.dx, 5, Pragma.u), Loop(LoopType.dy, 5, Pragma.u)]
        self.hwTemplate.archP = [Loop(LoopType.rowcol, RowCol(1,1), Pragma.p)]

        tiling = [  Loop(LoopType.fm, 1, Pragma.n),
                    Loop(LoopType.kern, 1, Pragma.n),
        
                    Loop(LoopType.rowcol, RowCol(55,55), Pragma.n),
                    Loop(LoopType.fm, 1, Pragma.n),
                    Loop(LoopType.kern, 32, Pragma.n),                    
                              
                    Loop(LoopType.kern, 256, Pragma.n),
                    Loop(LoopType.rowcol, RowCol(55,55), Pragma.n),
                    Loop(LoopType.fm, 48, Pragma.n)
        
                  ] 


#       Eyeriss template
#        tiling = [  Loop(LoopType.kern, 1, Pragma.n),
#        
#                    Loop(LoopType.rowcol, RowCol(55,55), Pragma.n),
#                    Loop(LoopType.fm, 1, Pragma.n),
#                    Loop(LoopType.kern, 32, Pragma.n),                    
#                              
#                    Loop(LoopType.kern, 256, Pragma.n),
#                    Loop(LoopType.rowcol, RowCol(55,55), Pragma.n),
#                    Loop(LoopType.fm, 48, Pragma.n)
#        
#                  ] 
        
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
                    if tile[index].type == LoopType.rowcol:
                        tile[index].size = RowCol(1,1)    
                    else:
                        tile[index].size = 1


#            if prev in allTiles:
#                allCount += 1                    
#            else:
#                allTiles.append(tiledTemplate)
#            totalIt += 1                  
#            print allCount, totalIt, len(allTiles)
            
#            print 'starting buffer calc'
#            printTile(self.hwTemplate.archU + self.hwTemplate.archP + list(tile),1)
            t1 = time.clock()
            self.OptimizeTile(tile)
            print "Time ", time.clock() - t1
            sys.exit()
            
    def copyTilingBest(self, tile):
        self.tilingBest = []
        for loop in tile:
            if (loop.type == LoopType.rowcol):
                self.tilingBest.append(Loop(loop.type, RowCol(loop.size.row, loop.size.col), loop.pragma))
            else:
                self.tilingBest.append(Loop(loop.type, loop.size, loop.pragma))
            
    def OptimizeTile (self, tiling):   
        
        self.buffBest = self.hwTemplate.calcBuffers(tiling, self.layer)
        
        if not self.buffFitMem(self.buffBest):
            
            print self.hwTemplate.calcEnergy(self.buffBest, self.energyModel)
            print self.buffBest
            print "The layer cannot fit hardware!"
            return        
                      
        self.energyBest = self.hwTemplate.calcEnergy(self.buffBest, self.energyModel)
        
        self.copyTilingBest(self.hwTemplate.archU + self.hwTemplate.archP + list(tiling))        
        
        tiledArch = self.hwTemplate.archP + list(tiling)
        
        loopIndexByType = []
        for loop in LoopType:
            loopIndexByType.append([i for i,x in enumerate(tiledArch) if x.type == loop])

        iterationCounter = 0
        t2 = time.clock()
        # Pragma.u loops alwasy occupy all possible MACs
        self.hwTemplate.MACRestrictions(self.layer)
        for ALUlist in self.hwTemplate.ALUperm(self.hwTemplate.numU, self.hwRestrictions.MAC, self.hwRestrictions.MAC):
            self.hwTemplate.setALU(ALUlist)
            
            # find minimal value for each loop type from archU
            minLoopValues = [1]*len(LoopType)
            for i in LoopType:
                if i == LoopType.rowcol:
                    minLoopValues[i.value] = RowCol(1,1)
                    break
            
            for loop in self.hwTemplate.archU:
                if (loop.type == LoopType.rowcol):
                    minLoopValues[loop.type.value] = RowCol(loop.size.row, loop.size.col)
                else:
                    minLoopValues[loop.type.value] = loop.size
                    
            minLoopValues[LoopType.dx.value] = self.layer.dx
            minLoopValues[LoopType.dy.value] = self.layer.dy
                
                
            # update initial state of tiledArch to be minLoopValue except of 3 last loops, which have the max value
            for i, loop in enumerate(tiledArch[0:-3]):
                if loop.type == LoopType.rowcol:
                    tiledArch[i].size.row = minLoopValues[loop.type.value].row
                    tiledArch[i].size.col = minLoopValues[loop.type.value].col
                else:
                    tiledArch[i].size = minLoopValues[loop.type.value]                   
                    
#                for i, archLoop in enumerate(tiledArch):
#                    if archLoop.type == loop.type:
#                        if archLoop.type in [x.type for x in tiledArch[i+1:]]:
#                            tiledArch[i].size = loop.size
                        
#            print "ALUperm", iterationCounter, " took: ", time.clock() - t2
#            printTile(self.hwTemplate.archU + tiledArch, 1)
#            t2 = time.clock()
            currGroup = 0
            
            # Main loop
            # Exit condition: all loops reach maximum, i.e. the last group overflows
            while currGroup < len(LoopType) - 2:

                currGroupSliceIndexes = loopIndexByType[currGroup]
                res, currLoopIndex = self.incrementTiling(tiledArch, currGroupSliceIndexes, self.layer.getMaxLoopSizeByIndex(currGroup), minLoopValues[currGroup])

                if res == IncTileResult.LoopOverflow:
                    currGroup += 1
                    continue
                    
                if res == IncTileResult.LoopIncremented:
#                    print ""
#                    printTile(self.hwTemplate.archU +tiledArch, 1)
                    iterationCounter += 1
#                    print 'iteration ', iterationCounter
#                    if iterationCounter > 500000:
#                        return
    
                    self.hwTemplate.archP = tiledArch[:len(self.hwTemplate.archP)]
                    currBuff = self.hwTemplate.calcBuffers(tiledArch[len(self.hwTemplate.archP):], self.layer)
                    
#                    print currBuff

                    if not self.buffFitMem(currBuff):
                        self.updateTilingOnBuffOverflow(tiledArch, loopIndexByType[0:currGroup+1], currLoopIndex)
                        currGroup = 0
#                        print 'buff overflow'
                        continue
                    
                    currEnergy = self.hwTemplate.calcEnergy(currBuff, self.energyModel)
                    printTile(self.hwTemplate.archU +tiledArch, 3, currEnergy, sum(currBuff.Bin)+sum(currBuff.Bkern)+sum(currBuff.Bout))
#                    print currEnergy
                    if currEnergy < self.energyBest:
                        self.energyBest = currEnergy
                        del self.buffBest
                        self.buffBest = currBuff
#                        del self.tilingBest
#                        self.tilingBest = self.hwTemplate.archU + tiledArch
                        self.copyTilingBest(self.hwTemplate.archU + tiledArch)
#                        print 'new best! '
#                        printTile(self.hwTemplate.archU + tiledArch,1)
#                        print self.energyBest
#                        print self.buffBest
                    else:
#                        diff = self.hwRestrictions.memory[0]*self.hwRestrictions.memory[1] - currBuff.totalSpace() 
                        # increment currlooop so that buffer is maximised
                        del currBuff
                
                    currGroup = 0
                    continue
                    
        print "Total iterations", iterationCounter, self.energyBest
        print self.buffBest
        printTile(self.tilingBest,1)
 
    def buffFitMem(self, buff):
        if buff.totalSpace() > self.hwRestrictions.memory[0]*self.hwRestrictions.memory[1]:
            return False
        else:
            return True
            
    def nextDivisor(self, curr, maxi):
        for i in xrange(maxi, 0, -1):
            t = ceil(maxi/float(i))
            if t > curr:
                return int(t)
        
    def incrementTiling(self, tiledTemplate, groupIndexes, maxLoopValue, minLoopValue):
    # if current group of loops has iterated through all the variants - reset them to 1,1,max
        # and try to increment the next loop
        if tiledTemplate[groupIndexes[0]].size == maxLoopValue:
#           TODO: this is probably not needed - [-1] is always maxLoopValue
#            tiledTemplate[groupIndexes[-1]].size = maxLoopValue
            for i in groupIndexes[:-1]:
                if (tiledTemplate[i].type == LoopType.rowcol):
                    tiledTemplate[i].size.row = minLoopValue.row
                    tiledTemplate[i].size.col = minLoopValue.col
                else:
                    tiledTemplate[i].size = minLoopValue
            return IncTileResult.LoopOverflow, 0
        
        # increment the inner loop value until it reaches outer loop`s value
        # then increment outer loop and reset all the inner loops
        for i in xrange(len(groupIndexes)-1):
            
            # RoWCol requires separate processing
            if (tiledTemplate[groupIndexes[i]].type == LoopType.rowcol):             
                # if both row and col are equal reset the state of the current loop and increment the following
                if tiledTemplate[groupIndexes[i]].size == tiledTemplate[groupIndexes[i+1]].size:
                    for ii in xrange(i+1):        
                        tiledTemplate[groupIndexes[ii]].size.row = minLoopValue.row
                        tiledTemplate[groupIndexes[ii]].size.col = minLoopValue.col
                # if only cols are equal reset col and increment row
                elif tiledTemplate[groupIndexes[i]].size.col == tiledTemplate[groupIndexes[i+1]].size.col:
                    tiledTemplate[groupIndexes[i]].size.row = self.nextDivisor(tiledTemplate[groupIndexes[i]].size.row, maxLoopValue.row)    
                    tiledTemplate[groupIndexes[i]].size.col = minLoopValue.col
                    return IncTileResult.LoopIncremented, i
                # in normal case increase col
                else:
                    tiledTemplate[groupIndexes[i]].size.col = self.nextDivisor(tiledTemplate[groupIndexes[i]].size.col, maxLoopValue.col)                    
                    return IncTileResult.LoopIncremented, i
                        
            # normal loops
            else:
                if tiledTemplate[groupIndexes[i]].size < tiledTemplate[groupIndexes[i+1]].size:
                    tiledTemplate[groupIndexes[i]].size = self.nextDivisor(tiledTemplate[groupIndexes[i]].size,tiledTemplate[groupIndexes[i+1]].size)  
                    return IncTileResult.LoopIncremented, i
                else:
                    for ii in xrange(i+1):
                            tiledTemplate[groupIndexes[ii]].size = minLoopValue
                    continue
        print "you should never be here"
        raise Exception('you should never be here')
            
    def updateTilingOnBuffOverflow(self, tiledTemplate, currGroupSlice, currLoopIndex):
        # in all the groups <= to current group        
        # set all values to max
        for loopGroup in currGroupSlice:
            for x,y, in zip(loopGroup[-2::-1], loopGroup[::-1]):
                if x == currLoopIndex:
                    break
                else:
                    if (tiledTemplate[x].type == LoopType.rowcol):
                        tiledTemplate[x].size.row = tiledTemplate[y].size.row
                        tiledTemplate[x].size.col = tiledTemplate[y].size.col
                    else:
                        tiledTemplate[x].size = tiledTemplate[y].size
                    

class HWrestrictions:
# Zynq Z-7007S
#    memory = (36000, 50)                # (size of a block, number of blocks)
    
    def __init__(self, memory=((36000, 50)) , MAC = 66):
        self.memory = memory
        self.MAC = MAC
        
def printTile (tile, i=2, energy=0, buff=0):
    if i == 1:
        print  [(elm.type.name, elm.size) for elm in tile]
    elif i == 2:
        print  [elm.type.name for elm in tile]
    elif i == 3:
        lst = [str(elm.size) for elm in tile]
        print '\t'.join(lst+[str(int(energy))]+[str(buff)])
    else:
        print tile
    
def maxLoop(a,b):
    if a.value >= b.value:
        return a
    else:
        return b

        
#    printBuff(buff)
            

#    writer.writerow()

###############################################################################

sys.stdout = open('CNP5000', 'w')
##
###AlexNet1 = Layer("AlexNet1")
AlexNet2 = Layer("AlexNet2", X=55, Y=55, fm=48, kern=256, dx=5, dy=5, stride=1)
##
zinq=HWrestrictions((5000,1), 256)
#
##f = open(sys.argv[2], 'wb')
##writer = csv.writer(f, delimiter=',', quotechar=',')
##writer.writerow(('Template','Layer'))
#
template = CNP()
#template = Eyeriss()
#
optimizer = TemplateOptimizer(template, AlexNet2, zinq)
optimizer.Optimize()
#    
    
#dian = DiaNNao()
#dian.archU = [Loop(LoopType.fm, 4, Pragma.u), Loop(LoopType.kern, 4, Pragma.u)]
#dian.archU = [Loop(LoopType.dx, 5, Pragma.u), Loop(LoopType.dy, 5, Pragma.u), Loop(LoopType.kern, 4, Pragma.u )]

#dian = CNP()
#dian.archU = [Loop(LoopType.dx, 5, Pragma.u), Loop(LoopType.dy, 5, Pragma.u)]
#dian.archP = [Loop(LoopType.rowcol, RowCol(7,10), Pragma.p)]
#
##dian = Eyeriss()
##dian.archU = [Loop(LoopType.dy, 5, Pragma.u), Loop(LoopType.rowcol, RowCol(28,1), Pragma.u), Loop(LoopType.kern, 1, Pragma.u)]
##dian.archP =[Loop(LoopType.rowcol, RowCol(28,1), Pragma.p), Loop(LoopType.dx, 5, Pragma.p), Loop(LoopType.fm, 3, Pragma.p)]
#
##from test import divisorsCeil
##for ii1 in divisorsCeil(1, 55):
##    for i1 in divisorsCeil(1, 55):
##        
##        for i in divisorsCeil(1, i1):
##            for ii in divisorsCeil(1, ii1):
#
#
#tiling = [      Loop(LoopType.fm, 4, Pragma.n),
#Loop(LoopType.kern, 43, Pragma.n),
#
#            Loop(LoopType.rowcol, RowCol(55,55), Pragma.n),
#            Loop(LoopType.fm, 4, Pragma.n),
#            Loop(LoopType.kern, 43, Pragma.n),                    
#            
#            Loop(LoopType.rowcol, RowCol(55,55), Pragma.n),
#            Loop(LoopType.kern, 256, Pragma.n),
#            Loop(LoopType.fm, 48, Pragma.n)
#
#          ]
#
#buff = dian.calcBuffers(tiling, AlexNet2)  
#
#currEnergy = dian.calcEnergy(buff, EnergyModel())
#printTile(dian.archU + dian.archP + tiling,1)
#print buff
##print (ii, i), (ii1, i1), currEnergy, sum(buff.Bin)+sum(buff.Bkern)+sum(buff.Bout)
#print currEnergy, sum(buff.Bin)+sum(buff.Bkern)+sum(buff.Bout)

#f.close()