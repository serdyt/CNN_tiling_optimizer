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
from LoopType import LoopType, Loop, RowCol, OptArgs
from Buffers import Buffers
from Layer import *
from HwTemplate import *
from Model import *
import copy

from multiprocessing import Pool, cpu_count
import os as os

#netFileName = sys.argv[1]

# Arguments list:
# Caffe CNN description file name
# Log file name




class IncTileResult(Enum):
    LoopIncremented = 1
    LoopOverflow = 2

class TemplateOptimizer(object):
    
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
    
        # add loops missing from archP
        tiling = [Loop(x,1,Pragma.n) for x in LoopType if (x != LoopType.dx and x != LoopType.dy and \
                x not in [y.type for y in self.hwTemplate.archP + self.hwTemplate.skipLoop])]

        #check whether dx and dy are already in, add them if not  
        if LoopType.dx not in [x.type for x in self.hwTemplate.archU + self.hwTemplate.archP + tiling]:
            tiling = [Loop(LoopType.dx,self.layer.dx,Pragma.n)] + tiling
        if LoopType.dy not in [x.type for x in self.hwTemplate.archU + self.hwTemplate.archP + tiling]:
            tiling = [Loop(LoopType.dy,self.layer.dy,Pragma.n)] + tiling

        t1 = time.time()

        p = Pool(processes = cpu_count())
            
        for tilingBest, buffBest, energyBest in p.imap_unordered(OptimizeTile, self.permutate(tiling)):
            printTile(tilingBest, 1)
            print buffBest
            print "Energy ", energyBest
            
        p.close()
        p.join()
        print "Time ", time.time() - t1

#########################################
############ single thread ##############
#########################################
#        for i in self.permutate(tiling):
#            OptimizeTile(i)
        

    def permutate(self, tiling):
        usedTiles = []
        archTile = [x.type for x in self.hwTemplate.archP]

        ptile = tiling + [Loop(x,1,Pragma.n) for x in LoopType if (x != LoopType.dx and x != LoopType.dy)]
        for ctile in itertools.permutations(ptile):
            
            # skip permutations where two neighbours have the same loop type
            fctile = list(archTile) + [x.type for x in list(ctile)]
            if any([x == y for x,y in zip(fctile, fctile[1:])]):
                continue

            
            if fctile in usedTiles:
                continue
            else:
                usedTiles.append(fctile)
        
            curtile = list(ctile) + [Loop(x,1,Pragma.n) for x in LoopType if (x != LoopType.dx and x != LoopType.dy)]

            # init by putting all data to DRAM
            for index in xrange(len(curtile)):
                loopType = curtile[index].type
                right = [g.type for g in curtile[index+1:]]
                if loopType not in right:
                    curtile[index].size = self.layer.getMaxLoopSize(loopType)
                else:
                    if curtile[index].type == LoopType.rowcol:
                        curtile[index].size = RowCol(1,1)    
                    else:
                        curtile[index].size = 1

                            
#            print "New Tile"
#            printTile(curtile,1)
#            print "hwTemaplate"
#            printTile(self.hwTemplate.archU,1)
#            printTile(self.hwTemplate.archP,1)
                        
            yield OptArgs(curtile, self.hwTemplate, self.layer, self.energyModel, self.hwRestrictions)

#**********************************************#
            
def nextDivisor(curr, maxi):
    for i in xrange(maxi, 0, -1):
        t = ceil(maxi/float(i))
        if t > curr:
            return int(t)
      
def incrementTiling(tiledTemplate, groupIndexes, maxLoopValue, minLoopValue):
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
                tiledTemplate[groupIndexes[i]].size.row = nextDivisor(tiledTemplate[groupIndexes[i]].size.row, maxLoopValue.row)    
                tiledTemplate[groupIndexes[i]].size.col = minLoopValue.col
                return IncTileResult.LoopIncremented, i
            # in normal case increase col
            else:
                tiledTemplate[groupIndexes[i]].size.col = nextDivisor(tiledTemplate[groupIndexes[i]].size.col, maxLoopValue.col)                    
                return IncTileResult.LoopIncremented, i
                    
        # normal loops
        else:
            if tiledTemplate[groupIndexes[i]].size < tiledTemplate[groupIndexes[i+1]].size:
                tiledTemplate[groupIndexes[i]].size = nextDivisor(tiledTemplate[groupIndexes[i]].size,tiledTemplate[groupIndexes[i+1]].size)  
                return IncTileResult.LoopIncremented, i
            else:
                for ii in xrange(i+1):
                        tiledTemplate[groupIndexes[ii]].size = minLoopValue
                continue

    raise Exception('you should never be here')

def updateTilingOnBuffOverflow(tiledTemplate, currGroupSlice, currLoopIndex):
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

def OptimizeTile (args):
    tiling = copy.deepcopy(args.tiling)
    hwTemplate = copy.deepcopy(args.hwTemplate)
    layer = args.layer
    energyModel = args.energyModel
    hwRestrictions = args.hwRestrictions
    
    tiledArch = hwTemplate.archP + list(tiling)
    printTile(tiledArch, 2)
    
    buffBest = hwTemplate.calcBuffers(tiling, layer)
    energyBest = hwTemplate.calcEnergy(buffBest, energyModel, layer)
    tilingBest = copyTilingBest(hwTemplate.archU + hwTemplate.archP + list(tiling)) 
    
    if not buffFitMem(buffBest, hwRestrictions):
        return tilingBest, buffBest, energyBest           
    
    loopIndexByType = []
    for loop in LoopType:
        loopIndexByType.append([i for i,x in enumerate(tiledArch) if x.type == loop])

    iterationCounter = 0
#        t2 = time.clock()
    # Pragma.u loops alwasy occupy all possible MACs
    hwTemplate.MACRestrictions(layer)
    for ALUlist in hwTemplate.ALUperm(hwTemplate.numU, hwRestrictions.MAC, hwRestrictions.MAC):
        hwTemplate.setALU(ALUlist)
        
        # find minimal value for each loop type from archU
        minLoopValues = [1]*len(LoopType)
        for i in LoopType:
            if i == LoopType.rowcol:
                minLoopValues[i.value] = RowCol(1,1)
                break
        
        for loop in hwTemplate.archU:
            if (loop.type == LoopType.rowcol):
                minLoopValues[loop.type.value] = RowCol(loop.size.row, loop.size.col)
            else:
                minLoopValues[loop.type.value] = loop.size
                
        minLoopValues[LoopType.dx.value] = layer.dx
        minLoopValues[LoopType.dy.value] = layer.dy
            
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
            res, currLoopIndex = incrementTiling(tiledArch, currGroupSliceIndexes, layer.getMaxLoopSizeByIndex(currGroup), minLoopValues[currGroup])

            if res == IncTileResult.LoopOverflow:
                currGroup += 1
                continue
                
            if res == IncTileResult.LoopIncremented:
                iterationCounter += 1

                hwTemplate.archP = tiledArch[:len(hwTemplate.archP)]
                currBuff = hwTemplate.calcBuffers(tiledArch[len(hwTemplate.archP):], layer)

                if not buffFitMem(currBuff, hwRestrictions):
                    updateTilingOnBuffOverflow(tiledArch, loopIndexByType[0:currGroup+1], currLoopIndex)
                    currGroup = 0
                    continue
                
                currEnergy = hwTemplate.calcEnergy(currBuff, energyModel, layer)
#                printTile(hwTemplate.archU +tiledArch, 3, currEnergy, sum(currBuff.Bin)+sum(currBuff.Bkern)+sum(currBuff.Bout))
#                    print currEnergy
                if currEnergy < energyBest:
                    energyBest = currEnergy
                    del buffBest
                    buffBest = currBuff
                    tilingBest = copyTilingBest(hwTemplate.archU + tiledArch)
                else:
                    del currBuff
            
                currGroup = 0
                continue
            
    print "Total iterations ", iterationCounter
#    printTile(tilingBest, 2)
#    print buffBest
#    print energyBest
                
    return tilingBest, buffBest, energyBest

def buffFitMem(buff, hwRestrictions):
    return buff.totalSpace() <= hwRestrictions.memory[0] * hwRestrictions.memory[1]
    
def copyTilingBest(tile):
    best = []
    for loop in tile:
        if (loop.type == LoopType.rowcol):
            best.append(Loop(loop.type, RowCol(loop.size.row, loop.size.col), loop.pragma))
        else:
            best.append(Loop(loop.type, loop.size, loop.pragma))              
    return best
        
    

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

if len(sys.argv) == 5:
    sys.stdout = open(sys.argv[4], 'w')
    
##
#AlexNet1 = Layer(name="AlexNet1", X=227, Y=227, fm=3, kern=46, dx=3, dy=3, stride=1)
#AlexNet2 = Layer("AlexNet2", X=55, Y=55, fm=48, kern=256, dx=5, dy=5, stride=1)
##
    
zinq=HWrestrictions((int(sys.argv[3]),1), 256)

if sys.argv[1] == 'CNP':
    template = CNP()
elif sys.argv[1] == 'CNPfm':
    template = CNPfm()
elif sys.argv[1] == 'Eyeriss':    
    template = Eyeriss()
elif sys.argv[1] == 'DNNweaver':    
    template = DNNweaver()
else:
    raise Exception('''
Wrong template name
Check HwTemplaty.py
Currently implemented: CNP, CNPfm,Eyeriss''')
    
if sys.argv[2] == 'AlexNet1':
    layer = AlexNet1()
elif sys.argv[2] == 'AlexNet2':
    layer = AlexNet2()
elif sys.argv[2] == 'AlexNet3':
    layer = AlexNet3()    
else:
    raise Exception('''
Wrong layer name
Check Layer.py''')
    
optimizer = TemplateOptimizer(template, layer, zinq)
optimizer.Optimize()

    
############################################
##### Manual testing of a single string#####    
############################################

#template = DiaNNao()
#template.archU = [Loop(LoopType.fm, 4, Pragma.u), Loop(LoopType.kern, 4, Pragma.u)]
#template.archU = [Loop(LoopType.dx, 5, Pragma.u), Loop(LoopType.dy, 5, Pragma.u), Loop(LoopType.kern, 4, Pragma.u )]

#template = CNP()
#template.archU = [Loop(LoopType.dx, 5, Pragma.u), Loop(LoopType.dy, 5, Pragma.u)]
#template.archP = [Loop(LoopType.rowcol, RowCol(5,14), Pragma.p)]
#
#template = CNPfm()
#template.archU = [Loop(LoopType.dx, 5, Pragma.u), Loop(LoopType.dy, 5, Pragma.u), Loop(LoopType.fm, 10, Pragma.u)]
#template.archP = [Loop(LoopType.rowcol, RowCol(7,11), Pragma.p)]
#
#template = Eyeriss()
#template.archU = [Loop(LoopType.dy, 5, Pragma.u), Loop(LoopType.rowcol, RowCol(28,1), Pragma.u), Loop(LoopType.kern, 1, Pragma.u)]
#template.archP =[Loop(LoopType.rowcol, RowCol(28,1), Pragma.p), Loop(LoopType.dx, 5, Pragma.p), Loop(LoopType.fm, 3, Pragma.p)]

#template = DNNweaver()
#template.archU = [Loop(LoopType.rowcol, RowCol(1,15), Pragma.u), Loop(LoopType.kern, 16, Pragma.u)]
#template.archP =[Loop(LoopType.dx, 5, Pragma.p), Loop(LoopType.dy, 5, Pragma.p), Loop(LoopType.rowcol, RowCol(5,55), Pragma.n)]
#
#
##from test import divisorsCeil
##for ii1 in divisorsCeil(1, 55):
##    for i1 in divisorsCeil(1, 55):
##        
##        for i in divisorsCeil(1, i1):
##            for ii in divisorsCeil(1, ii1):
#
#
#
#tiling = [            
#            Loop(LoopType.kern, 64, Pragma.n),                
#            Loop(LoopType.fm, 8, Pragma.n),
#Loop(LoopType.rowcol, RowCol(55,55), Pragma.n),
#             
#            Loop(LoopType.rowcol, RowCol(55,55), Pragma.n),
#            Loop(LoopType.kern, 256, Pragma.n),
#            Loop(LoopType.fm, 48, Pragma.n)
#
#          ]
#
#args = OptArgs(tiling, template, layer, EnergyModel(200, 6 ,1 ,1), zinq)
#OptimizeTile(args)
#
#buff = template.calcBuffers(tiling, layer)  
#
#currEnergy = template.calcEnergy(buff, EnergyModel(), layer)
#printTile(template.archU + template.archP + tiling,1)
#print buff
##print (ii, i), (ii1, i1), currEnergy, sum(buff.Bin)+sum(buff.Bkern)+sum(buff.Bout)
#print currEnergy, sum(buff.Bin)+sum(buff.Bkern)+sum(buff.Bout)