import math
 
# RRout is multiplied by 2 in the main optimizer,
# as we may have multiple component in RR (dx, dy, kern)


# TODO: this is true only for tiles - the whole layer will not have dx,dy component
def kernBuff(row, dx , col, dy, fm):
   return (row + dx - 1) * (col + dy - 1) * fm
   
def kernRR( kern1, kern0, dx=1, dy=1, row=1, col=1):
    return int(math.ceil( float(kern1) / kern0 ) )#* \
            #dy * dx * (col * row) / ((row + dy - 1.0) * (col + dx - 1.0) )

def dxdyBuff(col, dx, row, dy, fm, kern):
    return ( (row + dy + 1) * (col + dx + 1) * fm,
            row * col * kern )

#def dxBuff(col, dx, row, dy, fm, kern):
#    return _dBuff(col, dx, row, dy, fm, kern)
#            
#def dyBuff(col, dx, row, dy, fm, kern):
#    return _dBuff(col, dx, row, dy, fm, kern)
    
def dxRR(dx1, dx0, col):
    rr = math.ceil(dx1 / float(dx0))
    return ( int(rr * col / (col + dx1 - 1)),
            int(rr))
            
def dyRR(dy1, dy0, row):
    rr = math.ceil(dy1 / float(dy0))
    return ( rr * row / (row + dy1 - 1),
            rr)
            
def fmBuff(row, col, kern):
    return row * col * kern
    
def fmRR(fm1, fm0):
    return int(math.ceil(fm1 / float(fm0)))
    
def rowcolBuff(kern, fm, dx, dy):
    return kern * fm * dx * dy
    
def rowRR(row1, row0):
    return int(math.ceil(row1 / float(row0)))
    
def colRR(col1, col0):
    return int(math.ceil(col1 / float(col0)))
