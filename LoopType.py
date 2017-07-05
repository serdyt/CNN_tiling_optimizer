from enum import Enum
from recordtype import recordtype

Loop = recordtype("Loop", ['type', 'size', 'pragma'])

RC = recordtype("RowCol", ['row', 'col'])

class RowCol(RC):
    def __init__(self, row, col):
        super(RowCol, self).__init__(row, col)
        
#    def __lt__(self, other):
#        if (self.col < other.col and self.row < other.row):
#            return True
#        else:
#            return False
            
    def __str__(self):
        return str(self.row) + "," + str(self.col)
            
class LoopType(Enum):
    rowcol = 0
    kern = 1
    fm = 2
    dx = 3
    dy = 4   