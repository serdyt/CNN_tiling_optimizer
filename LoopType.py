from enum import Enum
from recordtype import recordtype

Loop = recordtype("Loop", ['type', 'size', 'pragma'])

class LoopType(Enum):
    fm = 0
    kern = 1
    col = 2
    row = 3
    dx = 4
    dy = 5