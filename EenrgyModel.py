class EnergyModel(object):
    def __init__(self, DRAM=200, buff=6, RF=1, ALU=1):
        self.DRAM = DRAM
        self.buff = buff
        self.RF = RF
        self.ALU = ALU