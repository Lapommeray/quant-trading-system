# Stores and recalls patterns across asset+time layers
class MultiDimensionalMarketMemory:
    def __init__(self): self.memory = {}
    def store(self, symbol, tf, pattern):
        self.memory[(symbol, tf)] = pattern
    def recall(self, symbol, tf):
        return self.memory.get((symbol, tf), None)
