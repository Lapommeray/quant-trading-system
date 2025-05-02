
import numpy as np

class GhostIntentEngine:
    def __init__(self):
        self.cancel_heatmap = np.zeros((100, 100))  # Price/time grid

    def track_canceled_orders(self, orderbook):
        for cancel in orderbook.cancel_stream:
            x = int(cancel.price % 100)
            y = int(cancel.time_delay // 10)
            self.cancel_heatmap[x][y] += cancel.amount
        return np.argwhere(self.cancel_heatmap > 50)
