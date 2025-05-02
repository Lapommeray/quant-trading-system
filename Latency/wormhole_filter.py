
def get_clock_drift(binance_time, cme_time):
    drift = abs(binance_time - cme_time)
    if drift > 50_000:
        return "UNSAFE"
    return "SAFE"

def execute_if_synced(exchange, trade, binance_time, cme_time):
    if get_clock_drift(binance_time, cme_time) == "SAFE":
        exchange.execute(trade)
    else:
        trade.void()
