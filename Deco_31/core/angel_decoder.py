
def is_angel_number_present(price_data):
    """
    Checks if close price contains any divine sequences like 111, 888, 369, etc.
    """
    close_price = str(round(price_data['close'], 2)).replace('.', '')

    patterns = ['111', '222', '333', '369', '444', '555', '666', '777', '888', '999']
    return any(pat in close_price for pat in patterns)

def get_angelic_alignment_score(price_data):
    """
    Returns 1.0 if strong divine presence detected, else 0.0
    """
    return 1.0 if is_angel_number_present(price_data) else 0.0
