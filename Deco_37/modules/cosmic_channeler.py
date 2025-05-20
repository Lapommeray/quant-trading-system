
import datetime
import math

def get_moon_phase(date):
    """
    Returns a value between 0.0 (New Moon) to 1.0 (Full Moon)
    """
    diff = date - datetime.datetime(2001, 1, 1)
    days = diff.days + (diff.seconds / 86400)
    lunations = 0.20439731 + (days * 0.03386319269)
    return lunations % 1

def is_planetary_gate_open(date):
    # Planetary gate opens on these spiritually aligned days
    return date.day in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def get_cosmic_alignment_score(current_time):
    moon_phase = get_moon_phase(current_time)
    gate_open = is_planetary_gate_open(current_time)

    # Favorable when close to Full or New Moon
    moon_alignment = 1.0 - abs(moon_phase - 0.5) * 2

    # Gate bonus
    bonus = 0.1 if gate_open else 0.0

    return round(min(1.0, moon_alignment + bonus), 3)
