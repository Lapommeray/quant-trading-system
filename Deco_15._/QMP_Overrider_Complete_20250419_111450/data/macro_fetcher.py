
import requests

FRED_API_KEY = 'ea67abdfefec0dc91da0a0d6219f6c08'
BASE_URL = 'https://api.stlouisfed.org/fred/series/observations'

def get_latest_macro_value(series_id):
    try:
        url = f"{BASE_URL}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json&sort_order=desc&limit=1"
        response = requests.get(url)
        data = response.json()
        obs = data['observations'][0]
        return float(obs['value'])
    except Exception as e:
        print(f"Error fetching {series_id}:", e)
        return None

def get_macro_snapshot():
    return {
        "CPI": get_latest_macro_value('CPIAUCSL'),              # Inflation
        "M2_Money_Supply": get_latest_macro_value('M2SL'),      # Liquidity
        "Fed_Funds_Rate": get_latest_macro_value('FEDFUNDS'),   # Interest Rate
        "Unemployment": get_latest_macro_value('UNRATE'),       # Job Market
    }

def is_macro_environment_favorable(thresholds=None):
    if thresholds is None:
        thresholds = {
            "CPI": 4.0,              # Inflation below this
            "Fed_Funds_Rate": 5.0,   # Interest rates below this
            "Unemployment": 6.0      # Unemployment below this
        }

    snapshot = get_macro_snapshot()
    checks = [
        snapshot["CPI"] is not None and snapshot["CPI"] < thresholds["CPI"],
        snapshot["Fed_Funds_Rate"] is not None and snapshot["Fed_Funds_Rate"] < thresholds["Fed_Funds_Rate"],
        snapshot["Unemployment"] is not None and snapshot["Unemployment"] < thresholds["Unemployment"],
    ]

    return all(checks)
