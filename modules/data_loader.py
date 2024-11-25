import pandas as pd

def load_charging_data(filepath):
    """
    전기차 충전소 데이터를 로드합니다.
    """
    return pd.read_csv(filepath, encoding='utf-8')

def load_ev_distribution_data(filepath):
    """
    지역별 전기차 등록 대수 데이터를 로드합니다.
    """
    return pd.read_csv(filepath, encoding='utf-8')
