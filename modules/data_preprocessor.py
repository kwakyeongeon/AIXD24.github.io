import pandas as pd
import numpy as np

def standardize_region_names(df, region_col):
    """
    지역 이름을 표준화합니다.
    """
    region_mapping = {
        "강원": "강원특별자치도",
        "경기": "경기도",
        "경남": "경상남도",
        "경북": "경상북도",
        "광주": "광주광역시",
        "대구": "대구광역시",
        "대전": "대전광역시",
        "부산": "부산광역시",
        "서울": "서울특별시",
        "세종": "세종특별자치시",
        "울산": "울산광역시",
        "인천": "인천광역시",
        "전남": "전라남도",
        "전북": "전라북도",
        "전북특별자치도": "전라북도",
        "제주": "제주특별자치도",
        "충남": "충청남도",
        "충북": "충청북도"
    }
    df[region_col] = df[region_col].replace(region_mapping)
    return df

def preprocess_ev_distribution_data(df):
    """
    전기차 등록 대수 데이터를 전처리합니다.
    """
    df = df.melt(id_vars=["기준일"], var_name="시도", value_name="전기차 대수")
    return standardize_region_names(df, "시도")

def merge_data(ev_data, charging_data):
    """
    전기차 등록 대수와 충전소 데이터를 병합합니다.
    """
    charging_counts = charging_data.groupby("시도").size().reset_index(name="충전소 수")
    merged = ev_data.groupby("시도")["전기차 대수"].sum().reset_index()
    merged = pd.merge(merged, charging_counts, on="시도", how="left").fillna(0)

    # 충전소 부족률 계산
    merged["충전소 부족률"] = np.where(
        merged["충전소 수"] == 0, np.inf, merged["전기차 대수"] / merged["충전소 수"]
    )
    return merged
