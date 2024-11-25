from modules.data_loader import load_charging_data, load_ev_distribution_data
from modules.data_preprocessor import preprocess_ev_distribution_data, standardize_region_names, merge_data
from modules.visualizer import configure_korean_fonts, plot_charging_station_deficit

# 데이터 경로
charging_data_path = "data/ev_charging_station_data.csv"
ev_distribution_data_path = "data/ev_distribution_data.csv"

# 데이터 로드
charging_data = load_charging_data(charging_data_path)
ev_distribution_data = load_ev_distribution_data(ev_distribution_data_path)

# 데이터 전처리
ev_distribution_data = preprocess_ev_distribution_data(ev_distribution_data)
charging_data = standardize_region_names(charging_data, "시도")

# 데이터 병합
merged_data = merge_data(ev_distribution_data, charging_data)

# 결과 출력
print("Missing Values:\n", merged_data.isnull().sum())
print("Summary Statistics:\n", merged_data.describe())

# 시각화
configure_korean_fonts()
plot_charging_station_deficit(merged_data)
