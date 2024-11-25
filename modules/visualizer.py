import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

def configure_korean_fonts():
    """
    한글 폰트를 설정합니다.
    """
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)

def plot_charging_station_deficit(merged_data, output_path="charging_station_deficit_updated.png"):
    """
    충전소 부족률을 시각화합니다.
    """
    plt.figure(figsize=(12, 6))
    plt.bar(merged_data["시도"], merged_data["충전소 부족률"], color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel("충전소 부족률")
    plt.title("지역별 충전소 부족률")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
