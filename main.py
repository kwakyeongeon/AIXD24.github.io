import numpy as np
from modules.modules import build_model
from modules.utils import data_preprocess, plot_history, plot_results, cluster, adjust

if __name__ == '__main__':

    data_path = './data/ev_charging_station_data.csv'

    # 데이터 전처리    
    train_x, test_x = data_preprocess(data_path)

    # 모델 빌드 및 학습
    model = build_model()
    history = model.fit(train_x, train_x, epochs=100, batch_size=32, validation_data=(test_x, test_x))
    
    # 모델 결과 출력
    plot_history(history)
    plot_results(train_x, test_x, model)

    # 데이터셋 군집화 
    cluster(train_x, test_x, model)
    adjust(test_x, model)

    # 랜덤한 더미 데이터 삽입 및 위치 보정 

    # 데이터의 분포를 통해 사람이 전기차 충전소의 대략적인 위치를 입력해주면 기존 데이터의 분포를 고려하여 적절한 위치로 보정해줌 
    