# **Optimizing EV Charging Station Placement with Deep Learning**

---

## **Contents**
1. [Members](#members)  
2. [I. Proposal](#i-proposal)  
    - [Motivation](#motivation)  
    - [Objective](#Objective)  
3. [II. Datasets](#ii-datasets)  
    - [Dataset Description](#dataset-description)  
4. [III. Methodology](#iii-methodology)  
    - [Algorithms and Models](#algorithms-and-models)  
    - [Features](#features)  
5. [IV. Evaluation & Analysis](#iv-evaluation--analysis)  
    - [Evaluation Metrics](#evaluation-metrics)  
    - [Visualization](#visualization)  
6. [V. Related Work](#v-related-work)  
7. [VI. Conclusion](#vi-conclusion)  

---

## **Members**
- 곽연건, 한양대학교 서울캠퍼스 기계공학부, kwak6072@hanyang.ac.kr  
- 정우진, 한양대학교 서울캠퍼스 기계공학부  

---

## **I. Proposal**
![1](https://github.com/user-attachments/assets/4f517007-24a3-4ad2-b00e-3efca4e26ff2)

<사진출처-현대자동차>
### **Motivation**
전기차는 전 세계적으로 빠르게 보급되고 있으나, 충전소 배치의 효율성은 여전히 전기차 사용자들의 요구를 충분히 충족시키지 못하고 있습니다.
충전소가 잘못 배치되면 사용자 접근성이 낮아지고, 충전 대기 시간이 길어지며, 지역 간 충전 인프라의 불균형 문제가 발생할 수 있습니다.
효율적인 충전소 배치는 전기차 사용자의 편의성을 높일 뿐만 아니라 에너지 자원을 효율적으로 활용하는 데도 기여할 수 있습니다.
이 프로젝트는 전기차 사용량 증가와 충전소 배치 문제를 데이터 기반으로 해결하려는 시도로, 다음을 목표로 합니다.

전기차 보급률 분석
지역별 전기차 분포 데이터와 기존 충전소 위치 데이터 수집 및 분석
이를 통해 전기차 사용자 경험을 개선하고, 충전소 배치의 효율성을 극대화할 수 있는 최적의 충전소 위치를 제안하고자 합니다.  

### **Objective**
- **1. 전기차 등록 대수와 충전소 분포 데이터를 결합하여 부족 지역 식별**  
  전기차 등록 대수와 현재 충전소 분포 데이터를 분석하여 충전소가 부족한 지역을 효과적으로 식별합니다. 이를 통해 충전소 배치의 불균형 문제를 구체적으로 파악할 수 있습니다.  

- **2. 딥러닝 모델을 활용한 최적 배치 방안 제안**  
  딥러닝 기술을 활용해 충전소 배치를 최적화할 수 있는 방안을 제안합니다. 모델은 지역 특성, 사용자 밀도, 교통 흐름 등 다양한 변수를 종합적으로 고려하여 데이터 기반의 예측 결과를 제공합니다.  

- **3. 데이터 시각화를 통한 설득력 있는 결과 제공**  
  분석 결과를 직관적으로 이해할 수 있도록 시각화 자료를 제작합니다. 이를 통해 사용자와 이해관계자들에게 설득력 있는 인사이트를 전달하며, 향후 충전소 인프라 개선을 위한 기초 자료로 활용합니다.  

---

## **II. Datasets**

### **Dataset Description**
1. **전기차 충전소 데이터**  
   - **출처**: [한국환경공단 공공데이터](https://www.data.go.kr/)  
   - **설명**: 설치년도, 위치, 충전 속도, 충전기 타입 등의 정보를 포함.  
   - **주요 변수**:  
     - `설치년도`, `시도`, `군구`, `위도/경도`, `충전기 타입`, `시설 구분`.

    ev_charging_station_data.csv (82176KB)

2. **전기차 등록 대수 데이터**  
   - **출처**: 공공데이터 API  
   - **설명**: 각 지역별 전기차 등록 대수와 연도별 증가 추이를 포함.  
   - **주요 변수**:  
     - `기준일`, `시도`, `전기차 대수`.

    ev_distribution_data.csv (2KB)

   ### **Data Preprocessing**
데이터는 다양한 전처리 과정을 거쳐 분석 가능한 형태로 가공되었습니다.

---

#### **Code 1: 데이터 로드**
```python
import pandas as pd

def load_charging_station_data(data_path):
    """
    충전소 데이터를 로드합니다.
    """
    return pd.read_csv(data_path, header=None).dropna()

```

#### **코드 설명**

load_charging_station_data 함수는 충전소 데이터를 CSV 파일에서 불러옵니다.
데이터에 결측값이 존재하면 이를 제거하여 데이터의 완전성을 유지합니다.


#### **Code 2: 데이터 전처리 및 스케일링**
```python

from sklearn.preprocessing import MinMaxScaler

def data_preprocess(data_path, train_ratio=0.8):
    """
    데이터 전처리 및 스케일링 수행.
    """
    data = pd.read_csv(data_path, header=None).dropna()
    location = data.iloc[1:].to_numpy()

    # 위도와 경도 추출
    inputs = []
    for record in location:
        lat = float(record[-1].split(',')[0])
        lon = float(record[-1].split(',')[1])
        inputs.append([lat, lon])
    inputs = np.array(inputs)

    # 데이터 스케일링
    scaler = MinMaxScaler()
    inputs = scaler.fit_transform(inputs)

    # 학습 및 테스트 데이터 분리
    train_x = inputs[:int(inputs.shape[0] * train_ratio)]
    test_x = inputs[int(inputs.shape[0] * train_ratio):]

    return train_x, test_x

```
#### **코드 설명**

data_preprocess 함수는 충전소 데이터에서 위도와 경도를 추출하여 전처리합니다.
MinMaxScaler를 사용하여 데이터를 0과 1 사이로 스케일링하여 학습 안정성을 높입니다.
데이터는 80%를 학습용, 20%를 테스트용으로 나누어 모델 학습과 평가를 분리합니다.

#### **Code 3: 지역 이름 통합 및 충전소 부족률 계산**
```python

import matplotlib.pyplot as plt

def visualize_data_distribution(data):
    """
    데이터 분포 시각화.
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 1], data[:, 0], s=10, color="blue", label="Data Points")
    plt.title("Geographic Distribution of Charging Stations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.savefig('./results/data_distribution.png')
    plt.show()

```
#### **코드 설명**

visualize_data_distribution 함수는 데이터 분포를 시각화하여 충전소의 지리적 위치를 확인합니다.
파란색 점은 각 충전소의 위치를 나타내며, 이를 통해 충전소의 밀집 및 부족 지역을 식별할 수 있습니다.

#### **결론**

데이터셋 전처리는 본 프로젝트의 핵심 단계로, 다음과 같은 작업을 수행했습니다.

충전소 데이터와 전기차 등록 데이터를 로드 및 전처리.
위도와 경도 정보를 추출하여 스케일링 후 학습용/테스트용으로 분리.
데이터의 지리적 분포를 시각화하여 추가 분석과 모델 학습에 필요한 인사이트를 도출.
이를 통해 전기차 충전소 부족 문제를 해결할 기반 데이터를 성공적으로 구축했습니다.


---

## **III. Methodology**

### **Algorithms and Models**

본 프로젝트에서는 전기차 충전소 최적 배치를 위해 Autoencoder를 기반으로 한 딥러닝 모델을 사용하였습니다. 
Autoencoder는 입력 데이터를 잠재 공간(latent space)으로 압축한 뒤, 
이를 복원하는 과정을 통해 데이터의 특징을 학습하는 비지도 학습 모델입니다. 
이 모델은 다음과 같은 두 가지 주요 컴포넌트로 구성됩니다:

1.Encoder 
Encoder는 입력 데이터를 저차원 잠재 공간(latent space)으로 압축합니다. 이 과정에서 데이터의 핵심적인 특징만 유지됩니다.

2. Decoder
Decoder는 잠재 공간의 표현(latent representation)을 기반으로 원래 입력 데이터를 복원합니다. 이를 통해 모델은 입력 데이터를 압축하고 복원하는 과정을 반복 학습하여 데이터의 구조를 학습합니다.

Model Architecture
Code Implementation

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():
    """
    Autoencoder 모델 정의
    """
    # Encoder
    latent_dim = 2
    encoder_input = layers.Input(shape=(2,))
    encoded = layers.Dense(64, activation='relu')(encoder_input)
    latent = layers.Dense(latent_dim, activation='linear')(encoded)
    encoder = models.Model(encoder_input, latent, name="encoder")

    # Decoder
    decoder_input = layers.Input(shape=(latent_dim,))
    decoded = layers.Dense(64, activation='relu')(decoder_input)
    output = layers.Dense(2, activation='linear')(decoded)
    decoder = models.Model(decoder_input, output, name="decoder")

    # Autoencoder
    autoencoder_input = layers.Input(shape=(2,))
    encoded_latent = encoder(autoencoder_input)
    decoded_output = decoder(encoded_latent)
    autoencoder = models.Model(autoencoder_input, decoded_output, name="autoencoder")

    # 모델 컴파일
    autoencoder.compile(optimizer='adam', loss='mse', metrics=["mae"])
    return autoencoder
```

Why Autoencoder?
Autoencoder는 다음과 같은 이유로 본 프로젝트에 적합합니다:

공간 데이터의 학습: Autoencoder는 전기차 충전소의 위도와 경도 데이터를 학습하여 주요 패턴을 압축적으로 표현할 수 있습니다.
결손 데이터 예측: Autoencoder는 학습된 패턴을 기반으로 부족한 충전소 위치를 복원하거나 예측하는 데 유용합니다.
군집화와 시각화: Latent space를 기반으로 충전소 위치를 군집화하고, 이를 분석하여 최적의 위치를 제안할 수 있습니다.
Training Process
Data Preprocessing:

데이터는 위도와 경도 값으로 구성되며, 이를 MinMaxScaler를 사용하여 스케일링.
학습 데이터와 테스트 데이터를 80:20 비율로 분리.
Model Training:

mean squared error (MSE)와 mean absolute error (MAE)를 손실 함수로 사용하여 학습.
100 epochs 동안 학습하며, 학습 및 검증 손실 추이를 시각화.
Evaluation:

Autoencoder의 복원 결과를 테스트 데이터에 대해 평가.
Latent space를 기반으로 새로운 충전소 위치를 추천.



 ```python

def build_model():
    latent_dim = 2
    encoder_input = layers.Input(shape=(2,))
    encoded = layers.Dense(64, activation='relu')(encoder_input)
    latent = layers.Dense(latent_dim, activation='linear')(encoded)

    encoder = models.Model(encoder_input, latent, name="encoder")
    
    decoder_input = layers.Input(shape=(latent_dim,))
    decoded = layers.Dense(64, activation='relu')(decoder_input)
    output = layers.Dense(2, activation='linear')(decoded)

    decoder = models.Model(decoder_input, output, name="decoder")

    autoencoder_input = layers.Input(shape=(2,))
    encoded_latent = encoder(autoencoder_input)
    decoded_output = decoder(encoded_latent)

    autoencoder = models.Model(autoencoder_input, decoded_output, name="autoencoder")
    autoencoder.compile(optimizer='adam', loss='mse', metrics=["mae"])
    return autoencoder

```

설명: 이 모델은 Autoencoder를 통해 데이터를 잠재 공간으로 변환하여 최적의 충전소 위치를 예측합니다.

KMeans 군집화

지역 데이터를 군집화하여 고밀도 지역과 저밀도 지역을 구분합니다.



### **Features**
- 주요 피처:  
  - 기존 충전소와의 거리.  
  - 충전소 밀도 및 유형.  
  - 위도/경도.  

---

## **IV. Evaluation & Analysis**

### **Evaluation Metrics**
- **RMSE (Root Mean Square Error)**: 위치 예측 정확도 측정.  
- **R² (결정계수)**: 모델의 설명력을 평가.  

### **Visualization**
- **히트맵**:  
  - 기존 충전소 밀도를 시각적으로 표현하여 충전소가 부족한 지역을 식별.  
- **시계열 그래프**:  
  - 전기차 대수의 시간적 증가 추이를 보여줌.  

---

## **V. Related Work**

### **참조한 문헌 및 도구**
- **도구**: Python (`pandas`, `numpy`, `TensorFlow`, `matplotlib`), Google Colab.  
- **관련 연구 및 블로그**:  
  - [Kaggle의 EV 충전소 데이터 분석 예제](https://www.kaggle.com/)  
  - [논문: 전기차 인프라와 충전소 최적 배치](https://scholar.google.com/)  

---

## **VI. Conclusion**

### **Discussion**
- 충전소 최적 배치의 경제적, 환경적 이점 논의.  
- 데이터 기반 인프라 계획의 가능성을 보여줌.  
- 연구의 한계:  
  - 추가적인 충전소 사용 패턴 데이터 필요.  
  - 설치 비용, 사용자 편의성 같은 비정형 데이터 활용 필요.  

### **Future Work**
- 충전소 유형별 최적 배치 연구.  
- 사용자 이동 데이터를 추가하여 실시간 충전소 추천 시스템 구축.  

---

## **References**
- [한국환경공단 공공데이터](https://www.data.go.kr/)  
- [TensorFlow](https://kr.mathworks.com/discovery/autoencoder.html)
- [MathWorks](https://www.tensorflow.org/tutorials/generative/autoencoder?hl=ko)
- 

