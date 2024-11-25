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

2. **전기차 등록 대수 데이터**  
   - **출처**: 공공데이터 API  
   - **설명**: 각 지역별 전기차 등록 대수와 연도별 증가 추이를 포함.  
   - **주요 변수**:  
     - `기준일`, `시도`, `전기차 대수`.  

---

## **III. Methodology**

### **Algorithms and Models**
- **지도 학습 기반 딥러닝 모델**:  
  - **LSTM**: 시간에 따른 전기차 등록 대수의 증가 추이 예측.  
  - **CNN**: 지리적 패턴 분석 및 충전소의 최적 위치 선정.  
- **예측 모델 구성 및 훈련 과정**:  
  - 기존 데이터셋에서 충전소가 부족한 지역 탐지.  
  - 새로운 충전소 위치를 예측하기 위한 회귀 분석 모델 훈련.  

### **Features**
- 주요 피처:  
  - 지역별 전기차 대수.  
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
- [Kaggle: EV 충전소 데이터](https://www.kaggle.com/)  
- 딥러닝 관련 논문 및 참고 자료.  


