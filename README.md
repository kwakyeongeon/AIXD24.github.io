# Optimizing EV Charging Station Placement with Deep Learning

## Members
- 곽연건, 한양대학교 서울캠퍼스 기계공학부, kwak6072@hanyang.ac.kr
- 정우진, 한양대학교 서울캠퍼스 기계공학부

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Motivation](#motivation)  
3. [Datasets](#datasets)  
4. [Methodology](#methodology)  
5. [Evaluation & Analysis](#evaluation--analysis)  
6. [Conclusion](#conclusion)  
7. [References](#references)

---

## 1. Introduction
전기차는 지속 가능한 교통수단으로 점점 더 중요한 요소가 되고 있습니다.  
효율적인 전기차 충전소 배치는 사용자 편의를 높이고 인프라 활용도를 극대화하는 데 필수적입니다.  
본 프로젝트는 딥러닝을 활용하여 기존 충전소 데이터와 전기차 분포 데이터를 분석해 새로운 충전소의 최적 위치를 제안합니다.

---

## 2. Motivation
### Why this project?  
- 전기차 보급률이 증가함에 따라 더 나은 인프라 계획이 필요합니다.  
- 기존 충전소가 지역별로 고르게 배치되지 않아 접근성이 떨어집니다.  
- 공간 데이터와 전기차 분포 데이터를 결합하여 새로운 충전소의 최적 위치를 도출할 수 있습니다.

### Objective
사용 패턴, 지역 특성, 전기차 보급률 데이터를 기반으로 새로운 충전소의 최적 위치를 제안하는 모델을 개발하는 것이 목표입니다.

---

## 3. Datasets
### 1. **EV Charging Station Data**
- **출처**: [한국환경공단](https://www.data.go.kr/data/15076352/openapi.do)에서 제공된 공공 데이터  
- **설명**: 설치 연도, 위치, 충전 속도, 충전기 유형 등의 정보를 포함합니다.  
- **주요 속성**:  
  - `설치년도`, `시도`, `군구`, `충전기 타입`, `위도/경도`  

### 2. **Regional EV Distribution Data**
- **출처**: 지역별 전기차 대수 데이터  
- **설명**: 각 지방자치단체의 전기차 대수를 월별로 제공합니다.  
- **주요 속성**:  
  - `기준일`, `시도`, `전기차 대수`

---

## 4. Methodology
### Data Preprocessing
1. **데이터 병합**  
   - `EV Charging Station Data`와 `Regional EV Distribution Data`를 지역 정보를 기준으로 병합.  
2. **결측치 처리**  
   - `pandas`를 활용하여 결측치를 보완하거나 제거.  
3. **정규화**  
   - 전기차 대수, 기존 충전소 밀도와 같은 수치형 데이터를 정규화.  
4. **지리적 시각화**  
   - 위도/경도 데이터를 활용해 지도 상에 충전소와 전기차 분포를 시각화.

### Model Selection
- **Deep Neural Network (DNN)**:  
  - 공간적 데이터와 통계 데이터를 결합하여 최적의 위치를 예측합니다.  
  - LSTM을 통해 시간에 따른 전기차 분포를 예측.  

### Training
1. **입력 특징**:  
   - 지역별 전기차 대수, 기존 충전소와의 거리, 충전기 타입  
2. **출력 결과**:  
   - 새로운 충전소의 최적 위도/경도 좌표  

### Evaluation Metrics
- **지리적 예측 정확도**: Mean Absolute Error (MAE)  
- **시각적 분석**: 히트맵을 통해 충전소가 부족한 지역을 시각화.  

---

## 5. Evaluation & Analysis
### 1. 주요 결과
- **지역별 분석**:  
  - 전기차 대수는 많지만 충전소가 적은 지역이 최적의 위치로 나타남.  
  - 교통량이 많은 도심 지역은 시골 지역보다 고속 충전소가 더 많이 필요함.  

### 2. 결과 시각화
- **히트맵**: 충전소가 부족한 지역을 지도에 시각화.  
- **산점도**: 전기차 대수와 충전소 밀도 간의 상관관계 분석.  

---

## 6. Conclusion
- **성과**:  
  - 데이터를 기반으로 충전소 배치 문제를 해결할 수 있는 모델을 제시.  
  - 지역별 전기차 대수와 기존 충전소 데이터를 활용하여 설득력 있는 결과 도출.  
- **한계점**:  
  - 추가적인 실제 사용 데이터가 필요.  
  - 충전소 설치 비용과 같은 비정형 데이터의 부족.  
- **향후 연구 방향**:  
  - 충전소 유형별 최적 배치 연구  
  - 사용자 이동 패턴을 고려한 실시간 추천 시스템  

---

## 7. References
- [한국환경공단 데이터](https://www.data.go.kr/data/15076352/openapi.do)  
- [Kaggle](https://www.kaggle.com/)  
- 딥러닝 관련 논문 및 참고 자료  
- Python 라이브러리: `pandas`, `numpy`, `tensorflow`, `matplotlib`  


