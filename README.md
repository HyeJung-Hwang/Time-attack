# Hypotension prediction using mean blood pressure values using Deep Learning

## Overview
설명

## Our Goal
설명

## Deep Learning-based  Hypotension prediction
설명



## Raw Data
### 1️⃣ vital db 어쩌구저쩌구
### 2️⃣ 데이터셋 붉균형이 심하다

## making sequence data

설명


## Models and training details

- MLP
- LSTM
- 된다면 cnn

## Results

|제목|auroc|auprc|recall|precision|f1-score|
|------|---|---|---|---|---|
|mlp|테스트2|테스트3|테스트3|테스트3|테스트3|
|lstm|테스트2|테스트3|테스트3|테스트3|테스트3|
|cnn|테스트2|테스트3|테스트3|테스트3|테스트3|
  
## Conclusion & Discussion

낮은 모델의 성능에 대해 분석해본 결과 , 제 application 의 3가지 문제점을 낮은 모델의 성능으로 꼽을 수 있었습니다.

첫 번째로 , Data Imbalance Problem 을 완전히 해결하지 못했습니다. Random Over Sampling 외에 Loss Ratio를 조절하는 방법을 사용해보고자 합니다.

두 번째로 , 학습 Feature 개수의 부족입니다. 본 application 과 유사하게 ,  Bio-Signals를 기반으로 Acute Kidney Injury를 예측하는 타 논문의 경우 , 78 개의 feature를 사용한 것을 참고하여 , 추가 feature 를 사용하고자 합니다.

세 번째로 , vital sign 의 시계열적 특성을 살리기 위한 딥러닝 approach 를 사용해보고자합니다.

## Reference 
- [Predicting Acute Kidney Injury via Interpretable
Ensemble Learning and Attention Weighted
Convoutional-Recurrent Neural Networks]( https://engineering.jhu.edu/nsa/wp-content/uploads/2021/02/YPeng_CISS_2021_preprint.pdf )
- [Vital DB Examples ]( https://github.com/vitaldb/examples/blob/master/comments_in_Korean/mbp_aki.ipynb )


