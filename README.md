# 🩸 Hypotension Prediction using Mean Blood Pressure 


## Overview
본 웹사이트는 수술 중 저혈압 예측 모델을 활용하여, 특정 환자의 수술 중 저혈압 지표를 시각화 한 웹사이트입니다. 특정 환자의 case_id를 선택하면, 해당 환자의 시간에 따른 수술 중 Mean Blood Pressure 수치와 LSTM 모델로 예측한 수술 중 저혈압 발생 구간을 확인 할 수 있습니다.  

저혈압 발생 예측의 경우, 20초 길이의 평균 혈압 데이터로  수술 중 1 분 뒤 저혈압이 발생할 확률을 예측하였습니다.

## Deep Learning-based  Hypotension prediction
- 2 sec 간격으로 샘플링 된 20 sec 길이의 평균 혈압 데이터들 사이의 비선형적 관계를 파악하여 예측하기 위해, 딥러닝을 활용하였습니다.

- 시계열 데이터 예측 모델에 많이 쓰이는 LSTM 모델을 활용했으며, 모델의 정확도 및 결과값을 비교하기 위해 MLP, CNN 모델도 사용하였습니다. 


## Raw Data
[vital db](https://vitaldb.net/) 에서 제공하는 환자 데이터셋을 사용했습니다.


## Making Sequence Data

[vital db examples](https://github.com/vitaldb/examples/blob/master/hypotension_mbp.ipynb) 을 참고하여
평균 혈압데이터가 65 mmHg 아래인 상태를 hypotension 으로 간주하였고 , 
20 sec 간의 평균 혈압데이터를 input data , 1분 뒤의 평균 혈압데이터가 65 mmHg인지 유무를 라벨데이터로 데이터셋을 구축하였습니다.

![image](https://user-images.githubusercontent.com/79091824/193498449-f12b8e5e-471e-48dc-b511-45b557e13e68.png)
(사진 출처 : [vital db examples](https://github.com/vitaldb/examples/blob/master/hypotension_mbp.ipynb))

## Models and Training Details

- MLP
- LSTM(tensorflow & pytorch) 
- CNN
- 3 가지 모델 모두 데이터셋 불균형(hypotension : non hypotension =  1 : 20) 을 해결하기 위해, 1:20으로 training ratio 를 설정하였습니다.


## Results

|제목|auroc|auprc|acc|f1-score|
|------|---|---|---|---|
|mlp|0.79|0.58|0.91|0.57|
|lstm|0.97|0.77|0.94|0.71|
|cnn|0.87|0.63|0.94|0.57|

- 모델이 저혈압 발생(positive) 예측을 정확하게 하는 지 평가하는데 효과적인 metric인 auprc score를 주요 성능 비교 지표로 정하였습니다. 
- lstm 모델(tensorflow)의 auprc score가 0.77로 다른 모델들보다 높아, 저혈압 발생 예측을 가장 정확하게 하였다고 판단했습니다.


## Applications

Streamlit 라이브러리와 Tensorflow 라이브러리를 이용해 환자의 수술 중 혈압 데이터와 모델 예측 결과를 보여주는 시각화 프로토타입을 제작했습니다. <br>

<img width="600" alt="demo-streamlit" src="https://user-images.githubusercontent.com/79091824/193453626-f0949fe0-faae-4329-b975-7284336d9126.gif">

## Reference 
* [vital db examples](https://github.com/vitaldb/examples/blob/master/hypotension_mbp.ipynb)

