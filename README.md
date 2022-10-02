# Hypotension prediction using mean blood pressure values using Deep Learning

## Overview
설명

## Our Goal
설명

## Deep Learning-based  Hypotension prediction
설명



## Raw Data
[vital db](https://vitaldb.net/) 에서 제공하는 환자 데이터셋을 사용했다.


## Making Sequence Data

[vital db examples](https://github.com/vitaldb/examples/blob/master/hypotension_mbp.ipynb) 을 참고하여
평균 혈압데이터가 65 mmHg 아래인 상태를 hypotension 으로 간주하였고 , 
20 sec 간의 평균 혈압데이터를 input data , 1분 뒤의 평균 혈압데이터가 65 mmHg인지 유무를 라벨데이터로 데이터셋을 구축하였다.


## Models and training details

- MLP
- LSTM
- CNN

## Results

|제목|auroc|auprc|recall|precision|f1-score|
|------|---|---|---|---|---|
|mlp|0.79|0.58|0.62|0.53|0.57|
|lstm|0.89|0.64|0.87|0.40|0.55|
|cnn|0.87|0.63|0.81|0.45|0.57|

실제로 양성일 환자를 양성으로 모델이 예측한 확률을 나타내는 recall 지표와 precision과 recall을 모두 보는 지표인 auprc 2가지로 모델의 성능을 비교해보았을 때,
lstm 모델의 성능이 가장 좋은 것으로 간주하였다.

## Applications

Streamlit 라이브러리를 통해 환자의 수술 중 혈압 데이터와 모델 예측 결과를 보여주는 시각화 프로토타입을 제작하였습니다. <br>

<img width="600" alt="demo-streamlit" src="https://user-images.githubusercontent.com/79091824/193453626-f0949fe0-faae-4329-b975-7284336d9126.gif">

## Reference 
* [vital db examples](https://github.com/vitaldb/examples/blob/master/hypotension_mbp.ipynb)

