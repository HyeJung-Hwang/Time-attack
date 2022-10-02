import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Input, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
st.set_page_config(layout="wide")
st.title('Mean Blood Pressure Analysis of Postoperative Patients')

# data load
c = np.load("C:/Users/82103/Downloads/c.npy")
x = np.load("C:/Users/82103/Downloads/x.npy")
y = np.load("C:/Users/82103/Downloads/y.npy")
valid_mask = np.load("C:/Users/82103/Downloads/valid_mask.npy")
df = pd.read_csv("C:/Users/82103/Downloads/test.csv", index_col = 0)
MINUTES_AHEAD = 1

# model load
weight_path = "C:/Users/82103/Downloads/weights.hdf5"
model = Sequential()
model.add(LSTM(16, input_shape=x.shape[1:]))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.load_weights(weight_path)

# get caseid
occupation = st.selectbox("Select Patient ID",
                              [5267, 5344, 5368, 5439, 5499, 5512, 5537, 5601, 5647, 5805, 5829,
                               5842, 5848, 6010, 6071, 6127, 6235, 6257, 6335, 6351])
caseid = occupation

case_mask = (c == caseid)
case_len = np.sum(case_mask)
case_x = x[case_mask]
case_y = y[case_mask]
case_valid_mask = valid_mask[case_mask]

col1,col2= st.columns([1,3])

with col1:
    # get patient case
    st.write("##### Patient ", occupation, "Info")
    new_df = df[df["caseid"] == caseid]
    new_df = new_df.dropna(axis=1)
    new_df = new_df.astype('str')
    new_df = new_df.transpose()
    new_df.columns = ["data"]
    st.dataframe(new_df, 800, 400)
    st.write("###### Table Description")
    st.write("- Red : Patient's Mean Blood Pressure Value")
    st.write("- Blue : Model Prediction of hypotension")


# predict


with col2:
    st.write("#####      Patient ", caseid, "'s Intraoperative Mean blood Pressure Values and Model Prediction Results")
    case_predy = model.predict(case_x).flatten()  # model
    case_rmse = np.nanmean((case_y - case_predy) ** 2) ** 0.5
    
    fig = plt.figure(figsize=(20, 8))
    plt.xlim([0, case_len + MINUTES_AHEAD * 30])
    plt.xlabel('Time(minute)')
    t = np.arange(0, case_len)
    
    # red bars for the event
    ax1 = plt.gca()
    for i in range(len(case_y)):
        if case_y[i]:
            ax1.axvspan(i + MINUTES_AHEAD * 30, i + MINUTES_AHEAD * 30 + 1, color='r', alpha=0.1, lw=0)

    # 65 mmHg bar
    ax1.axhline(y=65, color='r', alpha=0.5)
    ax1.plot(t + 10, case_x[:, -1], color='r')
    ax1.set_ylabel(r"Mean Blood Pressure(mmHg)")
    ax1.set_ylim([0, 150])

    ax2 = ax1.twinx()

    # draw valid samples
    case_predy[~case_valid_mask] = np.nan
    ax2.plot(t, case_predy)
    ax2.set_ylabel(r"Hypotension Probability")
    ax2.set_ylim([0, 1])

    st.pyplot(fig)





