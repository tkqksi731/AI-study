import numpy as np

target = np.array([12,13,14,15,16]) # 원래값
pred = np.array([11,10,13,12,11]) # 예측값

a = np.mean(np.abs(pred - target))
print("평균 절대값 오차 : ", a)