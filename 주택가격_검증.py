from keras.datasets import boston_housing
import numpy as np

(train_data, train_target), (test_data, test_target) = boston_housing.load_data()
print( train_data.shape, type(train_data))
print( train_target.shape, type(test_target))

# 딥러닝시 데이터는 꼭 정규화작업을 해야한다 (normalize)
#axis 축, axis=0 가로 axis=1 세로
"""
mean = train_data.mean(axis=0) #평균값 구하기


# for i in range(10):
#     print( mean[i], train_data[i])


train_data = test_data - mean
std = train_data.std(axis=0)
train_data / std
print(train_data[:10])
"""
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
scale.fit(train_data) #학습을 하면 - train_data로만
train_data = scale.transform(train_data)
test_data = scale.transform(test_data)

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation="relu", input_shape=(13,)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    #회귀의 경우 마지막에 데이터를 하나만 예측하면 되기 때문에
    #그리고 그 값은 scalar 형태라서 별도의 함수를 사용하지 않는다
    model.add(layers.Dense(1))
    model.compile(optimizer="rmsprop", loss="mse",#평균최소곱오차
                                        metrics=["mae"])
    return model

# K겹 검증
# 전체 샘플의 개수
# k = 4

# 전체데이터개수 : 100 k = 4 0..24, 25..49, 50..74, 75..99
#                 100//4 몫 25
#                 terain_data[0:25] 0~24
#                 terain_data[25:50] 25~49
#                 terain_data[50:75] 50~74
#                 terain_data[75:10] 75~99
#                 i=0    i*25=0      (i+1)*25 = 25
#                 i=1    i*25=25     (i+1)*25 = 50
#                 i=2    i*25=25     (i+1)*25 = 75
#                 i=3    i*25=75     (i+1)*25 = 100

def KfoldVarify(train_data, train_target, k=5, epochs=50, verbose=1):
    num_val_samples = len(train_data) // k
    num_epochs = epochs #에포크수
    all_scores = list() # K겹 검증한 평가값 넣을 리스트
    for i in range(k):
        print("처리중인 폴드 #", i)
        #검증 데이터 가져오기
        start = i * num_val_samples
        end = (i+1)*num_val_samples
        val_data = train_data[ start : end]
        val_targets = train_target[start:end]
        print(start, end)


        # i번쨰가 검증셋
        #훈련셋은 : 0~ i-1, i+1분터 끝까지
        partial_train_data = np.concatenate(
            [train_data[:start], train_data[end:]], axis=0
        )
        partial_train_target = np.concatenate(
            [train_target[:start], train_target[end:]], axis=0
        )

        #학습하기
        model = build_model() #새로운 모델 객체를 만든다.
        history = model.fit(partial_train_data, partial_train_target, epochs=num_epochs, batch_size=1,
        validation_data=(val_data, val_targets),
        verbose=verbose) #verbose=0을 주면 epoche과정이 보임

        val_mse, val_mae = model.evaluate( val_data, val_targets)
        print( history.history.keys() )
        all_scores.append(history.history["val_mean_absolute_error"])
    
    return all_scores


n_epochs=10
"""
scores = KfoldVarify(train_data, train_target, epochs=n_epochs, k=5)

avg_mae_history = [ np.mean([x[i] for x in scores]) for i in range(n_epochs)]


#차트그리기
import matplotlib.pyplot as plt 
x = range(1,  len(avg_mae_history)+1)
y = avg_mae_history 
plt.plot(x, y)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show() 
"""


#예측하기 끝내기
model = build_model()
model.fit(train_data, train_target, epochs=n_epochs, verbose=1, batch_size=100)
pred = model.predict(test_data)

#평가
test_mse_score, test_mae_score = model.evaluate(test_data, test_target)
print("실제값하고의 예측값하고의 오차평균 : ", test_mae_score)
for i in range(30):
    print(pred[i], test_target[i])
