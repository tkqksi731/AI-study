from keras.datasets import boston_housing

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

model = build_model()
model.fit(train_data, train_target, epochs=120, batch_size=100)

val_mse, val_mae = model.evaluate(test_data, test_target)

pred = model.predict(test_data)
for i in range(10):
    print(pred[i], test_target[i])