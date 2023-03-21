from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
house = fetch_california_housing()
print( house["DESCR"]) #데이터에 대한 설명
import numpy as np

# data = house["data"]
# target = house["target"]

# print(data[:10])
# print(target[:10])

train_data, test_data, train_target, test_target = train_test_split(
    house["data"], house["target"], random_state=0)

#신경망을 만든다 
from keras import models 
from keras import layers 

train_data = np.array(train_data)

print(train_data.shape)


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation="relu", input_shape=(8,)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    #회귀의 경우 마지막에 데이터를 하나만 예측하면 되기 때문에
    #그리고 그 값은 scalar 형태라서 별도의 함수를 사용하지 않는다
    model.add(layers.Dense(1))
    model.compile(optimizer="rmsprop", loss="mse",#평균최소곱오차
                                        metrics=["mae"])
    return model

def KfoldVarify(train_data, train_target, k=5, epochs=10, verbose=1):
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

n_epochs=2

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
model = build_model()
model.fit(train_data, train_target, epochs=120, batch_size=100)

val_mse, val_mae = model.evaluate(test_data, test_target)

pred = model.predict(test_data)
for i in range(10):
    print(pred[i], test_target[i])
"""