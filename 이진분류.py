from keras import models
from keras import layers
import numpy as np

# 데이터 준비
# 분류하기 * 포유류, 조류, 기타 (0,1,2): 3가지 분류

#입력자료 [털, 날개] 털만 있고 날개 없으면 포유류
#                   털과 날개있으면 조류
#                   날개있고 털이 없으면, 둘다 없어도 기타

x_data = [
    [0,0], [0,1], [1,1], [1,0], [0,0],
    [1,0], [1,0], [1,1], [1,0], [0,0],
    [1,0], [1,1], [1,1], [1,0], [0,0]
]
# 위의 타입은 파이썬의 list 타입임, 머신러닝은 벡터만 가능

x_data = np.array(x_data)
# 원핫인코딩 - 분류할 때 딥러닝의 경우에 0 1 2 형태로 하지 않고
# 이를 벡터화 해야한다
# 0 1 2
# 1 0 0 - 기타 - 해당항목만 1일고 나머지는 모두 0으로
# 0 1 0 - 포류류
# 0 1 1 - 조류
# 분류의 label 개수만큼 벡터로 만들어서 모든 요소는 0으로 하고
# 해당 label만 1로 하는 방식 (one-hot encoding)

# 결과 데이터 [0 - 기타, 1- 포유류, 2-조류]
# 딥러닝할때 분류를 원핫인코딩 방식으로 해야한다 0,1,2
# 범주형 데이터 [1,0,0] - 기타 0,0 0,1
#              [0,1,0] - 포유류 1,0
#              [0,0,1] - 조류 1,1
y_data = [[1,0,0], [1,0,0], [0,0,1], [0,1,0], [1,0,0],
            [0,1,0], [0,1,0], [0,0,1], [0,1,0], [1,0,0],
             [0,1,0], [0,0,1], [0,0,1], [0,1,0], [0,0,1]]

y_data = np.array(y_data)
# 네트워크 모델 만들기
model = models.Sequential()
#layer추가하기
model.add(layers.Dense(16, activation="relu", input_shape=(2, )))
#중간에 새로운 layer 추가해보기
model.add(layers.Dense(16, activation="relu"))
#output layer
model.add(layers.Dense(3, activation="softmax"))
#모델 컴파일
model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                metrics=["accuracy"])

model.fit( x_data, y_data, epochs=30, batch_size=100)

#평가
results = model.evaluate( x_data, y_data)
print(results)