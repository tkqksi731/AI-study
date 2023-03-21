from keras import models
from keras import layers
from keras.datasets import reuters

#1. 데이터 가져오기
(train_data, train_labels), (test_data, test_lables) = reuters.load_data(num_words=5000)

print(train_data.shape)
print(test_data.shape)
print(train_labels.shape)


from commonUtil import sequencesToWords
word_index = reuters.get_word_index()
result = sequencesToWords(word_index, train_data[0])
print("첫번째 문장 : ", result)
result = sequencesToWords(word_index, train_data[1])
print("두번째 문장 : ", result)

from commonUtil import vectorize_sequences
# sequence -> vector로
x_train = vectorize_sequences(train_data,5000)
x_test = vectorize_sequences(test_data,5000)

# 출력 데이터 one-hot 인코딩
from keras.utils import to_categorical
y_train = to_categorical(train_labels)
y_test = to_categorical(test_lables)

print("분류해야할 데이터 개수 ", y_train.shape)

#4.네트워크 모델 만들기
model = models.Sequential()
model.add( layers.Dense(64, activation="relu", input_shape=(5000,)))
model.add( layers.Dense(46, activation="softmax"))

#5.컴파일
model.compile( optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

#검증용 데이터와 훈련데이터 쪼개기
x_val = x_train[1000:]
pratial_x_train = x_train[:1000]

y_val = y_train[1000:]
pratial_y_train = y_train[:1000]

#학습하기
history = model.fit(pratial_x_train, pratial_y_train, epochs=10,
    batch_size=100, validation_data=(x_val, y_val))

results = model.evaluate(x_test, y_test)
print(results)

pred = model.predict(x_test)
pred_class = model.predict_classes(x_test)
print(pred_class[:10])
print(test_lables[:10])



from commonUtil import drawChartLoss, drawChartAccuary
#drawChartAccuary(history)
# drawChartLoss(history)

import numpy as np 
pred_class = model.predict_classes(x_test)
print(pred_class[:20])

test=list() 
for item in y_test:
    test.append(np.argmax(item))
print(test[:20])

