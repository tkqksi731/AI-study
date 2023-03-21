# 1.데이터 불러오기
from keras.datasets import imdb
from keras import models
from keras import layers

(train_data, train_labels), (test_data, test_labels)= imdb.load_data(num_words=10000)
#자주 쓰는 데이터만
# print( train_data[:5])
# print( train_labels[:5])

# 데이터가 시퀀스로 온다. 문자열(단어)의 경우 그 자체로는 딥러닝을 못 한다
# 사전을 정해두고 단어들을 사전의 index로 바꾸어서 해야 한다
# 이를 시퀀스라고 한다

#원래의 단어로 바꾸어서 보자
word_index = imdb.get_word_index()
print(type(word_index.items()))  # dict하고 달라서 dict타입으로 전환
# 내부 구조 확인해보기
def showDictionary(cnt):
    i = 0
    for key in dict(word_index.items()):
        if i >= cnt:
            break
        i = i + 1
        print(key, dict(word_index.items())[key])

showDictionary(10)

temp = [(value, key) for (key, value) in word_index.items()]
reverse_word_index = dict(temp)
# print( reverse_word_index[:10])
# index -> 단어로 바꾸어서 - 다 합쳐서 보자

decoded_review = " ".join( [reverse_word_index.get(i-3, "?")
    for i in train_data[0]])
print( decoded_review )

# join 함수
# test = " ".join(["박동석", '이용희', "이상민", "정자은"])
# print(test)

#현재 imdb데이터 셋이 2d텐서가 아니라서 2d텐서로 바꿔야한다.
import numpy as np
def vectorize_sequences( sequences, dimension=10000):
    result = np.zeros( (len(sequences), dimension))
    #np.zeros(행,열) - 행 by 열 만큼 0으로 이루어진 행렬을
    #                 만들어주는 함수이다
    # 미리 메모리 확보하고
    for i, sequences in enumerate(sequences):
        #enumerate 함수, 데이터를 인덱스를 데이터로 변환
        result[i, sequences] = 1. #result[i]에서 특정인덱스를 1로
    return result

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print( x_train[:5])
print( x_test[:5])

# 출력데이터를 one-hot 인코딩해서 작업
from keras.utils import to_categorical

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

print(train_data.shape)
print(test_data.shape)

#훈려넷만들기 -검증용 (전체25000개임)
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#네트워크 모델 만들기
model = models.Sequential()

# 입력 layer 만들기
model.add( layers.Dense(16, activation="relu", input_shape=(10000,)))
#중간에 새로운 층 삽입
model.add(layers.Dropout(0.5)) #랜덤하게 가중치중 일부값을 0으로 초기화
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dropout(0.5))
#출력 layer 만들기, 2진분류의 경우 다르게 작성
model.add(layers.Dense(2, activation='softmax'))

#모델컴파일
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

#학습시작하기
history = model.fit( partial_x_train, partial_y_train, epochs=20, batch_size=100, validation_data=(x_val, y_val))
# fit함수가 학습에 필요했던 모든 역사(history)를 저장했다가 변환한다
results = model.evaluate(x_test, y_test)
print(results)

pred = model.predict(x_test)
pred_class = model.predict_classes(x_test)
print(pred_class[:40])
print(test_labels[:40])

from commonUtil import drawChartLoss
from commonUtil import drawChartAccuary


drawChartLoss(history)
drawChartAccuary(history)

acc = history.history["acc"]
loss = history.history["loss"]
val_acc = history.history["val_acc"]
val_loss = history.history["val_loss"]

for i,j,k,l in zip(acc, loss, val_acc, val_loss):
    print("훈련셋 : {} 손실:{}, 검정셋 : {} 손실:{}".format(i, j, k, l))