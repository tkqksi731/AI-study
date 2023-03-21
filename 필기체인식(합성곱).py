from keras.datasets import mnist
import numpy as np

(train_img, train_labels), (test_img, test_labels) = mnist.load_data()

# 이미 데이터가 만들어져 있다. 60000개의 훈련데이터셋 (* 28 * 28)
# 10000개의 테스트셋.
# 신경망의 이미지는 3차원을 2차원으로 바꾸어서 인식한다.
print(type(train_img))
print(train_img.shape)  # ==> 60000 * 28 * 28
print(train_img[0])

print(type(train_labels))
print(train_labels.shape)
print(train_labels[:10])

# 이미지 그려보기
import matplotlib.pyplot as plt

def imgShow(id_):
    img = train_img[id_]
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show()

# imgShow(1)

# 신경망을 만든다.
from keras import models
from keras import layers

# 네트워크 - models.Sequential()
network = models.Sequential()  # 신경망 모델 만들기
# 컨브넷 추가하기
network.add(layers.Conv2D(32, (3,3), activation="relu",
                            input_shape=(28, 28, 1)))
network.add(layers.MaxPool2D((2,2)))                                
network.add(layers.Conv2D(64, (3,3), activation="relu"))
network.add(layers.MaxPool2D((2,2)))            
network.add(layers.Conv2D(64, (3,3), activation="relu"))
network.add(layers.Flatten())
# 입력레이어 추가.
network.add(layers.Dense(512, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))  

# 컴파일
# optimizer="사용할 함수 이름", loss="사용할 함수 이름"
network.compile(optimizer='rmsprop'  # 최적화
                , loss='categorical_crossentropy'  # 손실함수
                , metrics=['accuracy'])  # 결과값

# train_img = train_img.reshape((60000, 28*28))
#컨브넷에서는 아래처럼
train_img = train_img.reshape((60000, 28, 28, 1))
train_img = train_img.astype('float32') / 255

# test_img = test_img.reshape((10000, 28*28))
test_img = test_img.reshape((10000, 28, 28, 1))
test_img = test_img.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 학습
network.fit(train_img
            , train_labels
            , epochs=5  # 학습횟수
            , batch_size=512)
train_loss, train_acc = network.evaluate(train_img, train_labels)
# 테스트 데이터 손실정도, 정확도
test_loss, test_acc = network.evaluate(test_img, test_labels)
print('훈련셋 손실   : {}  정확도 : {}'.format(train_loss, train_acc))
print('테스트셋 손실 : {}  정확도 : {}'.format(test_loss, test_acc))

# 값 예측해보기
predict = network.predict(test_img)
# 예측한 값을 분류화
result = network.predict_classes(test_img, verbose=0)
print("예측값 softmax 적용\n", predict[:100])
# print("\n클래스 예측\n", result[:100])
# print("\n실제값\n", test_labels[:100])