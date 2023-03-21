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
# 입력레이어 추가.3s3
network.add(layers.Dense(512  # 512 - ?? 적당히 설정
                        , activation='relu'  # activation(활성화 함수) 가중치를 적용한 출력이 1을 넘어갈 경우에
                                             # 0 ~ 1 사이에 머물러서 다음 계층으로 값을 전달하거나 하지 않을 것을
                                             # 정하는 함수. 요즘은 다 relu를 사용 
                        , input_shape=(28*28,))  # input_shape는 입력되는 데이터의 크기 지정
                                                 # 튜플형태로, comma를 꼭 붙여줘야 한다.
                        )
# 출력레이어 추가.  -  출력층 레이어는 첫번째 인자가 분류 계수
# 활성화 함수 'softmax'
# softmax -> 계산을 해서 결과를 뽑으면 임의 숫자 배열(현재 10)이 나온다.
#            이 임의의 숫자들을 확률로 치환하여 '임의숫자들 합 = 1'로 만들어주는 함수
# 출력층 : softmax, 그밖의 레이어는 relu 함수를 사용하면 된다. (선생님 말씀.)
network.add(layers.Dense(10, activation='softmax'))  

# 컴파일
# optimizer="사용할 함수 이름", loss="사용할 함수 이름"
network.compile(optimizer='rmsprop'  # 최적화
                , loss='categorical_crossentropy'  # 손실함수
                , metrics=['accuracy'])  # 결과값

# 입력데이터를 스케일링 해줘야 한다.
# 이미지색 0 ~ 255  ==>  0 ~ 1
# reshape는 튜플형태로 데이터를 넣어줘야 한다.
# 기존 데이터 형식은 3차원(60000*28*28 - train_img)이다.
# 이를 2차원 형태로 바꿔줘야하는데 머신러닝이나 딥러닝의
# 입력데이터를 2차원 텐서로 받기 때문이다.
train_img = train_img.reshape((60000, 28*28))  # 2차원 형태(60000*784)로 바꿔준다.
train_img = train_img.astype('float32') / 255

test_img = test_img.reshape((10000, 28*28))
test_img = test_img.astype('float32') / 255

# 기본적으로 데이터의 단위가 너무 크면 학습과정에서
# 왜곡되거나 학습이 잘 안되는 경우가 있어 단위를 0 ~ 1 사이로 바꾼다.

# 결과를 인코딩해야한다. (one-hot encoding)
# 숫자형 타입을 -> 범주형 타입으로 전환
from keras.utils import to_categorical
# 범주형(분류에 적절한..) 데이터의 경우 반드시 one-hot 인코딩으로 바꿔줘야한다.
# one-hot encoding이란 범주의 개수만큼 배열을 만들고
# 각 데이터에 해당하는 부분을 1로 하고 나머지를 0으로 만드는 기법이다.
# 현재 총 범주가 10개(0, 1, 2, ... , 8, 9)일 때
# one-hot 인코딩을 해주면 [1, 2, ... , 9, 10]으로 배열을 만들어주고
# 배열의 인덱스를 통해 범주가 0인 경우 1, 0, 0, ... , 0, 0
#                         1인 경우 0, 1, 0, ... , 0, 0
#                         2인 경우 0, 0, 1, ... , 0, 0
# 으로 만들어 준다.
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# 학습
network.fit(train_img
            , train_labels
            , epochs=5  # 학습횟수
            , batch_size=512)  # 데이터가 클 때, batch_size에서 준 만큼만 읽어와서 실행.
# 훈련 데이터 손실정도, 정확도
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
print("\n클래스 예측\n", result[:100])
print("\n실제값\n", test_labels[:100])