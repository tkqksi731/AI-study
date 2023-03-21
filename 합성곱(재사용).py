import os, shutil 
from keras import layers
from keras import models
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import pickle
from keras.preprocessing import image
from keras.applications import VGG16

original_dataset_dir='./datasets/cats_and_dogs/train'
base_dir ='./datasets/cats_and_dogs_small'
test_dir = base_dir+ '/test' #훈련용
train_dir = base_dir+ '/train'#훈련용
valid_dir = base_dir+ '/validation'#훈련용

conv_base = VGG16(weights="imagenet",
                    include_top=False,
                    input_shape=(150,150,3))

conv_base.summary()

#미리 학습된 컨브넷으로 부터 특성값을 추출해야 한다
def extract_features( dicetory, sample_count):
    datagen = ImageDataGenerator(1/255)
    generator = datagen.flow_from_directory(
        dicetory,
        target_size=(150,150),
        batch_size=20,
        class_mode="binary"
    )

    # block5_pool (MaxPooling2D)   (None, 4, 4, 512)
    # 우리도 이거에 맞춰서 4 by 4 by 512개 만들어야 한다
    features = np.zeros( shape=(sample_count, 4, 4, 512))
    #미리 데이터 들어갈 메모리 확보, np.zeros 함수는 shape에
    #전달된 차원만큼 멤리 확보후 0으로 채워준다

    # 라벨 - 정답 들어갈 데이터 메모리 확보
    labels = np.zeros(shape=sample_count)

    batch_size = 20
    i = 0
    for input_batch, labels_batch in generator:
        #컨버넷에 학습된 내용들로부터 우리가 갖고 있는 데이터의
        #특성들을 추출하자
        features_batch = conv_base.predict(input_batch)
        #위에 만들어놓은 배열에 features_batch값을 저장한다
        features[i*batch_size:(i+1)*batch_size] = features_batch
        labels[i*batch_size:(i+1)*batch_size] = labels_batch
        #제너레이터가 무한정 데이터를 생성해내므로 적당한 선에서 멈춰야 한다
        i = i + 1
        if i * batch_size >= sample_count:
            break
    return features, labels

def Predict():
    model = load_model("cats_and_dogs_small_2.h5")
    file = open("cats_and_dogs.hist","rb")
    history = pickle.load(file)
    file.close()
    drawChart(history)

def drawChart(history):
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss= history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1, len(acc)+1)

    plt.plot(epochs, acc, "bo", label = "Training acc")
    plt.plot(epochs, val_acc, "b", label = "Validation acc")
    plt.title("Training and Validation Accuarcy")
    plt.legend()
    plt.show()

    plt.figure() # 차트 refresh 다시그림
    plt.plot(epochs, loss, "bo", label = "Training loss")
    plt.plot(epochs, val_loss, "b", label = "Training loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()



# 특성 추출하기
train_fetures, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(valid_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

#Maxpool층으로부터 완전연결층(딥러닝 기본 layer)에 데이터 전달을 해야 한다.
#Maxpool층은 4d 텐서, 완전연결층 2d 텔서 사용

train_fetures = np.reshape(train_fetures, (2000, 4*4*512))
validation_features = np.reshape(validation_features, (1000, 4*4*512))
test_features = np.reshape(test_features, (1000, 4*4*512))

#완전연결층과 연결하기
model = models.Sequential()
model.add(layers.Dense(256, activation="relu", input_dim=4*4*512))
model.add(layers.Dropout(0.5))
#가끔 정기적인 패턴을 깨기 위해서 일정 비율의 값들을 0으로 만든다
#같은 패턴에 대한 학습을 막기 위해서 일부러 노이즈를 일으킨다
#과대적합 막기

model.add(layers.Dense(1, activation="sigmoid"))
#출력레이어 이진부류라서 activation은 sigmoid 사용
# 컴파일 하기
model.compile( optimizer=optimizers.RMSprop(lr=2e-5),
                loss="binary_crossentropy",
                metrics=["acc"])

history = model.fit(train_fetures, train_labels,
            epochs=30, batch_size=20,
            validation_data=(validation_features, validation_labels))


drawChart(history)
