#개요 : 컨버넷(합성곱망)은 이미 학습했던거 가져다 다시 쓸 수 있다
#       완전연결망은 아님(일반 layer층)
#       컨버넷 가져와서 특성추출과 증식을 같이

# 1. 이전에 학습했던 합성곱 가져오기
import keras
from keras.applications import VGG16

conv_base = VGG16(weights="imagenet",
                    include_top=False, #합성곱만 가져오겠다
                    input_shape=(150,150, 3))
conv_base.summary()

# 2.데이터 준비
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = "./datasets/cats_and_dogs_small"

train_dir = base_dir + "/" + "train"
validation_dir = base_dir + "/" + "validation"
test_dir = base_dir + "/" + "test"

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20 #한번에 가져오는 데이터 수

# 3.네트워크 구성하기 - 불러온 합성곱(컨브넷) 포함하기
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt

model = models.Sequential()

model.add(conv_base) #합성곱층을 추가한다

#합성곱층에 완전연결층을 연결한다 
model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu",
                        input_dim= 4 * 4 * 512))
model.add(layers.Dense(1, activation="sigmoid"))

model.summary()

# **************************** 이미 학습된 convnet을 가져와서 그냥 쓰면
# 다시 역전파가 이워지면서 다시 학습이 됨 이걸 막아야 한다
# 이미 학습된 부분은 가증치 생긴이 이루어지지 않게 해야 한다
# 학습을 동결시킨다
#####################################################################
conv_base.trainable = False

# 우리데이터 준비 작업
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True,
    fill_mode = "nearest"
)

# 검증데이터는 증식이되어서는 안된다. 그래서 별도로 제너레이터를 만든다
test_datagen = ImageDataGenerator(rescale = 1./255,)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=batch_size,
    class_mode="binary"
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=batch_size,
    class_mode="binary"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=batch_size,
    class_mode="binary"
)

#컴파일
model.compile(loss="binary_crossentropy",
                optimizer=optimizers.RMSprop(lr=2e-5),
                metrics=["acc"])

#학습
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data = validation_generator,
    validation_steps=50
)

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


# 저장하기
model.save("result.h5")
import pickle

f = open("result.hist", "wb")
pickle.dump(history, file=f)
f.close()

drawChart(history)