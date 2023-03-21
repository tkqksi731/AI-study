import os,shutil

from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import load_model

base_dir='./datasets/handwriting_shape'
train_dir=base_dir+'/train'
test_dir=base_dir+'/test'
# valid_dir=base_dir+'/validation'

def makeModel():
    model=models.Sequential()
     #커버넷 추가
     #보통 3,3 또는 5,5 필터를 사용한다
     #32 -> 출력할 unit의 개수

    model.add(layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)))  
    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(32,(3,3),activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Flatten())

    #출력층
    model.add(layers.Dense(512,activation='relu'))
    model.add(layers.Dense(3,activation='softmax'))

    #컴파일
    model.compile(loss='categorical_crossentropy',
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=['acc'])

    return model

def dataPreprocessing():   
    train_datagen = ImageDataGenerator(rescale=1./255)
    # test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150,150),batch_size=20,class_mode='categorical')
    # valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=(150,150),batch_size=20,class_mode='binary')
    # test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150,150),batch_size=20,class_mode='categorical')

    #학습시작   - Imagegenerate 사용 시에는 fit_generator 사용
    model=makeModel()
    history=model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=10)
    print (history)

    model.save("handwriting.h5")
    import pickle
    #pickle 확장자는 내 마음대로, wb binary형태로 저장하기
    file = open("handwriting_hist.bin", "wb")
    pickle.dump(history, file=file)
    file.close()

def Predict():
    #저장했던 모델 불러오기
    #에측하기
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(test_dir,
     target_size=(150,150),
     batch_size=3,
     class_mode='categorical')

    model = load_model("handwriting.h5")
    pred = model.predict_generator(test_generator, steps=6)
    # strps * batchs_size
    #분류 인덱스 출력하기
    print(test_generator.class_indices)

    class_name=["circle", "rectangle", "triangle"]
    cnt = [0,0,0]
    for item in pred:
        pos = np.argmax(item)
        cnt[pos] = cnt[pos]+1
        print(class_name[pos])

    print("circle {} rectangle {} triangle {}".format(cnt[0], cnt[1], cnt[2]))

if __name__ == "__main__":
    # dataPreprocessing()
    Predict()