from keras import models 
from keras import layers 
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt 
import os, shutil
base_dir = './datasets/handwriting_shape' # 원본 데이터셋을 압축 해제한 디렉터리 경로 
train_dir = base_dir + '/train' 
test_dir = base_dir + '/test'

# 모델 구성하기 
def makeModel():
    """
    컨브넷 추가 - 컨브넷은 3차원 이미지의 특성을 추출한다.
    보통 (3, 3), (5, 5) 필터를 사용한다 
    3d tensor를 입력으로 사용한다 
    32 -> 출력할 unit의 개수 
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(), # Full connected layers에서 사용하기 위해 3d tensor 펼치기 
        # 출력층
        layers.Dense(512, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])

    # 컴파일
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc']) # optimizer -> 조정해야해서 이렇게 사용

    # 모델 구조 출력하기
    model.summary()

    return model

# 데이터 전처리
def dataPreprocessing():
    train_datagen = ImageDataGenerator(rescale=1./255) # 모든 이미지를 1/255로 스케일 조정
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, # target directory
        target_size=(150, 150),
        batch_size=20,
        class_mode='sparse'
    )

    # 예측 - test_generator 
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='sparse'
    )

    model = makeModel()

    # 학습시작 - ImageGenerator 쓸때는 fit_generator 함수 사용 
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
    )

    # 학습모델 저장하기 
    model.save('handwriting_shape_small_2.h5')

    # history는 pickle을 이용해 저장하기 
    import pickle 
    file = open('handwriting_shape.hist', 'wb')
    pickle.dump(history, file=file)
    file.close()

if __name__ == '__main__':
    # ImageDistribution() # 준비작업
    dataPreprocessing()
