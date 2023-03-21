
#폴더를 만들거나 삭제 또는 이동, 파일 수정, 삭제등은 os영역이다. 

#  datasets
#      ㄴ  cats_and_dogs_small 
#              ㄴ  train 
#                   ㄴ cats 폴더 순서대로 자동으로 라벨로 인신한다 
#                       ㄴ dogs 
#                   ㄴ test #cats와 dogs 만들어야 함 
#                   ㄴ   validation cats와 dogs 만들어야 함 


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

original_dataset_dir='./datasets/cats_and_dogs/train'
base_dir ='./datasets/cats_and_dogs_small'
test_dir = base_dir+ '/test' #훈련용
train_dir = base_dir+ '/train'#훈련용
valid_dir = base_dir+ '/validation'#훈련용


train_cats_dir = train_dir + "/cats"
train_dogs_dir = train_dir + "/dogs"

test_cats_dir = test_dir+"/cats"
test_dogs_dir = test_dir+ "/dogs"

valid_cats_dir = valid_dir +"/cats"
valid_dogs_dir = valid_dir +"/dogs"
#다운로드받은 이미지 있는곳 

def ImageDistribution():
    #소규모 데이터셋을 저장하는 디렉토리
    if( os.path.exists(base_dir)): #이미 경로가  존재하면 
        shutil.rmtree(base_dir )
        #이미지 포함 base_dir 아래의 모든 경로 삭제 
    os.mkdir(base_dir) #디렉토리 만들기 
    os.mkdir(train_dir)
    os.mkdir(valid_dir)
    os.mkdir(test_dir)
    os.mkdir(train_cats_dir)
    os.mkdir(train_dogs_dir)
    os.mkdir(test_cats_dir)
    os.mkdir(test_dogs_dir)
    os.mkdir(valid_cats_dir)
    os.mkdir(valid_dogs_dir)
   

    #파일 복사하기 
    #디렉토리내의 파일 개수 조사하기 
    totalCount = len( os.listdir(original_dataset_dir))
    #os.listdir(경로명 ) 해당 경로에 있는 파일 목록을 가져온다 
    print("전체 개수 :", totalCount)
    ImageCopy(train_cats_dir,   0,  1000,  "cat")
    ImageCopy(train_dogs_dir,   0,  1000,  "dog")
    ImageCopy(test_cats_dir, 1000,  1500,  "cat")
    ImageCopy(test_cats_dir, 1000,  1500,  "dog")
    ImageCopy(valid_cats_dir,1500,  2000,  "cat")
    ImageCopy(valid_dogs_dir,1500,  2000,  "dog")
    

def ImageCopy(destdir, start, end, imagename):
    # 파일명들이 cat.0.jpg, cat.1.jpg... 파일이름을 만들어내야 한다
    fnames = ["{}.{}.jpg".format(imagename,i) for i in range(start,end)]
    print(fnames[:5])
    for fname in fnames:
        src = original_dataset_dir+"/"+fname # 원본파일명
        dest = destdir+"/"+fname # 복사할 파일명
        shutil.copyfile(src,dest)


# 모델 구성하기
def makeModel():
    model = models.Sequential()
    # 컨버넷 추가 - 컨보넷은 3차원 이미지의 특성을 추출한다.
    # 보통 3,3 또는 5,5 필터를 사용한다
    # 3d tensor를 입력으로 사용한다
    # 32 -> 출력할때 unit의 개수

    model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(150,150,3)))

    model.add(layers.MaxPooling2D(2,2))
    model.add(layers.Conv2D(64, (5,5), activation="relu"))
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Conv2D(128, (3,3), activation="relu"))
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Conv2D(128, (3,3), activation="relu"))
    model.add(layers.MaxPooling2D(2,2))

    model.add(layers.Flatten())

    # 출력층
    model.add(layers.Dense(1, activation="sigmoid"))
    
    model.compile(loss="binary_crossentropy",
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=["acc"])

    # 모델 구조 출력하기
    model.summary()
    return model

# 데이터 전처리
def dataPreprocessing():
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir, # target 디렉토리
        target_size=(150,150),
        batch_size=20,
        class_mode="binary"
    )

    validation_generator = test_datagen.flow_from_directory(
        valid_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode="binary"
    )

    # 예측 - test_generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150,150),
        batch_size=20,
        class_mode="binary"
    )
    
    # 학습 - ImageGenerator 쓸때는 fit_generator 함수 사용
    model = makeModel()

    history = model.fit_generator(train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data = validation_generator,
        validation_steps = 50)

    # 학습모델 저장하기
    model.save("dats_and_dogs_small_2.h5")

    # history 는 pickle을 이용해서 저장해보자
    
    file = open("cats_and_dogs.hist", "wb")
    pickle.dump(history, file=file)
    file.close()

def Predict():
    model = load_model("cats_and_dogs_small_2.h5")
    file = open("cats_and_dogs.hist","rb")
    history = pickle.load(file)
    file.close()
    drawChart(history)

"""
    print("개 : {} 고양이 : {}".format(dogcount, catcount))

    score = model.evaluate_generator(test_generator, steps=50)
    print("Test loss : ", score[0])
    print("Test accuracy : ", score[1])

"""

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

def ImageIncrease():
    #os.listdir(경로명)해당 폴더의 파일 목록을 가져온다
    filenamelist = os.listdir(train_cats_dir)
    print(filenamelist[:20])

    #경로 포함한 파이명
    filename = [train_cats_dir + "/" + fname for fname in filenamelist]
    print(filename[:20])

    increaseImage = filename[3] #0번 이미지를 증식해보자
    img = image.load_img(increaseImage)
    plt.imshow(img)
    plt.title="priginal image"
    plt.show()
    
    #1.이미지를 numpy 배열로 바꿔야 한다
    data = image.img_to_array(img) # 150, 150, 3
    print(data)
    #2.원하는 차원으로 재가공
    data = data.reshape( (1,) + data.shape) # 1. 150,150,3
    print(data)
    #3.이미지 가공 객체 만들기
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    #4. ImageDataGenerator 객체로부터 새오룬 이미지 만들기
    #냅두면 계속 생성한다. 그래서 i를 카운트로 두고 4개 만들면
    #그만 만들게 하였음
    i = 0
    for batch in datagen.flow( data, batch_size=1):
        plt.figure(i) # i번쨰 위치에
        # datagen객체가 생성하는 이미지는 배열임
        #배열을 다시 이미지로 환원해야 한다 .array_to_image
        plt.imshow(image.array_to_img(batch[0]))
        i = i + 1
        if i%10 == 0:
            break
    plt.show()

def DataIncreaseFit():

    model = makeModel()

    #trainge
    datagen = ImageDataGenerator(
        rescale=.1/255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    test_datagen = ImageDataGenerator(.1/255)
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size=20, # batch_size와 steps_per_epoch 두개를 곱한개수
                       # 1000개 넘으면 나머지 이미지는 자기가 증식
        class_mode="binary"
    )

    validation_generator = test_datagen.flow_from_directory(
        valid_dir,
        target_size=(150,150),
        batch_size=10,
        class_mode="binary"
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150,150),
        batch_size=10,
        class_mode="binary"
    )

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs = 100,
        validation_data = validation_generator,
        validation_steps=50
    )

    # 학습모델 저장하기
    # model.save("dats_and_dogs_small_3.h5")

    # history 는 pickle을 이용해서 저장해보자
    import pickle
    # file = open("cats_and_dogs2.hist", "wb")
    pickle.dump(history, file=file)
    file.close()
    
    # 차트 그리기
    drawChart(history)

if __name__ == '__main__':
    # ImageDistribution()
    dataPreprocessing()
    # Predict()
    # ImageIncrease()
    # DataIncreaseFit()