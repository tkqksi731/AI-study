#기온예측하기 
#기온 데이터 입력받기 
import os 
data_dir = "./datasets/jena_climate"
fname = data_dir + "/jena_climate_2009_2016.csv"

f = open(fname, 'r', encoding='utf8')
data = f.read() #데이터 모두 읽기
f.close()
#print(data)

#데이터 한줄씩 분리 
lines = data.split('\n')
header = lines[0].split(",") #첫번째 라인에 제목이 있음 
#제목은 csv라서 , 로 분리됨 
print(header)
#나머지 라인은 데이터로 
lines = lines[1:] #제목줄 빼고 나머지 라인만 복사 
#print(lines[:5])#앞에서 5줄만 데이터 확인 
print("데이터 개수 : " , len(lines))

#넘파이 배열로 바꾸자 
import numpy as np 
float_data = np.zeros( (len(lines), len(header)-1) )
for i, line in enumerate( lines ):
    #라인 하나를 , 로 구분해서 데이터들을 float 형으로 전환후 
    #첫번째 컬럼 버리고 나머지 컬럼들을 저장
    values = [float(x) for x in line.split(',')[1:]]
    if i<5: #처음 다섯줄만 인쇄 
        print(values)
    #float_data[i, :] - 행만 지정 :-열전체 
    float_data[i, :] = values

import matplotlib.pyplot as plt 

y = float_data[:, 1] #온도만 , :-전체행, 1-컬럼 
x = range(len(y))
#시계열자료 차트 그리기 - 시계열 : 시간과 계절에 영향을 받는다 
#
# plt.plot(x, y)
#plt.show() 

#처음 10일간만 가지고 차크 그리기
# plt.plot( range(1440), y[:1440])
#plt.show() 


#데이터를 정규화 하자(Normalize) :신경망은 데이터의 단위가 
#0~1 사이에 머무르도록 해야 학습이 잘된다. 
# (데이터값 - 평균값)/표준편차 

#20000개의 데이터만 
mean = float_data[:20000].mean() #
float_data = float_data  - mean 
std = float_data[:20000].std()
float_data = float_data/ std 
#5개만 학인
print( float_data[:5])

#데이터 생성기 
def generator(data, lookback, delay, 
            min_index, max_index,
            shuffle=False, batch_size=128, step=6):
    #min_index부터 max_index까지가 작업할 내용 
    if max_index is None: #최대치가 전달이 안되면 
        max_index = len(data)-delay-1 

    #0~20000개 
    i = min_index + lookback  #lookback=1440 개 , 데이터가 0번째 행보다 
    #더 내려가면 안되니까 
    while 1: #무한루프 계속 돈다 
        if shuffle: #데이터를 섞으라고 하면 
            rows = np.random.randint( min_index+lookback,
                     max_index, size=batch_size) 
            # 1440 ~ 10000, 128 
        else: #섞으라고 안하면 
            if i+batch_size >= max_index:
                i = min_index + lookback 
            rows =np.arange(i, min(i+batch_size, max_index))
            i = i + len(rows)
        #  /  결과가 실수 
        #  // 결과가 정수, 몫 구하는 연산자  
        # data.shape[-1] : 열의 개수 
        samples = np.zeros( (len(rows), lookback//step,
                                     data.shape[-1])  ) 
        targets = np.zeros((len(rows), )) 
    
        for j, row in enumerate( rows ):
            indicies = range(rows[j]-lookback, rows[j], step)
            samples[j] = data[indicies]
            targets[j] = data[rows[j]  + delay][1]
        
        yield samples, targets 

lookback = 1440 #열흘치 자료로 내일 온도 예측하기
step=6  #10분마다 온도를 재었음, 한시간에 한번씩만 가져오기 
delay = 144 #하루치
batch_size=128 #한번에 128개씩 던짐 

train_gen = generator( data=float_data, lookback=lookback, 
                 delay = delay,  
                 min_index=0,
                 max_index=200000,
                 shuffle=True,
                 step = step,
                 batch_size=batch_size)
#제너레이터(yield 구문 있는 함수 ), 데이터를 보내고도 함수 자체가 
#                                종료안함 
#이터레이터 객체를 이용해 호출
e = iter(train_gen)
print( next(e) )

val_gen = generator( data=float_data, lookback=lookback, 
                 delay = delay,  
                 min_index=200001,
                 max_index=300000,
                 shuffle=True,
                 step = step,
                 batch_size=batch_size)

test_gen = generator( data=float_data, lookback=lookback, 
                 delay = delay,  
                 min_index=300001,
                 max_index=None,
                 shuffle=True,
                 step = step,
                 batch_size=batch_size)

#전체 검증 세트를 순회하기 위해 val_gen에서 추출할 횟수
val_steps = (300000 - 200001 - lookback)//batch_size 

# bactch_size * step_per_epochs = 건수 
test_steps = (len(float_data) - 300001 - lookback) //batch_size 

#평균제곱오차를 이용해 기본 예측을 해보자 
def evaluate_native_method():
    batch_maes=[]
    for step in range(val_steps):
        samples, targets = next(val_gen) #val_gen을 스탭만큼 실행 
        preds = samples[:, -1, 1]
        mae = np.mean( np.abs(preds-targets))
        batch_maes.append( mae )

    print( np.mean(batch_maes))

evaluate_native_method()

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

# model = Sequential()
# model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
# model.add(layers.Dense(32, activation="relu"))
# model.add(layers.Dense(1))

# #기온예측 - 회귀, 최소제곱 오차법
# model.compile(optimizer=RMSprop(), loss="mae")
# history = model.fit_generator(train_gen,
#         steps_per_epoch=500,
#         epochs=20,
#         validation_data = val_gen,
#         validation_steps=val_steps
#         )

# model.save("기본.h5")
# import pickle
# f = open('history1.hist', "wb") # 저장전에 파일을 만들어야
# pickle.dump(history, file=f) #파일저장
# f.close()

# import commonUtil
# commonUtil.drawChartLoss(history)

# 순환싱경망
model = Sequential()
# 순환신경망 : SimpleRNN, LTMS, GRU
model.add( layers.GRU(32,
                    input_shape=(None, float_data.shape[-1]),
                    dropout = 0.2,
                    recurrent_dropout=0.2))
model.add( layers.Dense(1))

# #기온예측 - 회귀, 최소제곱 오차법
# model.compile(optimizer=RMSprop(), loss="mae")
# history = model.fit_generator(train_gen,
#         steps_per_epoch=500,
#         epochs=40,
#         validation_data = val_gen,
#         validation_steps=val_steps
#         )

# model.save("기본3.h5")
# import pickle
# f = open('history3.hist', "wb") # 저장전에 파일을 만들어야
# pickle.dump(history, file=f) #파일저장
# f.close()

# import commonUtil
# commonUtil.drawChartLoss(history)

