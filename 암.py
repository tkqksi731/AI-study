from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 
from keras.datasets import mnist 

cancer = load_breast_cancer()

train_data, test_data, train_labels, test_labels = train_test_split( 
    cancer["data"], cancer["target"],  random_state=42
)
#실제 데이터 출력하려고 보관 
test = test_labels
print( cancer["data"].shape)

#신경망을 만든다 
from keras import models 
from keras import layers 

#네트워크 - models.Sequential() 
network = models.Sequential()  #신경망 모델 만들기 
#입력레이어 추가 
#16 - 출력값의 개수 
network.add( layers.Dense(16, activation='relu',
                         input_shape=(30, )) ) 
network.add( layers.Dense(2, activation='softmax'))
   
#컴파일 - 손실함수, 옵티마이저(최적화), 
#결과값-어떤걸 모니터링할지
network.compile( optimizer="rmsprop",
                 loss="categorical_crossentropy",
                 metrics=['accuracy'])

#sklearn에서 사용하는 nomalize 지원함수
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit( train_data) #학습 : 0~1까지 수정하기 위한 수식
train_data = scalar.transform(train_data)
test_data = scalar.transform(test_data)

print( train_data[:100])
print( test_data[:100])


#결과를 onehot 인코딩을 해야 한다 
#숫자형 타입을 -> 범주형 타입으로 전환 
from keras.utils import to_categorical 

train_labels = to_categorical( train_labels )
print( train_labels.shape )
print( train_labels[:10])

test_labels = to_categorical( test_labels )
print( test_labels[:5])


network.fit (  train_data, 
               train_labels, 
               epochs=100, #학습횟수 
               batch_size=128 #데이터가 클때 학습한번하자고
                             #모든데이터한번에 메모리에
                             #로딩 못함, batch_size 에서
                             #준만큼 읽어와서 실행
            )

#머신러닝 score 함수 대신에 평가 
train_loss, train_acc = network.evaluate( train_data, 
                         train_labels)
print("훈련셋 손실 {} 정확도 {}".format(train_loss, train_acc))

test_loss, test_acc = network.evaluate(test_data, 
                        test_labels)
print("테스트셋 손실 {} 정확도 {}".format(test_loss, test_acc))

#값 예측해보기 
predict = network.predict( test_data )
#예측한 갓값을 분류화 하자 
result = network.predict_classes(test_data, verbose=0)
print("예측값 softmax 적용\n", predict[:100] )
print("\n클래스예측\n", result[:100])
print("\n실제값\n", test[:100])
