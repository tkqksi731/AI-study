import tensorflow as tf
import numpy as np

# 분류하기 * 포유류, 조류, 기타 (0,1,2): 3가지 분류

#입력자료 [털, 날개] 털만 있고 날개 없으면 포유류
#                   털과 날개있으면 조류
#                   날개있고 털이 없으면, 둘다 없어도 기타

x_data = [
    [0,0], [0,1], [1,1], [1,0], [0,0], [1,0], [1,0]
    ,[1,1], [1,0], [0,0], [1,0], [1,1], [1,1], [1,0]
]
# 위의 타입은 파이썬의 list 타입임, 머신러닝은 벡터만 가능
x_data = np.array(x_data)
# 결과 데이터 [0 - 기타, 1- 포유류, 2-조류]
# 딥러닝할때 분류를 원핫인코딩 방식으로 해야한다 0,1,2
# 범주형 데이터 [1,0,0] - 기타 0,0 0,1
#              [0,1,0] - 포유류 1,0
#              [0,0,1] - 조류 1,1
y_data = [[1,0,0], [1,0,0], [0,0,1], [0,1,0], [1,0,0], [0,1,0], [0,1,0],
          [0,0,1], [0,1,0], [1,0,0], [0,1,0], [0,0,1], [0,0,1], [0,1,0]]

# 신경망 모델 구성
X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# y = f(w1x1 + w2x2 + w3x3 + .... + b)
# 신경망은 2차원으로 [입력층, 출력층] -> [2,3]으로 정한다
# 가중치 벡터 - 초기에 랜덤하게
W = tf.Variable(tf.random_uniform([2,3], -1.0, 1.0))

# 편향
b = tf.Variable(tf.zeros([3]))
# 신경망에 가중치 w와 b를 적용한다
# 텐서의 matmul이 numpy의 dot이다. 행렬의 곱
L = tf.add(tf.matmul(X, W), b)

#출력값 L을 0~1 사이에 머무르게 해야 한다
#활성화 함수를 사용한다. (계단함수, 시그모이드, relu중에 선택)
# 요즈은 거의 relu를 사용한다
L = tf.nn.relu(L)

# 출력값을 조정을 해야 한다 예) [8.94, 2.76, -6.52]이 값을
#전체 합이 1이 되도록 확률을 조정해야 한다 softmax 함수
# 출력값을 확률로 모두 합쳐서 1이 되도록 조정
# [0.54, 0.36, 0.1] 기타가 될 확률이 높다
model = tf.nn.softmax( L ) #tensorflow 연산들

#손실함수를 이용해서 출력결과하고 얼마만큼의 오차가 나는지를
#손실함수 - 교차엔트로피
#axis - 축, 0 이면 행, 1이면 열
cost = tf.reduce_mean( -tf.reduce_sum(y * tf.log(model), axis=1))

#최적화를 한다.(optimizer를 이용해서 최적의 해를 구한다)
# (경사하강법)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#손실을 최소화한다.
train_op = optimizer.minimize( cost )


#####################
# 신경망 모델 학습하기
#####################

# 변수초기화
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#학습을 하자
for step in range(100):
    sess.run( train_op, feed_dict={X:x_data, y:y_data})
    c = sess.run(cost, feed_dict={X:x_data, y:y_data})
    print(step, c)

# 결과 확인
# 0:기타, 1:포유류, 2:조류
# 결과가 softmax 함수에 의해서 0.54, 0.36, 0.1 등의 형태로
# 오는데 이중에서 가장 확률이 가장 큰 값의 인덱스를
# 가져와야 한다 tf.argmax 함수를 이용해 이 값을 가져온다
prediction = tf.argmax( model, 1)#예측값
target = tf.argmax(y, 1) #실제값

print("실제값 ", sess.run(target, feed_dict={y:y_data}))
print("예측값 ", sess.run(prediction, feed_dict={X:x_data}))

# 정확도측정하기
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

print("정밀도", sess.run(accuracy*100, feed_dict={X:x_data, y:y_data}))