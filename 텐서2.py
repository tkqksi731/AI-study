import tensorflow as tf

#상수만들기
a = tf.constant( 10 )
b = tf.constant( 20 )
#텐서는 작동안되는 데이터 플로우 차트일뿐
#그래서 세션이라는 객체를 만들어서 이 객체에 필요한
#텐서를 담은 다음 실핼을 시켜야한다
sess = tf.Session()
result = sess.run( a + b )
# run 함수가 실행한 텐서의 값을 반환한다

print(result)

