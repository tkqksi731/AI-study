import tensorflow as tf

#팰리스홀더 (placeholder, 값을 나중에 넣어줄 수 있다.)

# 실수값을 하나 담을 수 있는 공간, 그러나 변수는 아니다
x = tf.placeholder( tf.float32 )
y = tf.placeholder( tf.float32 )
z = tf.placeholder( tf.float32 )

result = x + y + z

sess = tf.Session()

temp = sess.run(result, feed_dict={x:10, y:20, z:30})
print(temp)