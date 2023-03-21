import tensorflow as tf

# None 크기가 정해지지 않음 n by 5를 전달 하겠다
# [1,2,3,4,5], [[1,2,3,4,5], [4,5,6,7,7]]
#[[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]]
x = tf.placeholder( tf.float32, [None, 5] )
y = x * 2

sess = tf.Session()

# 전달할 배열 만들기 - list타입으로
x_data = [[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]]
temp = sess.run(y, feed_dict={x:x_data})
print(temp)