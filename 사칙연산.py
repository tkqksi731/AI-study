import tensorflow as tf

x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)

add = x + y
sub = x - y
mul = x * y
div = x / y

sess = tf.Session()
x_value = int(input("값1 : "))
y_value = int(input("값2 : "))

result1 = sess.run(add, feed_dict={x:x_value, y:y_value})
result2 = sess.run(sub, feed_dict={x:x_value, y:y_value})
result3 = sess.run(mul, feed_dict={x:x_value, y:y_value})
result4 = sess.run(div, feed_dict={x:x_value, y:y_value})

print(result1)
print(result2)
print(result3)
print(result4)