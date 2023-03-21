import tensorflow as tf

x_value = int(input("값 : "))
x = tf.Variable( x_value ) # 현재 값 2.0으로 초기화
print(x)

#옛날버전
# init_op = tf.initialize_all_variables()
init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run( init_op )
print( sess.run(x) )