#coding=utf-8

import tensorflow as tf
import  numpy as np

BATCH_SIZ = 8
SEED = 2020

rng = np.random.RandomState(SEED)
X = rng.rand(32,2)
Y = [[int(x0+x1<1)] for (x0,x1) in X] #

print("X:\n",X)
print("Y:\n",Y)
#定义输入输出、参数、前向传播的过程
x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))

w1 =  tf.Variable(tf.random_normal([2,3], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3,1], stddev = 1, seed = 1))
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#定义损失以及反向传播的优化方法
loss = tf.reduce_mean(tf.square(y-y_))#计算均方误差
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer() #初始化所有参数
    sess.run(init_op)

    #输出未经训练的参数
    print("w1:\n",sess.run(w1))
    print("w2:\n",sess.run(w2))

    #训练模型
    STEPS = 3000
    for i in range(STEPS):
        start = (i*BATCH_SIZ) % 32
        end = start + BATCH_SIZ

        sess.run(train_step,feed_dict = {x:X[start:end],y_:Y[start:end]})
        if i % 500 == 0 :
            total_loss = sess.run(loss,feed_dict = {x:X,y_:Y})
            print("After %d training step(s),loss on all data is %g" %(i,total_loss))

    #打印训练后的参数取值
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
    tf.greater