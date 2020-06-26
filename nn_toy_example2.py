#coding=utf-8
#预测酸奶日销量 y，x1 和 x2 是影响日销量的两个因素。
"""
对于预测酸奶日销量问题，如果预测销量大于实际销量则会损失成本;
如果预测销量小于实际销量则 会损失利润。在实际生活中，往往制造一盒酸奶的成本和销售一盒酸奶的利润是不等价的。
因此，需 要使用符合该问题的自定义损失函数。
分段函数表示损失函数：
若预测结果 y 小于标准答案 y_，损失函数为利润乘以预测结果 y 与标准答案 y_之差;
若预测结果 y 大于标准答案 y_，损失函数为成本乘以预测结果 y 与标准答案 y_之差。
"""

import tensorflow as tf
import  numpy as np
#自定义损失函数
#酸奶
BATCH_SIZ = 8
SEED = 2020
#利润大于成本的情况
COST = 1
PROFIT = 9

rng = np.random.RandomState(SEED)
X = rng.rand(32,2)
Y = [[int(x0+x1<1)] for (x0,x1) in X] #

print("X:\n",X)
print("Y:\n",Y)
#定义输入输出、参数、前向传播的过程
x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))

#销量y = x1+x2
w1 = tf.Variable(tf.random_normal([2,1], stddev = 1, seed = 1))
y = tf.matmul(x,w1)

#定义损失以及反向传播的优化方法
loss = tf.reduce_mean(tf.square(y-y_))#计算均方误差
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y - y_)*COST,(y_ - y)*PROFIT))
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