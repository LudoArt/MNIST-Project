import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SUMMARY_DIR = "F:/mnist/to/log"
BATCH_SIZE = 100
TRAIN_STEPS = 3000


# 生成变量监控信息并定义生成监控信息日志的操作
# 其中var给出了需要记录的张量，name给出了在可视化结果中显示的图表名称，这个名称一般与变量名一致
def variable_summaries(var, name):
    # 将生成监控信息的操作放在同一个命名空间下
    with tf.name_scope('summaries'):
        # 通过tf.summary.histogram函数记录张量中元素的取值分布
        tf.summary.histogram(name, var)

        # 计算变量的平均值，并定义生成平均值信息日志的操作
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        # 计算变量的标准差，并定义生成其日志的操作
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


# 生成一层全链接层神经网络
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # 将同一层神经网络放在一个统一的命名空间下
    with tf.name_scope(layer_name):
        # 声明神经网络边上的权重，并调用生成权重监控信息日志的函数
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights, layer_name + '/weights')

        # 声明神经网络的偏置项，并调用生成偏置项监控信息日志的函数
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
            variable_summaries(biases, layer_name + '/biases')

        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            # 记录神经网络输出节点在经过激活函数之前的分布
            tf.summary.histogram(layer_name + '/pre_activations', preactivate)

        activations = act(preactivate, name='activations')

        # 记录神经网络输出节点在经过激活函数之后的分布
        tf.summary.histogram(layer_name + '/activations', activations)
        return activations


def main(_):
    mnist = input_data.read_data_sets("F:/mnist/to/mnist_data", one_hot=True)
    # 定义输入
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    # 将输入向量还原成图像的像素矩阵，并通过tf.summary.image函数定义将当前的图片信息写入日志的操作
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    hidden1 = nn_layer(x, 784, 500, 'layer1')
    y = nn_layer(hidden1, 500, 10, 'layer2', act=tf.identity)

    # 计算交叉熵并定义生成交叉熵监控日志的操作
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        tf.summary.scalar('cross entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # 计算模型在当前给定数据上的正确率，并定义生成正确率监控日志的操作
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # TensorFlow提供了tf.summary.merge_all函数来整理所有的日志生成操作
    # 在TensorFlow程序执行的过程中只需要运行这个操作就可以将代码中定义的所有日志生成操作执行一次
    # 从而将所有日志写入文件
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        # 初始化写日志的writer，并将当前TensorFlow计算图写入日志
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        tf.global_variables_initializer().run()

        for i in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 运行训练步骤以及所有的日志生成操作，得到这次运行的日志
            summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y_: ys})
            # 将所有日志写入文件，TensorBoard程序就可以拿到这次运行所对应的运行信息
            summary_writer.add_summary(summary, i)

    summary_writer.close()


if __name__ == '__main__':
    tf.app.run()
