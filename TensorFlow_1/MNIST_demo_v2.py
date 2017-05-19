# coding=UTF-8

import tensorflow as tf
import os
import urllib
import numpy
import gzip

"""
构建两层卷积网络模型，预测准确率为99.2%
"""
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def maybe_download(filename, work_directory):
    """Download the data from Yann's website, unless it's already here"""
    #  判断目录文件是否存在，不存在则创建该目录
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    # 需要读取的文件路径
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return filepath


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


# 将稠密标签向量变成稀疏的标签矩阵
# eg：若原向量的第i行为3，则对应稀疏矩阵的第i行下标为3的值为1，其余为0
def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    # labels_dense.ravel()将整个数组展成一个一维数组
    # labels_dense.flat[i]即将labels_dense看成一个一维数组，取其第i个变量
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1  # 报错？
    return labels_one_hot


def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels


class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape, labels.shape))
            self._num_examples = images.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)

            assert images.shape[3] == 1
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(numpy.float32)
            images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in xrange(784)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # 若当前训练读取的index>总体的images数时，则读取读取开始的batch_size大小的数据
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, fake_data=False, one_hot=False):
    class DataSets(object):
        pass

    data_sets = DataSets()
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    VALIDATION_SIZE = 5000
    local_file = maybe_download(TRAIN_IMAGES, train_dir)
    train_images = extract_images(local_file)

    local_file = maybe_download(TRAIN_LABELS, train_dir)
    train_labels = extract_labels(local_file, one_hot=one_hot)

    local_file = maybe_download(TEST_IMAGES, train_dir)
    test_images = extract_images(local_file)

    local_file = maybe_download(TEST_LABELS, train_dir)
    test_labels = extract_labels(local_file, one_hot=one_hot)

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]

    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    data_sets.train = DataSet(train_images, train_labels)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)
    return data_sets


# 权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置量初始化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积和池化
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 读取训练和测试样本数据
mnist = read_data_sets("MNIST_data/", one_hot=True)
""" 
模型需要输入的图数据，此时定义x作为一个占位符（placeholder)，在运行计算时输入这个值
定义x占位符的数据形式：希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。
我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。（这里的None表示此张量的第一个维度可以是任何长度的）
"""
x = tf.placeholder(tf.float32, [None, 784])
"""
模型还需要权重值和偏置量，同上面的x一样，可以使用占位符表示；
但TensorFlow有更好的方式表示它们：Variable。
一个Variable代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。
它们可以用于计算输入值，也可以在计算中被修改。对于各种机器学习应用，一般都会有模型参数，可以用Variable表示。

下面，我们都用全为零的张量来初始化W和b，因为我们要学习W和b的值，它们的初值可以随意设置。
"""
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

"""
定义一个占位符用于输入实际的分布
"""
y_ = tf.placeholder(tf.float32, [None, 10])

# 第一层卷积：它由一个卷积接一个max pooling完成
"""
input是28*28*1的图像（宽28，高28，由于是黑白色，所以深度为1）.
卷积在每个5x5的patch中算出32组特征（ 卷积层的Filter个数是一个超参数，可以自由设定）。
卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道(FeatureMap)数目。
而对于每一个输出通道都有一个对应的偏置量。
"""
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
"""
为了用第一层卷积，我们把输入的图片数据x变成一个4d向量，其第2、第3维对应图片的宽、高，
最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
"""
x_image = tf.reshape(x, [-1, 28, 28, 1])
""" x_image和权值向量进行卷积,加上偏置项，然后应用ReLU激活函数 """
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
""" 进行max pooling """
h_pool1 = max_pool_2x2(h_conv1)

"""
经过第一层卷积后，我们看下output是多大。由于我们采用的是SAME的padding算法，
我们按照上面的计算公式计算到上下左右各需要增加2位，所以输入从28*28*1，填充为32*32*1。

由于滤波器大小为5，且步幅为1，所以，32 -5 +1 = 28，即经过第一轮卷积后output为28*28*32。
然后再经过max pool池化话，就变为14*14*32了。
"""

# 第二层卷积
""" 每个5x5的patch会得到64个特征 """
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
""" x_image和权值向量进行卷积,加上偏置项，然后应用ReLU激活函数 """
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
""" 进行max pooling """
h_pool2 = max_pool_2x2(h_conv2)

"""
我们再次来计算下第二层卷积和池化之后，输出的shape是什么。同样我们计算出padding为SAME时，
上下左右需要padding的大小为2，所以第二层卷积的输入是18*18*32。当滤波器大小为5，strides为1时，18-5+1=14，
则第二层卷积后的shape是14*14*64。最后经过max pool池化后，output的shap是7*7*64。
"""

# 密集连接层
"""图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片"""
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
"""
我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
"""
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# Dropout
""" 
为了减少过拟合，我们在输出层之前加入dropout 
我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。 
TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。
所以用dropout的时候可以不用考虑scale。
"""
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

"""
实现模型 y = softmax(Wx + b)，y_conv用于保存预测的概率分布 """
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# model training

""" 
定义一个评价模型好坏的指标：交叉熵（cross-entropy）
通俗理解：交叉熵用来衡量我们的预测用于描叙真相的低效性
"""
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
"""
用TensorFlow来训练模型是非常容易的。因为TensorFlow拥有一张描述你各个计算单元的图，
它可以自动地使用反向传播算法(backpropagation algorithm)来有效地确定你的变量是如何影响你想要最小化的那个成本值的。
然后，TensorFlow会用你选择的优化算法来不断地修改变量以降低成本。

我们会用更加复杂的ADAM优化器来做梯度最速下降。
"""
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
"""
评估模型性能  tf.argmax 是一个非常有用的函数，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签。
比如tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签
"""
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
"""
correct_prediction表示的是一组布尔值。为了确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。
例如，[True, False, True, True] 会变成 [1,0,1,1] ，取平均值后得到 0.75.
"""
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
"""
这里，我们使用更加方便的InteractiveSession类。通过它，你可以更加灵活地构建你的代码。
它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的。
这对于工作在交互式环境中的人们来说非常便利，比如使用IPython。如果你没有使用InteractiveSession，
那么你需要在启动session之前构建整个计算图，然后启动该计算图。
"""
sess = tf.InteractiveSession()
"""
初始化所有创建的变量,并启动模型。
feed_dict中加入额外的参数keep_prob来控制dropout比例 ；训练次数20000次，每100次迭代输出一次日志
"""
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d, training accuracy %g" % (i, train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
