import tensorflow as tf

# 在名字为foo的命名空间内创建名字为v的变量
with tf.variable_scope("foo", reuse=False):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))

# 因为在命名空间foo中已经存在名字为v的变量，所以以下代码将会报错
with tf.variable_scope("foo", reuse=False):
    v = tf.get_variable("v", [1])

# 在生成上下文管理器时，将参数reuse设置为True。这样tf.get_variable函数将会直接获取已经声明的变量
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
    print(v == v1)  # True

# 将参数reuse设置为True时，tf.get_variable只能获取已经创建过的变量。
# 因为在命名空间bar内还没有创建变量v，所以以下代码将会报错
with tf.variable_scope("bar", reuse=True):
    v = tf.get_variable("v", [1])


tf.variable_scope嵌套
with tf.variable_scope("root"):
    print(tf.get_variable_scope().reuse)  # False

    with tf.variable_scope("foo", reuse=True):
        print(tf.get_variable_scope().reuse)  # True

        # 新建一个嵌套的上下文管理器但不指定reuse，这是reuse的取值会和外面一层保持一致
        with tf.variable_scope("bar"):
            print(tf.get_variable_scope().reuse)  # True

    # 回到了命名空间root底下，故reuse回到了False
    print(tf.get_variable_scope().reuse)  # False


# tf.variable_scope管理变量命名空间
v1 = tf.get_variable("v", [1])
print(v1.name)  # v:0 （v是变量的名称，0代表这个变量是生成变量这个运算的第一个结果）

with tf.variable_scope("foo"):
    v2 = tf.get_variable("v", [1])
    print(v2.name)  # foo/v:0 （在tf.variable_scope中创建的变量，名称前面会加入命名空间的名称）

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v", [1])
        print(v3.name)  # foo/bar/v:0 （命名空间可以嵌套）

    v4 = tf.get_variable("v1", [1])
    print(v4.name)  # foo/v1:0 （当命名空间退出之后，变量名称也就不会再被加入其前缀了）

# 创建一个名称为空的命名空间，并设置为reuse=True
with tf.variable_scope("", reuse=True):
    v5 = tf.get_variable("foo/bar/v", [1])  # 可以直接通过带命名空间名称的变量名来获取其他命名空间下的变量

    print(v5 == v3)  # True
    v6 = tf.get_variable("foo/v1", [1])
    print(v6 == v4)  # True
