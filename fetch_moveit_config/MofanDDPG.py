import tensorflow as tf
import numpy as np

#####################  hyper parameters  ####################

LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 300
BATCH_SIZE = 32


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, (224*224*4+3)*2+4), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound[1]
        self.S1 = tf.placeholder(tf.float32, [None, s_dim], 's1')
        self.S2 = tf.placeholder(tf.float32, [None, 224, 224, 4], 'view')
        self.S1_ = tf.placeholder(tf.float32, [None, s_dim], 's1_')
        self.S2_ = tf.placeholder(tf.float32, [None, 224, 224, 4], 'view_')

        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S1, self.S2, scope='eval', trainable=True)
            a_ = self._build_a(self.S1_, self.S2_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S1, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S1_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        if np.random.uniform() < 0.9*((self.pointer - 300) if self.pointer <= 300 else 1):
            end_position = np.array(s[0])
            image = np.array(s[1]).reshape(-1, 224, 224, 4)   # numpy.ndarray todo 1
            temp = self.sess.run(self.a, {self.S1: end_position[None, :], self.S2: image})
            print "the goal position from net :", temp
            return temp[0]
        else:
            action = np.random.uniform(low=-1.5, high=1.5, size=3)
            print "the goal position with random :", action[np.newaxis, :]
            return action[np.newaxis, :]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)
        VIEW_STATE = 224*224*4
        # extract the batch memory from memory repertory
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs1 = bt[:, :self.s_dim].reshape(-1, 3)
        bs2 = bt[:, self.s_dim: self.s_dim + VIEW_STATE].reshape(-1, 224, 224, 4)
        ba = bt[:, self.s_dim + VIEW_STATE: self.s_dim + self.a_dim + VIEW_STATE].reshape(-1, 3)
        br = bt[:, self.s_dim + self.a_dim + VIEW_STATE: self.s_dim + self.a_dim + VIEW_STATE + 1].reshape(-1, 1)
        bs1_ = bt[:, -VIEW_STATE-self.s_dim: -VIEW_STATE].reshape(-1, self.s_dim)
        bs2_ = bt[:, -VIEW_STATE:].reshape(-1, 224, 224, 4)

        self.sess.run(self.atrain, {self.S1: bs1, self.S2: bs2})
        # print self.sess.run(self.conv3, {self.S2_: bs2})
        #
        self.sess.run(self.ctrain, {self.S1: bs1, self.S2: bs2, self.a: ba, self.R: br, self.S1_: bs1_, self.S2_: bs2_})

    def store_transition(self, s, a, r, s_):
        a = np.array(a).reshape(-1, 3)
        if a[0][0] is np.nan:
            return
        s1, s2 = s
        s3, s4 = s_

        if str(type(s3)) == '<type \'numpy.float64\'>':
            s_ = s
        #  s3 == list == numpy.float todo
        s3, s4 = s_
        s1 = np.array(s1).reshape(-1, 3)  # end position
        s2 = np.array(s2).reshape(-1, 224 * 224 * 4)  # camera image rgbd
        r = np.array(r).reshape(-1, 1)  # reward
        s3 = np.array(s3).reshape(-1, 3)  # end position after action
        s4 = np.array(s4).reshape(-1, 224 * 224 * 4)  # camera image(rgbd) after action

        transition = np.hstack((s1, s2, a, r, s3, s4))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:      # indicator for learning
            self.memory_full = True

    def _build_a(self, s, view, scope, trainable):
        with tf.variable_scope(scope):
            # camera image for CNN  raw = 224 * 224 * 4
            view = tf.reshape(view, [-1, 224, 224, 4])
            self.conv1 = tf.layers.conv2d(inputs=view, filters=24,  # 224 * 224 * 24
                                         kernel_size=5, strides=2,
                                         padding='same')  # , activation=tf.nn.relu())
            self.pool1 = tf.layers.max_pooling2d(self.conv1, 2, 2)  # 55 * 55 * 24
            self.conv2 = tf.layers.conv2d(self.pool1, 48, 5, 2, 'same')  # , tf.nn.relu())  # 55 * 55 * 48
            self.pool2 = tf.layers.max_pooling2d(self.conv2, 2, 2)  # 13 * 13 * 48
            self.conv3 = tf.layers.conv2d(self.pool2, 48, 6, 2, 'same')  # , tf.nn.relu())  # 4 * 4 * 48
            self.conv_output = tf.reshape(self.conv3, [-1, 7*7*48])

            # end state for FCN
            print "(shape:)", self.conv_output.shape, s.shape
            temp = tf.concat([s, self.conv_output], 1)
            net = tf.layers.dense(temp, 300, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 300
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './params', write_meta_graph=False)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './params')

