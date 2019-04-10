import tensorflow as tf
import numpy as np
from layers import conv2d, max_pool2d, fc
from utils import mbgenerator, load_data, write2file
from datetime import datetime

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('train', False, 'If model should be trained.')
flags.DEFINE_bool('restore', True, 'If restore previous model.')
flags.DEFINE_integer('mb_size', 16, 'Size of minibatch')


class SiameseNet(object):

    def __init__(self, img_shape, lr=1e-4):

        self._img_shape = img_shape

        # sample from latent space
        self.img1_ph = tf.placeholder(tf.float32, shape=(None, *img_shape), name='img1-ph')
        self.img2_ph = tf.placeholder(tf.float32, shape=(None, *img_shape), name='img2-ph')
        self.target_ph = tf.placeholder(tf.float32, shape=(None, 1), name='target_ph')
        self.is_training = tf.placeholder(tf.bool, (), name='is_training')

        """Model"""
        with tf.variable_scope('Model'):
            sim1 = self._siamese(self.img1_ph, 'siamese-net')
            sim2 = self._siamese(self.img2_ph, 'siamese-net', reuse=True)

            logits, probs = self._similarity(sim1, sim2, 'similarity-net')

        """Loss"""
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=self.target_ph, name='sigmoid-cross-ent'))
        loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        params = tf.trainable_variables(scope='Model')
        grads = tf.gradients(loss, params)
        grads_and_vars = list(zip(grads, params))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        bn_opt = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Model')

        self.optimize = [optimizer.apply_gradients(grads_and_vars), bn_opt]

        self.probs = probs
        self.loss = loss
        self.sim1 = sim1
        self.sess = tf.get_default_session()
        self.global_step = tf.Variable(0)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

        self.summary = tf.Summary()
        self.merge_op = tf.summary.merge_all()
        print('Logging to:', './logs/' + str(datetime.now()))
        self.writer = tf.summary.FileWriter('./logs/' + str(datetime.now()))

    def _siamese(self, img, scope, reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            x = img
            h1 = conv2d(x, 64, kernel_size=10, strides=1, name='conv1', l2reg=1e-4,
                        training=self.is_training, use_bn=True)
            h1 = max_pool2d(h1, kernel_size=2, strides=2, name='pool1')
            h1 = tf.layers.dropout(h1, rate=.4, training=self.is_training)

            h2 = conv2d(h1, 128, kernel_size=7, strides=2, name='conv2', l2reg=1e-5,
                        training=self.is_training, use_bn=True)
            #h2 = max_pool2d(h2, kernel_size=2, strides=2, name='pool2')
            h2 = tf.layers.dropout(h2, rate=.4, training=self.is_training)

            h3 = conv2d(h2, 256, kernel_size=4, strides=2, name='conv3', l2reg=1e-5,
                        training=self.is_training, use_bn=True)
            #h3 = max_pool2d(h3, kernel_size=2, strides=2, name='pool3')
            h3 = tf.layers.dropout(h3, rate=.4, training=self.is_training)

            h4 = conv2d(h3, 512, kernel_size=4, strides=2, name='conv4', l2reg=1e-5,
                        training=self.is_training, use_bn=True)
            # h4 = max_pool2d(h4, kernel_size=2, strides=2, name='pool4')
            h4 = tf.layers.dropout(h4, rate=.4, training=self.is_training)

            h5 = conv2d(h4, 1024, kernel_size=3, strides=2, name='conv5', l2reg=1e-5,
                        training=self.is_training, use_bn=True)
            h5_flat = tf.layers.flatten(h5, name='flatten1')

            out = fc(h5_flat, 2056, name='out', activation_fn=tf.nn.sigmoid, l2reg=1e-6)

            return out

    def _similarity(self, sim1, sim2, scope, reuse=False):

        with tf.variable_scope(scope, reuse=reuse):
            l1_dist = tf.abs(sim1 - sim2)
            logits = fc(l1_dist, 1, name='logits', activation_fn=lambda x: x)

        return logits, tf.nn.sigmoid(logits)

    def train(self, x_train, y_train, mb_size=3):

        N = x_train.shape[0]
        x_val, y_val = x_train[int(.9 * N):], y_train[int(.9 * N):]
        x_train, y_train = x_train[:int(.9 * N)], y_train[:int(.9 * N)]
        Nval = x_val.shape[0]

        train_data = mbgenerator(x_train, y_train, mb_size)
        val_data = mbgenerator(x_val, y_val, Nval)

        step = self.sess.run(self.global_step)
        while step < 50000:
            imgs1, imgs2, labels = next(train_data)

            fd_map = {self.img1_ph: imgs1, self.img2_ph: imgs2,
                      self.target_ph: labels, self.is_training: True}
            loss, _ = self.sess.run([self.loss, self.optimize], feed_dict=fd_map)

            if step % 100 == 0:
                _str = '*' * 20 + f' {step} ' + '*' * 20
                print(_str)
                print(f'training loss: {loss:.6f}\n')
                print('*' * len(_str))
            if step % 1000 == 0:
                self.sess.run(self.global_step.assign(step))
                self.save_model()
                imgs1, imgs2, labels = next(val_data)
                fd_map = {self.img1_ph: imgs1, self.img2_ph: imgs2,
                          self.target_ph: labels, self.is_training: False}
                loss, _ = self.sess.run([self.loss, self.optimize], feed_dict=fd_map)
                _str = '*' * 20 + f' {step} ' + '*' * 20
                print(_str)
                print(f'validation loss: {loss:.6f}\n')
                print('*' * len(_str))
            step += 1

    def eval(self, x_train, y_train, x_test, y_test, test_names):

        import matplotlib.pyplot as plt
        import pandas as pd

        """extract features"""
        fd_map = {self.img1_ph: x_test, self.is_training: False}
        sim = self.sess.run(self.sim1, feed_dict=fd_map)

        df = pd.DataFrame(index=test_names, data=sim)
        df.to_csv(f'features2_test.csv')
        raise KeyboardInterrupt

        N = y_test.size
        probsarr = []
        imgs2 = x_test
        for i in np.unique(y_train):
            imgs1 = x_train[y_train == i][:30]
            imgs1 = np.concatenate([imgs1 for i in range(N // len(imgs1) + 1)], axis=0)[:N]
            labels = np.float32(y_test == i)[:, None]
            print(np.squeeze(labels))
            fd_map = {self.img1_ph: imgs1, self.img2_ph: imgs2,
                      self.target_ph: labels, self.is_training: False}
            loss, probs = self.sess.run([self.loss, self.probs], feed_dict=fd_map)
            print(loss)
            probsarr.append(probs)
        probs = np.squeeze(np.array(probsarr))
        # print(np.argmax(probs, axis=0), np.max(probs, axis=0))
        cls_probs = np.max(probs, axis=0)/probs.sum(axis=0)
        print(cls_probs)
        preds = np.argmax(probs, axis=0)
        print(np.sum(preds[y_test != 4] == y_test[y_test != 4]) / np.sum(y_test != 4))
        bad_idx = (cls_probs < 1).ravel()
        print(f'ratio: {bad_idx.sum() / N:.2f}')
        print(test_names[bad_idx])
        print(test_names[y_test == 4])
        for _cls, prob, img in zip(preds[bad_idx], cls_probs[bad_idx], x_test[bad_idx]):  # [bad_idx.ravel()]):
            plt.imshow(img)
            plt.title(f'cls: {_cls} || prob: {prob}')
            plt.show()

    def restore_model(self):
        print('RESTORING: {}'.format(tf.train.latest_checkpoint('./saved_model/')))
        self.saver.restore(self.sess, '{}'.format(tf.train.latest_checkpoint('./saved_model/')))

    def save_model(self):
        self.saver.save(self.sess, './saved_model/model', global_step=self.sess.run(self.global_step))

    def freeze_model(self):
        tf.train.write_graph(self.sess.graph.as_graph_def(), './saved_model/', 'model.pbtxt', as_text=True)


def main(_):
    dataset = load_data()
    (x_train, y_train), (x_test, y_test), (train_names, test_names) = dataset()

    np.random.seed(1)

    shape = (105, 105, 3)

    mb_size = FLAGS.mb_size

    with tf.Session().as_default():

        snet = SiameseNet(img_shape=shape)
        if FLAGS.restore:
            snet.restore_model()
        if FLAGS.train:
            snet.train(x_train, y_train, mb_size)
        snet.eval(x_train, y_train, x_test, y_test, test_names)


tf.app.run()
