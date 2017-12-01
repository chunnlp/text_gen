import time

import numpy as np
import tensorflow as tf
import data_reader

from tensorflow.python.client import device_lib

flags = tf.flags
logging = tf.logging

flags.DEFINE_string('model', 'medium', 'model config')
flags.DEFINE_string('data_path', 'data', 'path to data')
flags.DEFINE_string('save_path', 'model', 'path to save model')
flags.DEFINE_integer('num_gpus', 1, 'number of gpus')
flags.DEFINE_string('rnn_mode', None, 'rnn type')
flags.DEFINE_string('mode', 'train', 'train or test')

FLAGS = flags.FLAGS
BASIC = 'basic'
CUDNN = 'cudnn'
BLOCK = 'block'


class DataInput(object):
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps, name=name)


class Model(object):
    def __init__(self, is_training, config, input_, graph):
        self._is_training = is_training
        self._input = input_
        self._rnn_params = None
        self._cell = None
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        self.graph = graph

        with self.graph.as_default():
            with tf.device('/cpu:0'):
                embedding = tf.get_variable(
                    'embedding', [vocab_size, hidden_size], dtype=tf.float32)
                inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

            if is_training and config.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, config.keep_prob)

            output, state = self._build_rnn_graph(inputs, config, is_training)

            softmax_w = tf.get_variable(
                'softmax_w', [hidden_size, vocab_size], dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b', [vocab_size], dtype=tf.float32)
            logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
            logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])

            loss = tf.contrib.seq2seq.sequence_loss(
                logits,
                input_.targets,
                tf.ones([self.batch_size, self.num_steps], dtype=tf.float32),
                average_across_timesteps=False,
                average_across_batch=True)

            self._cost = tf.reduce_sum(loss)
            self._final_state = state

            if not is_training:
                return

            self._lr = tf.Variable(0., trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),
                                          config.max_grad_norm)
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
            self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.train.get_or_create_global_step())

            self._new_lr = tf.placeholder(
                tf.float32, shape=[], name='new_learning_rate')
            self._lr_update = tf.assign(self._lr, self._new_lr)

            self.saver = tf.train.Saver(tf.global_variables())

    def _get_lstm_cell(self, config, is_training):
        if config.rnn_mode == BASIC:
            return tf.contrib.rnn.BasicLSTMCell(
                config.hidden_size, forget_bias=0., state_is_tuple=True,
                reuse=not is_training)
        if config.rnn_mode == BLOCK:
            return tf.contrib.rnn.LSTMBlockCell(
                config.hidden_size, forget_bias=0.)
        raise ValueError('rnn_mode {} not supported'.format(config.rnn_mode))


    def _build_rnn_graph(self, inputs, config, is_training):
        def make_cell():
            cell = self._get_lstm_cell(config, is_training)
            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)
            return cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(config.batch_size, tf.float32)
        state = self._initial_state

        outputs = []
        with tf.variable_scope('RNN'):
            for time_step in range(self.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state


    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


    def with_prefix(self, prefix, name):
        return '/'.join((prefix, name))


    def export_ops(self, name):
        self._name = name
        ops = {self.with_prefix(self._name, 'cost'): self._cost}
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
            if self._rnn_params:
                ops.update(rnn_params=self._rnn_params)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self._initial_state_name = self.with_prefix(self._name, 'initial')
        self._final_state_name = self.with_prefix(self._name, 'final')
        for state_tuple in self._initial_state:
            tf.add_to_collection(self._initial_state_name, state_tuple.c)
            tf.add_to_collection(self._initial_state_name, state_tuple.h)
        for state_tuple in self._final_state:
            tf.add_to_collection(self._final_state_name, state_tuple.c)
            tf.add_to_collection(self._final_state_name, state_tuple.h)


    def import_state_tuples(self, state_tuples, name, num_replicas):
        restored = []
        for i in range(len(state_tuples) * num_replicas):
            c = tf.get_collection_ref(name)[2 * i + 0]
            h = tf.get_collection_ref(name)[2 * i + 1]
            restored.append(tf.contrib.rnn.LSTMStateTuple(c, h))
        return tuple(restored)


    def import_ops(self):
        if self._is_training:
            self._train_op = tf.get_collection_ref('train_op')[0]
            self._lr = tf.get_collection_ref('lr')[0]
            self._new_lr = tf.get_collection_ref('new_lr')[0]
            self._lr_update = tf.get_collection_ref('lr_update')[0]
            rnn_params = tf.get_collection_ref('rnn_params')
            if self._cell and rnn_params:
                params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
                    self._cell,
                    self._cell.params_to_canonical,
                    self._cell.canonical_to_params,
                    rnn_params,
                    base_variable_scope='Model/RNN')
                tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
        self._cost = tf.get_collection_ref(self.with_prefix(self._name, 'cost'))[0]
        num_replicas = FLAGS.num_gpus if self._name == 'Train' else 1
        self._initial_state = self.import_state_tuples(
            self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = self.import_state_tuples(
            self._final_state, self._final_state_name, num_replicas)


    @property
    def input(self):
        return self._input


    @property
    def initial_state(self):
        return self._initial_state


    @property
    def cost(self):
        return self._cost


    @property
    def final_state(self):
        return self._final_state


    @property
    def lr(self):
        return self._lr


    @property
    def train_op(self):
        return self._train_op


    @property
    def initial_state_name(self):
        return self._initial_state_name


    @property
    def final_state_name(self):
        return self._final_state_name


class MediumConfig(object):
    init_scale = 0.05
    learning_rate = 1.
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


class LargeConfig(object):
    init_scale = 0.04
    learning_rate = 1.
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


def run_epoch(session, model, eval_op=None, verbose=False):
    start_time = time.time()
    costs = 0.
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        'cost': model.cost,
        'final_state': model.final_state
    }

    if eval_op is not None:
        fetches['eval_op'] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[h] = state[i].c
            feed_dict[c] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals['cost']
        state = vals['final_state']

        costs += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print('{:.3f} perplexity: {:.3f} speed: {:.0f} wps'.format(
            step * 1. / model.input.epoch_size, np.exp(costs / iters),
            iters * model.input.batch_size * max(1, FLAGS.num_gpus) / (time.time() - start_time)))

    return np.exp(costs / iters)


def get_config():
    config = None
    if FLAGS.model == 'medium':
        config = MediumConfig()
    elif FLAGS.model == 'large':
        config = LargeConfig()
    else:
        raise ValueError('Invalid model: {}'.format(FLAGS.model))
    if FLAGS.rnn_mode:
        config.rnn_mode = FLAGS.rnn_mode
    if FLAGS.num_gpus != 1 or tf.__version__ < '1.3.0':
        config.rnn_mode = BASIC
    return config


def main(_):
    if not FLAGS.data_path:
        raise ValueError('data_path must be set')
    gpus = [
        x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'
    ]

    if FLAGS.num_gpus > len(gpus):
        raise ValueError('Invalid num_gpus')

    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    train_graph = tf.Graph()
    eval_graph = tf.Graph()
    infer_graph = tf.Graph()
    with train_graph.as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope('Train'):
            train_input = DataInput(config=config, data=train_data, name='TrainInput')
            with tf.variable_scope('Model', reuse=None, initializer=initializer):
                m = Model(is_training=True, config=config, input_=train_input, graph=train_graph)
            tf.summary.scalar('Training Loss', m.cost)
            tf.summary.scalar('Learning rate', m.lr)

    latest_ckpt = tf.train.latest_checkpoint(FLAGS.save_path)

    with train_graph.as_default():
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        config_proto = tf.ConfigProto(log_device_placement=False,
                                      allow_soft_placement=True)
        with sv.managed_session(config=config_proto) as train_sess:
        #with tf.Session(config=config_proto) as train_sess:
            train_sess.run(tf.global_variables_initializer())
            for i in range(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.)
                m.assign_lr(train_sess, config.learning_rate * lr_decay)
                train_perplexity = run_epoch(train_sess, m, #eval_op=m.train_op,
                                             verbose=True)
                print('Epoch {} Train Perplexity: {:.3f}'.format(i + 1,
                                                                 train_perplexity))
                if i % 5 == 0:
                    sv.saver.save(train_sess, FLAGS.save_path,
                                  global_step=sv.global_step)

if __name__ == '__main__':
    tf.app.run()
