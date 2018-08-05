# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell


INIT_RANDOM_RANGE = 0.1


class DMP_Model(object):
    """Discourse Marker Prediction Model"""
    def __init__(self, model_hparams, embeddings):
        self._previous_t_vars = tf.global_variables()

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.max_sentence_len = model_hparams.max_sentence_len
        self.embeddings = embeddings
        self.hidden_size = model_hparams.hidden_size
        self.predict_size = model_hparams.predict_size
        self.project_layer_num = model_hparams.project_layer_num
        self.learning_rate = model_hparams.learning_rate

        self._add_placeholder()
        self._build_graph()
        self._cal_loss()
        self._add_train_op()

        self._after_t_vars = tf.global_variables()
        p_t_vars = set(self._previous_t_vars)
        a_t_vars = set(self._after_t_vars)
        self._this_model_t_vars = a_t_vars - p_t_vars
        self.summary = tf.summary.merge_all()

    def _add_placeholder(self):
        self.sentence_1 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='sentence_1')
        self.sentence_2 = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='sentence_2')
        self.sentence_1_len = tf.placeholder(tf.int32, shape=[None], name="sentence_1_len")
        self.sentence_2_len = tf.placeholder(tf.int32, shape=[None], name="sentence_2_len")
        self.marker = tf.placeholder(tf.int32, shape=[None], name="marker")
        self.dropout_rate = tf.placeholder(tf.float32, shape=[], name="dropout_rate")

    def _cal_loss(self):
        raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.marker, logits=self.logits,
                                                                  name="each_sample_loss")
        self.losses = tf.reduce_mean(raw_loss, name="losses")
        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.arg_max(self.logits, dimension=1), tf.cast(self.marker, tf.int64)), tf.float32),
            name="final_acc")
        tf.summary.scalar('acc', self.acc)
        tf.summary.scalar('loss', self.losses)

    def _build_graph(self):

        def emb_drop(emb_matrix, x):
            emb = tf.nn.embedding_lookup(emb_matrix, x, name="input_emb")
            return tf.nn.dropout(emb, self.dropout_rate, name="input_dropout")

        with tf.name_scope("word_emb"):
            sen_1_x = emb_drop(self.embeddings, self.sentence_1)
            sen_2_x = emb_drop(self.embeddings, self.sentence_2)

        with tf.variable_scope("encoder") as bi_lstm_scope:
            fw_cell = LSTMCell(self.hidden_size, state_is_tuple=True)
            bw_cell = LSTMCell(self.hidden_size, state_is_tuple=True)

            sen_1_encoded, sen_1_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, sen_1_x,
                                                                         self.sentence_1_len,
                                                                         dtype=tf.float32)
            bi_lstm_scope.reuse_variables()
            sen_2_encoded, sen_2_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, sen_2_x,
                                                                         self.sentence_2_len,
                                                                         dtype=tf.float32)

        def maxpool(x):
            return tf.reduce_max(x, axis=-2, name="max_pool")

        with tf.name_scope("max_pooling"):
            sen_1_fw_maxpool = maxpool(sen_1_encoded[0])
            sen_1_bw_maxpool = maxpool(sen_1_encoded[1])
            sen_2_fw_maxpool = maxpool(sen_2_encoded[0])
            sen_2_bw_maxpool = maxpool(sen_2_encoded[1])

        def sen_repst(sen_fw_maxpool, sen_bw_maxpool, final_state_1, final_state_2):
            return tf.concat([sen_fw_maxpool, sen_bw_maxpool, final_state_1, final_state_2], axis=-1, name="represent")

        with tf.name_scope("sentence_representation"):
            sen_1_representation = sen_repst(sen_1_fw_maxpool, sen_1_bw_maxpool, sen_1_state[0].h, sen_1_state[1].h)
            sen_2_representation = sen_repst(sen_2_fw_maxpool, sen_2_bw_maxpool, sen_2_state[0].h, sen_2_state[1].h)

        with tf.name_scope("combination"):
            final_representation = tf.concat([sen_1_representation,
                                              sen_2_representation,
                                              sen_1_representation + sen_2_representation,
                                              sen_1_representation * sen_2_representation],
                                             axis=-1, name="combine")

        with tf.name_scope("projection"):
            input_temp = final_representation
            for i in range(self.project_layer_num - 1):
                with tf.variable_scope("pro_layer_{}".format(i)):
                    mapping_matrix = tf.get_variable("mapping_matrix",
                                                     shape=[input_temp.get_shape().as_list()[-1], 2 * self.hidden_size],
                                                     trainable=True)
                    input_temp = tf.einsum("ij,jk->ik", input_temp, mapping_matrix)
                    input_temp = tf.nn.relu(input_temp, name="activate")
                    input_temp = tf.nn.dropout(input_temp, self.dropout_rate)

            with tf.variable_scope("pro_layer_{}".format(self.project_layer_num - 1)):
                mapping_matrix = tf.get_variable("mapping_matrix",
                                                 shape=[input_temp.get_shape().as_list()[-1], 2 * self.hidden_size],
                                                 trainable=True)

                input_temp = tf.matmul(input_temp, mapping_matrix, name="mapping")
                input_temp = tf.einsum("ij,jk->ik", input_temp, mapping_matrix)
                input_temp = tf.nn.relu(input_temp, name="activate")
                input_temp = tf.nn.dropout(input_temp, self.dropout_rate)

            self.logits = input_temp
            tf.summary.histogram("logits", self.logits)

    def _add_train_op(self):
        self.train_op = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.losses,
                                                                                global_step=self.global_step)
        # self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        # t_vars = tf.trainable_variables()
        # self.gradients = tf.gradients(self.losses, t_vars)
        # self.train_op = self.optimizer.apply_gradients(zip(self.gradients, t_vars),
        #                                                global_step=self.global_step,
        #                                                name="train_op")

    def get_feed_dict(self, batch_data):
        feed_dict = {
            self.sentence_1: batch_data["sen_1_ids"],
            self.sentence_2: batch_data["sen_2_ids"],
            self.sentence_1_len: batch_data["sen_1_lens"],
            self.sentence_2_len: batch_data["sen_2_lens"],
            self.marker: batch_data["marker_ids"],
            self.dropout_rate: batch_data["dropout_keep_rate"]
        }
        return feed_dict

    def get_loss(self):
        return self.losses

    def get_t_vars(self):
        return self._this_model_t_vars

    def get_train_op(self):
        return self.train_op

    def get_logits(self):
        return self.logits

    def get_acc(self):
        return self.acc

    def get_summary(self):
        return self.summary
