# coding=utf-8
import tensorflow as tf
import numpy as np

class SiameseLSTMw2v(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an word embedding layer (looks up in pre-trained w2v), followed by a biLSTM and Energy Loss layer.
    """
    
    def stackedRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units,lstm_type):
        n_hidden=hidden_units
        n_layers=3
        # Prepare data shape to match `static_rnn` function requirements
        if lstm_type == 'fw_lstm':
            x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
            # print(x)
            # Define lstm cells with tensorflow
            # Forward direction cell

            with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope,reuse=tf.AUTO_REUSE):
                stacked_rnn_fw = []
                for _ in range(n_layers):
                    fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
                    stacked_rnn_fw.append(lstm_fw_cell)
                lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

                outputs, _ = tf.nn.static_rnn(lstm_fw_cell_m, x, dtype=tf.float32)
            # return outputs[-1]
            return tf.reduce_mean(outputs, 0)
        elif lstm_type == 'bi_lstm':
            with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn', reuse=True):
                lstm_fw_cell_list = []
                for _ in range(n_layers):  ## 每一层必须单独分开写前面的构成，不然会导致每层lstm名字相同报错
                    fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
                    lstm_fw_cell_list.append(lstm_fw_cell)
                lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=lstm_fw_cell_list, state_is_tuple=True)

            with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn', reuse=True):
                lstm_bw_cell_list = []
                for _ in range(n_layers):
                    bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout)
                    lstm_bw_cell_list.append(lstm_bw_cell)
                lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=lstm_bw_cell_list, state_is_tuple=True)
            return lstm_fw_cell_m, lstm_bw_cell_m

    def contrastive_loss(self, y,d,batch_size):
        tmp= y *tf.square(d)
        #tmp= tf.mul(y,tf.square(d))
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
        return tf.reduce_sum(tmp +tmp2)/batch_size/2

    def log_loss(self, y, d, batch_size):
        tmp = y * tf.square(d)
        # tmp= tf.mul(y,tf.square(d))
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2
    
    def __init__(
        self, sequence_length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size, trainableEmbeddings,lstm_type):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")
          
        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.constant(0.0, shape=[vocab_size, embedding_size]),
                trainable=trainableEmbeddings,name="W")
            self.embedded_words1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            self.embedded_words2 = tf.nn.embedding_lookup(self.W, self.input_x2)
        print self.embedded_words1
        # Create a convolution + maxpool layer for each filter size

        # with tf.name_scope("output"):
        #     self.out1=self.stackedRNN(self.embedded_words1, self.dropout_keep_prob, "side", embedding_size, sequence_length, hidden_units)
        #     self.out2=self.stackedRNN(self.embedded_words2, self.dropout_keep_prob, "side", embedding_size, sequence_length, hidden_units)
        #     self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1,self.out2)),1,keep_dims=True))
        #     self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.out2),1,keep_dims=True))))
        #     self.distance = tf.reshape(self.distance, [-1], name="distance")

        with tf.name_scope("midoutput"), tf.variable_scope('midoutput'):
            if lstm_type == 'fw_lstm':
                self.out1 = self.stackedRNN(self.embedded_words1, self.dropout_keep_prob, "side", embedding_size,
                                            sequence_length, hidden_units)
                # tf.get_variable_scope().reuse_variables()
                self.out2 = self.stackedRNN(self.embedded_words2, self.dropout_keep_prob, "side", embedding_size,
                                            sequence_length, hidden_units)
            elif lstm_type == 'bi_lstm':
                bilstm_fw, bilstm_bw = self.stackedRNN(None, self.dropout_keep_prob, "side", embedding_size,
                                                       sequence_length, hidden_units,lstm_type)
                x1 = tf.unstack(tf.transpose(self.embedded_words1, perm=[1, 0, 2]))
                outputs_x1, _, _ = tf.contrib.rnn.static_bidirectional_rnn(bilstm_fw, bilstm_bw, x1, dtype=tf.float32)
                self.out1 = tf.reduce_mean(outputs_x1, 0)
                x2 = tf.unstack(tf.transpose(self.embedded_words2, perm=[1, 0, 2]))
                outputs_x2, _, _ = tf.contrib.rnn.static_bidirectional_rnn(bilstm_fw, bilstm_bw, x2,
                                                                           dtype=tf.float32)
                self.out2 = tf.reduce_mean(outputs_x2, 0)

        with tf.name_scope('output'):
            print('output var scope :{}'.format(tf.get_variable_scope().name))
            self.mid_output = tf.abs(tf.subtract(self.out1, self.out2))
            self.output = tf.layers.dense(self.mid_output, 1, activation=tf.nn.sigmoid, name='sigmoid')
            self.output = tf.reshape(self.output, [-1], name='out')

        with tf.name_scope("loss"):
            # self.loss = self.contrastive_loss(self.input_y,self.distance, batch_size)
            self.loss = tf.losses.log_loss(self.input_y, self.output)

        #### Accuracy computation is outside of this class.
        with tf.name_scope("accuracy"):
            # self.temp_sim = tf.subtract(tf.ones_like(self.distance),tf.rint(self.distance), name="temp_sim") #auto threshold 0.5
            self.temp_sim = tf.rint(self.output, name="temp_sim")
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
