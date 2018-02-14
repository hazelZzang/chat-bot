import tensorflow as tf
from data.read_data import Data
from tensorflow.python.layers.core import Dense

# TO DO
# beam search
# bidirectional cell
# attention

class Seq2Seq:
    def __init__(self, batch_size, num_words, num_units, num_layers, output_keep_prob, data_size = 39613, embedding_size = 512, learning_rate=0.01):
        self.data = Data()
        #train : 39613, test : 16605
        self.data_size = data_size
        self.embedding_size = embedding_size

        self.num_words = num_words        # max words number [My, name, is, Hyeji, 0, 0, 0] -> 7
        self.num_units = num_units        # hidden cell number
        self.num_layers = num_layers        # hidden layer number
        self.vocab_size = self.data.word_num        # data words number

        self.learning_rate = learning_rate

        self.batch_size = batch_size
        self.output_keep_prob = output_keep_prob    # for dropout

        self.encoder_input = tf.placeholder(tf.int64, [None, None,])
        self.encoder_input_length = tf.placeholder(tf.int64, [None])

        self.decoder_input = tf.placeholder(tf.int64, [None, None,])
        self.decoder_input_length = tf.placeholder(tf.int64, [None])

        self.decoder_output = tf.placeholder(tf.int64, [None, None])
        self.decoder_output_length = tf.placeholder(tf.int64, [None])

        self.weights = tf.Variable(tf.ones([self.num_units, self.vocab_size]), name="weights")
        self.bias = tf.Variable(tf.zeros([self.vocab_size]), name="bias")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        self._embedded()

        self.logits, self.cost, self.train_op = self._run_model()
        self.outputs = tf.argmax(self.logits, 2)

    def _run_model(self):
        encoder_outputs, encoder_state = self._encoder()
        decoder_outputs = self._decoder(encoder_state)

        time_steps = tf.shape(decoder_outputs)[1]
        decoder_outputs = tf.reshape(decoder_outputs, [-1, self.num_units])

        logits = tf.matmul(decoder_outputs, self.weights) + self.bias
        logits = tf.reshape(logits, [-1, time_steps, self.vocab_size])

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.decoder_output))
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=self.global_step)

        return logits, cost, train_op

    def _embedded(self):
        initializer = tf.contrib.layers.xavier_initializer()

        self.embedding_matrix = tf.get_variable(
            name="embedding_matrix",
            initializer=initializer,
            shape=[self.vocab_size, self.embedding_size],
            dtype=tf.float32)

    def _cell(self):
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.num_units)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.output_keep_prob)
        return cell

    def _encoder(self):
        encoder_cell = tf.nn.rnn_cell.MultiRNNCell([self._cell() for _ in range(self.num_layers)])

        # Run Dynamic RNN
        #   encoder_outputs: [max_time, batch_size, num_units]
        #   encoder_state: [batch_size, num_units]
        with tf.variable_scope('encode') as encoder_scope:
            # encoder_input_embedded : [batch_size, num_words, embedding_size]
            self.encoder_input_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.encoder_input)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, self.encoder_input_embedded, sequence_length=self.encoder_input_length,  dtype=tf.float32)
        return encoder_outputs, encoder_state

    def _decoder(self, encoder_state):
        decoder_cell = tf.nn.rnn_cell.MultiRNNCell([self._cell() for _ in range(self.num_layers)])
        """
        projection_layer = Dense(
            self.vocab_size, use_bias=False)
        helper = tf.contrib.seq2seq.TrainingHelper(
            self.decoder_input, self.decoder_input_length, time_major=True)
        
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, encoder_state,
            output_layer=projection_layer)
        """
        with tf.variable_scope('decode') as decoder_scope:
            self.decoder_input_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.decoder_input)
            decoder_outputs, _ = tf.nn.dynamic_rnn(decoder_cell, self.decoder_input_embedded, sequence_length=self.decoder_input_length, initial_state=encoder_state, dtype=tf.float32)
        return decoder_outputs

    def _test(self, session, enc_input, enc_len, dec_input, dec_output, dec_len):
        prediction_check = tf.equal(self.outputs, dec_output)
        accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

        return session.run([self.outputs, accuracy], feed_dict={self.encoder_input:enc_input,
                                                                self.encoder_input_length:enc_len,
                                                                self.decoder_input:dec_input,
                                                                self.decoder_input_length:dec_len,
                                                                self.decoder_output:dec_output})

    def _train(self, session, enc_input, enc_len, dec_input, dec_output, dec_len):
        return session.run([self.train_op, self.cost],
                           feed_dict={self.encoder_input:enc_input,
                                      self.encoder_input_length:enc_len,
                                      self.decoder_input:dec_input,
                                      self.decoder_input_length:dec_len,
                                      self.decoder_output:dec_output})

    def _predict(self, session, enc_input, enc_len, dec_input):
        return session.run([self.outputs],
                           feed_dict={self.encoder_input:enc_input,
                                      self.encoder_input_length:enc_len,
                                      self.decoder_input:dec_input})