import tensorflow as tf
from model import Seq2Seq
from data.read_data import *


def train(epoch):
    model = Seq2Seq(batch_size=100, num_words=10, num_units=128, num_layers=3, output_keep_prob=0.5)

    input_reader = open("data/movie/input_train.txt", "r")
    target_reader = open("data/movie/target_train.txt", "r")
    input_data, target_data = process_data(input_reader, target_reader)
    batch_encoder_input, batch_encoder_num = read_encoder_batch(model.data, input_data, model.num_words,
                                                                model.batch_size)
    batch_decoder_input, batch_decoder_num = read_decoder_input_batch(model.data, target_data, model.num_words,
                                                                      model.batch_size)
    batch_decoder_output, _ = read_decoder_output_batch(model.data, target_data, model.num_words, model.batch_size)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('checkpoint')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("We have a checkpoint!")
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        total_batch = int(model.data_size/model.batch_size)

        for step in range(epoch):
            for batch_step in range(total_batch):
                batch_enc_input, batch_enc_num = sess.run([batch_encoder_input,batch_encoder_num])
                batch_dec_input, batch_dec_num, batch_dec_output = sess.run([batch_decoder_input,batch_decoder_num, batch_decoder_output])

                _, cost = model._train(sess,batch_enc_input,batch_enc_num,batch_dec_input,batch_dec_output,batch_dec_num)
            if(step % 5) == 0:
                print('Step:', '%03d' % model.global_step.eval(),
                      'cost =', '{:.6f}'.format(cost))
                saver.save(sess, 'checkpoint/train', step)

        coord.request_stop()
        coord.join(threads)

def test(batch_size):
    model = Seq2Seq(batch_size=100, num_words=10, num_units=128, num_layers=3, output_keep_prob=0.5)

    input_reader = open("data/movie/input_test.txt", "r")
    target_reader = open("data/movie/target_test.txt", "r")
    input_data, target_data = process_data(input_reader, target_reader)
    batch_encoder_input, batch_encoder_num = read_encoder_batch(model.data, input_data, model.num_words,
                                                                model.batch_size)
    batch_decoder_input, batch_decoder_num = read_decoder_input_batch(model.data, target_data, model.num_words,
                                                                      model.batch_size)
    batch_decoder_output, _ = read_decoder_output_batch(model.data, target_data, model.num_words, model.batch_size)

    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('checkpoint')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("no check point!")
            return

        batch_encoder_input, batch_encoder_num = read_encoder_batch(model.data,input_data, model.num_words, batch_size)
        batch_decoder_input, batch_decoder_num = read_decoder_input_batch(model.data,target_data, model.num_words,
                                                                          batch_size)
        batch_decoder_output, _ = read_decoder_output_batch(model.data,target_data, model.num_words, batch_size)

        total_batch = int(model.data_size / batch_size)

        for batch_step in range(total_batch):
            batch_enc_input, batch_enc_num = sess.run([batch_encoder_input, batch_encoder_num])
            batch_dec_input, batch_dec_num = sess.run([batch_decoder_input, batch_decoder_num])
            batch_dec_output = sess.run([batch_decoder_output])
            outputs, accuracy = model._test(sess, batch_enc_input, batch_enc_num, batch_dec_input, batch_dec_output, batch_dec_num)
            #TO DO, 예측값을 한글로 변환시켜야함

        coord.request_stop()
        coord.join(threads)

def main(train_flag):
    if(train_flag):
        train(epoch=100)
    else:
        test()


if __name__ == "__main__":
    main(1)