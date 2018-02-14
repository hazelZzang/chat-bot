import tensorflow as tf
from model import Seq2Seq
from data.read_data import *


def train(epoch):
    model = Seq2Seq(batch_size=100, num_words=10, num_units=128, num_layers=3, output_keep_prob=0.5)

    input_reader = open("data/movie/input_train.txt", "r")
    target_reader = open("data/movie/target_train.txt", "r")

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
                print(batch_step)
                batch_enc_input, batch_enc_num = read_encoder_batch(model.data,
                                                                     [input_reader.readline() for _ in range(100)],
                                                                     model.num_words)
                batch_dec_input, batch_dec_num, batch_dec_output = read_decoder_batch(model.data,
                                                                                       [target_reader.readline() for _ in range(100)], model.num_words)
                _, cost = model._train(sess,batch_enc_input,batch_enc_num,batch_dec_input,batch_dec_output,batch_dec_num)
            if(step % 5) == 0:
                print('Step:', '%03d' % model.global_step.eval(),
                      'cost =', '{:.6f}'.format(cost))
                saver.save(sess, 'checkpoint/train', step)

        coord.request_stop()
        coord.join(threads)

def test(batch_size):
    model = Seq2Seq(batch_size=100, num_words=10, num_units=128, num_layers=3, output_keep_prob=0.5)

    #input_file = "data/movie/input_test.txt"
    #target_file = "data/movie/target_test.txt"
    input_reader = open("data/movie/input_test.txt", "r", encoding="UTF-8")
    target_reader = open("data/movie/target_test.txt", "r", encoding="UTF-8")

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

        total_batch = int(model.data_size / batch_size)

        for batch_step in range(total_batch):
            batch_enc_input, batch_enc_num = read_encoder_batch(model.data, [input_reader.readline() for _ in range(100)], model.num_words)
            batch_dec_input, batch_dec_num, batch_dec_output = read_decoder_batch(model.data, [target_reader.readline() for _ in range(100)], model.num_words)
            outputs, accuracy = model._test(sess, batch_enc_input, batch_enc_num, batch_dec_input, batch_dec_output, batch_dec_num)
            print(batch_enc_input)
            #TO DO, 예측값을 한글로 변환시켜야함

            print('inputs =', '{}'.format(model.data.idx_to_str(batch_enc_input[0])),
                  'outputs =', '{}'.format(model.data.idx_to_str(outputs[0])),
                  'accuracy =', '{:.6f}'.format(accuracy))

        coord.request_stop()
        coord.join(threads)

def main(train_flag):
    if(train_flag):
        train(epoch=100)
    else:
        test(batch_size = 100)



if __name__ == "__main__":
    main(1)