import tensorflow as tf
import numpy as np
import re

class Data:
    def __init__(self):
        self.padding = ['_pad_', '_sos_','_eos_']
        self.word_dic, self.idx_dic = self.save_dic()
        self.word_num = len(self.word_dic)


    def save_dic(self):
        dic_word = {}
        dic_idx = {}
        reader = open("data/chat.voc", "r")
        words = reader.readlines()

        for i, word in enumerate(self.padding):
            dic_word[word.rstrip()] = i
            dic_idx[i] = word.rstrip()

        for i, word in enumerate(words):
            dic_word[word.rstrip()] = i+3
            dic_idx[i+3] = word.rstrip()
        return dic_word, dic_idx

    def list_to_idx(self, list):
        idx_list = []
        for word in list:
            try:
                idx_list.append(self.word_dic[word.lower()])
            except:
                idx_list.append(0)
        return idx_list

    def str_to_idx(self, words):
        idx_list = []
        words = re.findall('[^\w\s]+|\w+',words.lower())
        for word in words:
            try:
                idx_list.append(self.word_dic[word])
            except:
                idx_list.append(0)
        return idx_list

    def idx_to_str(self, idx_list):
        words = []
        for idx in idx_list:
            try:
                words.append(self.idx_dic[idx])
            except:
                words.append('_None_')
        return words


def put_padding(dic_class, words, length, pad):
    padded_line = []
    line_length = len(words)

    if(pad == 'sos'):
        padded_line.append('_SOS_')
        padded_line += words
        padded_line += ['_PAD_' for i in range(length - line_length - 1)]
        line_length += 1
    elif(pad == 'eos'):
        padded_line += words
        padded_line.append('_EOS_')
        padded_line += ['_PAD_' for i in range(length - line_length - 1)]
        line_length += 1
    else:
        padded_line += words
        padded_line += ['_PAD_' for i in range(length - line_length)]
    return np.array(dic_class.list_to_idx(padded_line)), np.array(line_length)

def filter_line(line):
    #훈련된 임베딩 사용하기
    #unk 로 대체하기
    pass

def process_data(input_reader, target_reader):
    return input_reader.readline(), target_reader.readline()

def read_encoder_batch(dic_class, input_data, max_length, batch_size):
    # encoder_input : [max_length, ]

    encoder_input, encoder_input_length = put_padding(dic_class,input_data, max_length, 'pad')
    # batch_encoder_input : [batch_size, max_length, ]
    batch_encoder_input, batch_encoder_length = tf.train.batch([encoder_input, encoder_input_length],
                                                               batch_size=batch_size)
    return batch_encoder_input, batch_encoder_length


def read_decoder_input_batch(dic_class, target_data, max_length, batch_size):
    decoder_input, decoder_input_length = put_padding(dic_class,target_data, max_length, 'sos')
    batch_decoder_input, batch_decoder_length = tf.train.batch([decoder_input, decoder_input_length],
                                                               batch_size=batch_size)
    return batch_decoder_input, batch_decoder_length


def read_decoder_output_batch(dic_class,target_data, max_length, batch_size):
    decoder_output, decoder_output_length = put_padding(dic_class,target_data, max_length, 'eos')
    batch_decoder_output, batch_decoder_length = tf.train.batch([decoder_output, decoder_output_length],
                                                               batch_size=batch_size)
    return batch_decoder_output, batch_decoder_length
