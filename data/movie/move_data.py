import re
"""
Separate movie conversations
"""


delimiter = " +++$+++ "
def get_lines_dic():
    global delimiter
    line_reader = open("movie_lines.txt", "rb")
    lines = line_reader.readlines()
    _delimiter = delimiter.encode()
    line_data = {}
    for line in lines:
        line = line.split(_delimiter)
        line_data[line[0]] = str(line[-1].decode('unicode_escape'))
    return line_data

def get_conv_list():
    global delimiter
    all_list = []
    con_reader = open("movie_conversations.txt", "rb")
    conv_lists = con_reader.readlines()
    for conv_list in conv_lists:
        conv_list = str(conv_list)
        conv_list = conv_list.split(delimiter)[-1]
        conv_list = re.findall(r"[\w]+", conv_list)[:-1]
        all_list.append([conv.encode() for conv in conv_list])
    return all_list

def write_movie_conv(train_file, test_file, max_length):
    conv_list = get_conv_list()
    lines_dic = get_lines_dic()
    train_num = int(len(conv_list) * 0.7)

    return write_data(train_file, conv_list,lines_dic,None,train_num,max_length),\
           write_data(test_file, conv_list,lines_dic,train_num,None,max_length)

def write_data(file, conv_list, lines_dic, list_from, list_to, max_length):
    writer_inp = open("input" + file, "w", encoding="UTF-8")
    writer_tar = open("target" + file, "w", encoding="UTF-8")
    count = 0
    for conv in conv_list[list_from:list_to]:
        for num in range(len(conv) - 1):
            input_data = lines_dic[conv[num]]
            target_data = lines_dic[conv[num + 1]]
            input_words = re.findall('[^\w\s]+|\w+', input_data)
            target_words = re.findall('[^\w\s]+|\w+', target_data)

            if len(input_words) < max_length and len(target_words) < max_length:
                count += 1
                writer_inp.write(lines_dic[conv[num]])
                writer_tar.write(lines_dic[conv[num + 1]])
    return count

if __name__ == "__main__":
    print(write_movie_conv("_train.txt","_test.txt", 10))