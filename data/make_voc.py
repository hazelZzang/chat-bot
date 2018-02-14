import re
"""
Make wordlists
"""
read_file = "movie/movie_conv.txt"
write_file = "chat.voc"

words = {}
with open(read_file, "r") as r:
    lines = r.readlines()
    for line in lines:
        if(line == " +++$+++ \n"): continue
        line = line.lower()
        for word in re.findall('\W+|\w+',line.rstrip()):
            try:
                words[word.rstrip()] += 1
            except:
                words[word.rstrip()] = 0


with open(write_file, "w") as w:
    words = sorted([(value, key) for (key, value) in words.items()], reverse=True)
    print(words)
    for word in words:
        if (word[0] > 10):
            w.write(word[1] + "\n")
