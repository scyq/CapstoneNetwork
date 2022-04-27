# -*- coding: utf-8 -*-

import jieba
import torch
import re
import logging

jieba.setLogLevel(logging.INFO)  #关闭jieba输出信息

corpus_file = 'clean_chat_corpus/qa_data.tsv'  #未处理的对话数据集
re_pattern = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")  #分词处理正则, 留下所有中文英文和数字
unknown = '</UNK>'  #unknown字符
eos = '</EOS>'  #句子结束符
sos = '</SOS>'  #句子开始符
padding = '</PAD>'  #句子填充负
max_voc_length = 20000  #字典最大长度, 即最多记录多少个词, 清华心理对话有71430个词语
min_word_appear = 10  #加入字典的词的词频最小值
max_sentence_length = 1200  #最大句子长度, 统计出来最长的句子有4362个字
save_path = 'corpus.pth'  #已处理的对话数据集保存路径


def preprocess():
    print("preprocessing...")
    '''处理对话数据集'''
    qa_pairs = []
    with open(corpus_file, encoding='utf-8') as f:
        lines = f.readlines()
        lens = []  # 为了更精确的调参, lens记录句子的长度
        for line in lines:
            values = line.strip('\n').split('\t')
            assert len(values) == 2, "对话长度不是2, 请检查数据集"
            sentences = []
            for value in values:
                sentence = jieba.lcut(re_pattern.sub(
                    "", value))  # 把所有非中文字符和非数字字符替换为空格并分词
                lens.append(len(sentence))
                sentence = sentence[:max_sentence_length] + [eos]
                sentences.append(sentence)
            # 每个sentences是一个QA对
            qa_pairs.append(sentences)
        lens.sort()
        print("句子的最大长度为{:d}, 最小长度为{:d}, 平均长度为{:2f}, 长度中位数位{:d}".format(
            lens[-1], lens[0],
            sum(lens) / len(lens), lens[(int)(len(lens) / 2)]))

    # 生成字典和句子索引
    word_nums = {}  #统计单词的词频

    def update(word_nums):

        def fun(word):
            word_nums[word] = word_nums.get(word, 0) + 1
            return None

        return fun

    lambda_ = update(word_nums)
    _ = {
        lambda_(word)
        for sentences in qa_pairs for sentence in sentences
        for word in sentence
    }
    #按词频从高到低排序
    word_nums_list = sorted([(num, word) for word, num in word_nums.items()],
                            reverse=True)
    #词典最大长度: max_voc_length 最小单词词频: min_word_appear
    words = [
        word[1] for word in word_nums_list[:max_voc_length]
        if word[0] >= min_word_appear
    ]
    #注意: 这里eos已经在前面加入了words,故这里不用重复加入
    words = [unknown, padding, sos] + words
    word2index = {word: index for index, word in enumerate(words)}
    index2word = {index: word for word, index in word2index.items()}

    index_corpus = [[[
        word2index.get(word, word2index.get(unknown)) for word in sentence
    ] for sentence in pair] for pair in qa_pairs]
    '''
    保存处理好的对话数据集

    index_corpus: list, 其中元素为[Question_Sequence_List, Answer_Seqence_List]
    Question_Sequence_List: e.g. [word_index, word_index, word_index, ...]

    word2index: dict, 单词:索引

    index2word: dict, 索引:单词

    # '''

    clean_data = {
        'corpus': index_corpus,
        'word2index': word2index,
        'index2word': index2word,
        'unknown': unknown,
        'eos': eos,
        'sos': sos,
        'padding': padding,
    }

    torch.save(clean_data, save_path)
    print('save clean data in %s' % save_path)


if __name__ == "__main__":
    preprocess()