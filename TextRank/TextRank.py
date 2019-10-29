import math
import numpy as np
import networkx as nx
from utils import sentence_segmentation
from bm25 import BM25

class TextRank():
    def __init__(self, sentences, similarity_type='origin'):
        self.sentences = sentences
        self.type = similarity_type

    def get_similarity(self, sentence, index):
        if self.type == 'origin':
            return self.get_similarity_orgin(sentence, index)
        elif self.type == 'BM25':
            return self.get_similarity_bm25(sentence, index)
        elif self.type == 'word2vec':
            return self.get_similarity_word2vec(sentence, index)
        else:  # 如果不存在，就抛出异常
            print('no exits type, please check the similarity type')
            raise ValueError

    # 计算两个句子的相似度，这里使用的是原始论文中的公式计算两个句子S1和S2的相似度
    def get_similarity_orgin(self, sentence, index):
        """get_similarity

        :param sentence1: 是一个用列表表示的句子，元素是结巴分词后去掉停用词剩余的词
        :param sentence2:
        """
        # 获得两个句子中共同出现词的个数
        words = list(set(sentence + self.sentences[index]))
        co_occur_num = sum([1 for word in words if (word in sentence) and (word in self.sentences[index])])

        # 得到公式中分母值
        denominator = math.log(float(len(sentence))) + math.log(float(len(self.sentences[index])))
        # 如果分母是0， 则返回0
        if abs(denominator) < 1e-12:
            return 0

        return co_occur_num / denominator


    def get_similarity_bm25(self, sentence, index):
        bm25 = BM25(self.sentences)
        return bm25.sim(sentence, index)

    def sort_sentence(self, text_rank_config={'alpha':0.85}):
        sentence_num = len(self.sentences)
        graph = np.zeros((sentence_num, sentence_num))

        # 求转移矩阵
        for i in range(sentence_num):
            for j in range(i, sentence_num):
                # 句子对自己本身的相似度设置为0
                if i == j:
                    graph[i, j] = 0.
                similarity = self.get_similarity(self.sentences[i], j)
                graph[i, j] = similarity
                graph[j, i] = similarity

        # 构建带权(句子相似度)无向图
        nx_grap = nx.from_numpy_matrix(graph)
        # 计算每个句子的textrank值，因为参数 text_rank_config 是字典，所以要使用 ** 来解包
        scores = nx.pagerank(nx_grap, **text_rank_config)   #结果得到的是一个字典
        # 对句子重要性进行降序排列
        sorted_sentences = sorted(scores.items(), key = lambda item: item[1], reverse=True) # 返回的是一个以tuple为元素的列表
        return sorted_sentences

    # 获得摘要，num 表示摘要的个数
    def get_key_sentences(self, num = 3):
        sorted_sentences = self.sort_sentence()

        result = []
        count = 1
        for sentence in sorted_sentences:
            if count > num:
                break
            result.append(self.sentences[sentence[0]])
            count += 1
        # 对句子进行合并
        summaries = ''
        for i, sent in enumerate(result):
            if i==num-1: 
                summaries += ''.join(sent)+'。'
            else:
                summaries += ''.join(sent)+'，'

        return summaries


if __name__=='__main__':
    text = sentence_segmentation('news.txt')
    textrank = TextRank(text, similarity_type='origin')
    summaries = textrank.get_key_sentences(3)
    print(summaries)


