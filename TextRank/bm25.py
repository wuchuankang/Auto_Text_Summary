
"""
用 BM25 算法来计算句子相似度
"""
import math

class BM25(object):

    def __init__(self, sentences):
        self.D = len(sentences)
        self.avgdl = float(sum([len(doc) for doc in sentences])) / len(sentences)
        self.sentences = sentences
        self.f = []  # 列表的每一个元素是一个dict，dict存储着一个句子中每个词的出现次数
        self.df = {} # 存储每个词及出现了该词的句子的数量
        self.idf = {} # 存储每个词的idf值
        self.k1 = 1.5
        self.b = 0.75
        self.init()

    def init(self):
        for sent in self.sentences:
            tmp = {}
            for word in sent:
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个句子中每个词的出现次数
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1 # 计算df，即出现该词的句子数目
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D-v+0.5)-math.log(v+0.5)

    def sim(self, sentence, index):
        score = 0
        for word in sentence:
            if word not in self.f[index]:
                continue
            d = len(self.sentences[index])
            score += (self.idf[word]*self.f[index][word]*(self.k1+1)
                      / (self.f[index][word]+self.k1*(1-self.b+self.b*d
                                                      / self.avgdl)))
        return score
