
# posseg 是 既给出 词性(part of speech)，又给分词
import jieba.posseg as pseg


allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']
# 依据 sentence_delimiters 中的符号分句子
def sentence_segmentation(path, allow_speech_tags=allow_speech_tags):
    # 用于分句子
    sentence_delimiters = ['，', ',', '?', '!', ';', '？', '！', '。', '；', '……', '…', '\n']
    # 用于保留句子中的成分

    with open(path, 'r') as f:
        text = f.readlines()

    for sep in sentence_delimiters:
        segs = []
        for line in text:
            segs += line.split(sep)
        text = [sentence for sentence in segs if len(sentence)>0]

    sentences = filter_words(text, allow_speech_tags)
    return sentences


# 分句后去除停用词
def filter_words(text, allow_speech_tags):
    # 读取停用词
    with open('stopword.txt','r') as f:
        stopwords = [word.strip() for word in f.readlines()]

    processed_text = []
    for sentence in text:
        sentence = pseg.cut(sentence)
        # filter 停用词
        sentence = [sen for sen in sentence if sen.word not in stopwords]
        # 过滤掉不必要的词性
        sentence = [sen.word for sen in sentence if sen.flag in allow_speech_tags]
        if len(sentence) > 0:
            processed_text.append(sentence)

    return processed_text


if __name__ == "__main__":
    sentences = sentence_segmentation('news.txt')
    print(sentences)

