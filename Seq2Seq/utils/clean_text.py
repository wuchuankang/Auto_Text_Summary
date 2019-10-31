import pandas as pd
import jieba

# 定义分词，sentence 是一个新闻或者摘要
def tokenizer(sentence):
    return [token for token in jieba.cut(sentence)]

# 获取停用词
def get_stopwords(path):
    with open(path, 'r', encoding='utf-8')as f:
        words = f.readlines()
        return [word.strip() for word in words]

# text 是所有新闻或者摘要，是以列表的形式存储，每一个元素是一个str形新闻或者摘要
def processed_text(text, stopwords):
    # text_tokens 是所有新闻或者摘要分词后存储的列表，其中元素是列表，表示一个新闻或者摘要
    text_tokens = []
    for sentence in text:
        tokens = tokenizer(sentence)
        tokens = [token for token in tokens if token not in stopwords]
        text_tokens.append(tokens)
    return text_tokens

def clean_text(text_path, stopwords_path):
    file = pd.read_csv(text_path)

    news = file.news.values
    summaries = file.summries.values

    stopwords = get_stopwords(stopwords_path)

    processed_news = processed_text(news, stopwords)
    processed_summaries = processed_text(summaries, stopwords)

    return processed_news, processed_summaries

if __name__ ==  "__main__":
    from config import config
    news, summs = clean_text(config['text_path'], config['stopwords_path'])
    print(news)
