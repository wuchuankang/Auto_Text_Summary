import torch
from torch.utils.data import TensorDataset
from utils import clean_text, config
from utils import build_vocab, build_dataset, get_pretrained_embedding
from seq2seq import EncoderRNN, DecoderRNN, training

# 清洗文本
cleaned_news, cleaned_summaries = clean_text(config['text_path'], config['stopwords_path'])

# 建立词典
vocab = build_vocab(cleaned_news, cleaned_summaries, min_freq=3)

# 生成 dataset 是DataTensor 格式
news_dataset = build_dataset(vocab, cleaned_news, config['max_len_news'], type='news')
summaries_dataset = build_dataset(vocab, cleaned_summaries, config['max_len_summaries'], type='summaries')
# 合并在一起
dataset = TensorDataset(news_dataset, summaries_dataset)

# 加载预训练的word2vec模型（使用搜狗新闻训练得到的word2vec），维度是300
pre_embeddings = get_pretrained_embedding(config['pretrained_vector_path'], vocab, vector_dim=300)

# 构建模型，选择隐状态和词向量维度相同，都是300
vocab_size = len(vocab)
# encoder 使用的是单层双向gru
encoder = EncoderRNN(vocab_size, 300, 300, n_layers=1, pre_embeddings=pre_embeddings)
# decoder 使用双层单项gru
decoder = DecoderRNN(vocab_size, 300, 300, n_layers=2, pre_embeddings=pre_embeddings)

# 训练模型
training(encoder, decoder, dataset, vocab, config['lr'], config['batch_size'], config['epochs'])






