import torch


UNK, PAD, BOS, EOS = '<unk>', '<pad>', '<bos>', '<eos>'
# 将新闻和摘要合起来建vocab，出现频率低于 min_freq 不纳入vocab中
def build_vocab(processed_news, processed_summaries, min_freq=3):
    all_content = []
    # 将两者合并到processed_news里
    all_content.extend(processed_news)
    all_content.extend(processed_summaries)
    # 统计词频
    tokens_dict = {}  
    for sent in all_content:
        for token in sent:
            tokens_dict[token] = tokens_dict.get(token, 0) + 1
            
    vocab = {}
    id = 0
    for key in tokens_dict:
        if tokens_dict[key] > min_freq:
            vocab[key] = id
            id += 1
        
    # 将特殊字符添加进词表
    vocab.update({UNK:len(vocab), PAD:len(vocab)+1, BOS:len(vocab)+2, EOS:len(vocab)+3})
    
    return vocab

# 将新闻或摘要转化为 vocab 对应的 id 的形式，注意这里的summary 中初始的 BOS 不在这里添加，在计算中已经指明
def build_dataset(vocab, processed_text, max_len, type='summary'):
    content = []

    for sent in processed_text:
        if type == 'summary':
            if len(sent) < max_len:
                sent.extend([EOS] + [PAD] * (max_len-len(sent)))
            else:
                sent = sent[:max_len] + [EOS]
        else: # 如果是 news
            if len(sent) < max_len:
                sent.extend([PAD] * (max_len - len(sent)))
            else:
                sent = sent[:max_len]

        sent_idx = []
        for word in sent:
            idx = vocab.get(word, vocab[UNK])
            sent_idx.append(idx)
        content.append(sent_idx)
        
    return torch.tensor(content)


# 得到与vocab对应的词向量，这个词向量是后面作为torch.nn.Embedding.from_pretrained(Embedding)中参数Embedding，
# 查看源码知道，这个 Embedding 参数是一个2dim的tensor，维度为(num_embeddings, embedding_dim)，这个Embedding参数
# 会加入到module.parameters()中。所以这里我们要得到这样的一个tensor,vocab中词的索引和embedding中的索引要对应一致。
# sgns.sogou.char 第一行是词向量总数和词向量维度，然后接下来每一行的组成是： 词  词向量 是用空格隔开的
def get_pretrained_embedding(pretrained_vector_path, vocab, vector_dim=300):
    with open(pretrained_vector_path, 'r', encoding='utf-8') as f:
        embeddings = torch.rand(len(vocab), vector_dim)   # 给vocab中的词初始化
        for i, line in enumerate(f.readlines()):
            if i == 0:   # 第一行是词向量总数和词向量维度，所以跳过
                continue
            line = line.strip().split(' ')
            if line[0] in vocab:   
                idx = vocab[line[0]]   #
                emb = [float(x) for x in line[1:]]
                embeddings[idx] = torch.tensor(emb)
        return embeddings


