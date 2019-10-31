"""
注意：
    1. encoder 使用单层双向 GRU, 最后一层的输出不是拼接，而是相加
    2. decoder 使用双层单项 GRU，第一个字符<bos>输入的hidden使用encoder 端的输出初始化的
    3. 为了架构清晰，encoder 和 decoder 的 hidden_siz 取相同值，并且取 hidden_siz = embed_size
""" 
import math
import torch
from torch import nn
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1, dropout=0.0, use_pretrained_embeddings=True, pre_embeddings=None):
        super(EncoderRNN, self).__init__()

        self.input_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        if use_pretrained_embeddings:
            self.embedding = nn.Embedding.from_pretrained(pre_embeddings)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, n_layers, bidirectional=True, dropout=dropout)

    def forward(self, inputs, init_hidden):
        # 输出维度:[batch, seq, embed_size]，因为 GRU 的输入维度是 :[seq, batch, embeddings]，所以需要转置
        embeddings = self.embedding(inputs).permute(1,0,2)
        outputs, hidden = self.gru(embeddings, init_hidden)
        # 将 bidirectional outputs 给加起来，当然也可以不加，但是加起来之后，对于后面的attention和decoder的计算会方便许多
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  
        return outputs, hidden

    def get_init_hidden(self):
        return None


# 定义 attention ，使用点积缩放模型，这里留了 method 参数借口，方便后面进行添加其他 attention 方法。
class Atten(nn.Module):
    def __init__(self, method='dot'):
        super(Atten, self).__init__()

        self.method = method

    def forward(self, dec_hidden, enc_outputs):
        """forward

        :param dec_hidden: 维度是 [batch, hidden_size]
        :param enc_outputs: 维度是 [seq, batch, hidden_size]
        """
        seq_len = enc_outputs.size(0)
        batch = enc_outputs.size(1)
        #定义个 变量energy，用于存储attention，注意要将其放到cuda上
        energy = torch.zeros((seq_len, batch)).to(device)
        for i in range(seq_len):
            energy[i] = self.scores(dec_hidden, enc_outputs[i])

        # 在时间步上做softmax 运算, alpha的维度 [seq, batch]
        alpha = F.softmax(energy, dim=0)
        # 通过将alpha增加一个维度后，可以使用广播机制，context_vector 维度 [batch, hidden_size]
        context_vector = (alpha.unsqueeze(-1) * enc_outputs).sum(dim=0)
        return context_vector


    def scores(self, dec_hidden, enc_output):
        """scores

        :param hidden: [batch, hidden_size]
        :param output: [batch, hidden_size]
        """
        if self.method == 'dot':
            d = enc_output.size(1)
            return (dec_hidden * enc_output).sum(dim=1) / math.sqrt(d)  # 维度是 [batch]



# 默认使用双层，更改层数，要注意更改 get_init_hidden 方法
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=2, dropout=0.1, use_pretrained_embeddings=True, pre_embeddings=None):
        super(DecoderRNN, self).__init__()

        if use_pretrained_embeddings:
            self.embedding = nn.Embedding.from_pretrained(pre_embeddings)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)

        # 有的GRU 第一个参数是 2*hidden_size，这是因为将 hidden_size 取为 embed_size
        # 这里参数相加的原因是 hidden_size对应的 context_vector 和 encoder 端词的embedding 之和 作为输入
        self.atten = Atten()
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, dec_hidden, enc_outputs):
        """forward

        :param inputs: [batch, 1]
        :param dec_hidden: [n_layers, batch, hidden_size]
        :param enc_outputs: [seq, batch, hidden_size]
        """
        # context_vector :[batch, hidden_size] , embeddings :[batch, embed_size]
        context_vector = self.atten(dec_hidden[-1], enc_outputs)
        embeddings = self.embedding(inputs)
        embed_context = torch.cat((embeddings, context_vector), dim=1)
        # gru 要求的输入是[seq, batch, hidden_size]，所以要增加个时间步
        # output : [1, batch, hidden_siz]
        output, dec_hidden = self.gru(embed_context.unsqueeze(0), dec_hidden)
        # 移除时间步，输出：[batch, vocab_size]
        output = self.out(output.squeeze(dim=0))
        return output, dec_hidden


    # 直接使用 encoder 端的输出的hidden作为decoder隐状态的初始化
    # encoder 端双向rnn，hidden输出维度是： [n_layers*num_direction, batch, hidden_size]，即[2, batch, hidden_size]
    def get_init_hidden(self, enc_hidden):
        return enc_hidden


