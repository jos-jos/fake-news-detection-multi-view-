import torch
import torch.nn as nn
from torchtext.vocab import GloVe


class MLP(nn.Module):
    def __init__(self, vocab, embed_size=50):
        super().__init__()
        self.glove = GloVe(name="6B", dim=50)
        self.unfrozen_embedding = nn.Embedding.from_pretrained(self.glove.get_vecs_by_tokens(vocab.get_itos()), padding_idx=vocab['<pad>'])
        self.frozen_embedding = nn.Embedding.from_pretrained(self.glove.get_vecs_by_tokens(vocab.get_itos()),
                                                             padding_idx=vocab['<pad>'],
                                                             freeze=True)
        self.classify1 = nn.Sequential(
            nn.Linear(256*50, 8*25),
            nn.ReLU(),
            nn.Linear(8*25, 16*5),
        )
        self.classify2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(16*10, 8*5),
            nn.Linear(8*5, 8),
        )
        # dropout，提升模型泛化能力
        # 逻辑回归做二分类
        self.apply(self._init_weights)

    def forward(self, x):
        x_unfrozen = self.unfrozen_embedding(x).flatten(1)
        x_frozen = self.frozen_embedding(x).flatten(1)

        x_unfrozen = self.classify1(x_unfrozen)
        x_frozen = self.classify1(x_frozen)

        # 将向量拼接起来后得到一个更长的向量
        feature_vector = torch.cat((x_unfrozen, x_frozen), dim=1)  # (batch_size, 600)
        output = self.classify2(feature_vector)  # (batch_size, 2)
        return output

    def _init_weights(self, m):
        # 仅对线性层和卷积层进行xavier初始化
        if type(m) in (nn.Linear, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
