import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available and torch.has_cudnn:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device("cpu")


class CNNEncoder(nn.Module):
    def __init__(self,
                 emb_size,
                 hidden_size,
                 vocab_size,
                 padding_idx,
                 kernel_size=3,
                 pretrained_embeddings=None):

        super(CNNEncoder, self).__init__()
        self.hidden_size = hidden_size

        padding = kernel_size // 2

        self.embedding = nn.Embedding(
            vocab_size, emb_size, padding_idx=padding_idx)

        self.conv1 = nn.Conv1d(emb_size, hidden_size,
                               kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(
            hidden_size, hidden_size, kernel_size=kernel_size, padding=padding)

        self.init_weights(pretrained_embeddings)

    def init_weights(self, pretrained_embeddings):
        if pretrained_embeddings is not None:
            self.embedding.weight.data = torch.from_numpy(
                pretrained_embeddings).float()

    def forward(self, x, lengths):
        batch_size, seq_len = x.size()
        embed = self.embedding(x).float()
        batch_size, seq_len = x.size()

        hidden = self.conv1(embed.transpose(1, 2)).transpose(1, 2)
        hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(
            batch_size, seq_len, hidden.size(-1))

        hidden = self.conv2(hidden.transpose(1, 2)).transpose(1, 2)

        hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(
            batch_size, seq_len, hidden.size(-1))
        hidden = torch.max(hidden, dim=1)[0]
        return hidden


class RNNEncoder(nn.Module):
    def __init__(self,
                 emb_size,
                 hidden_size,
                 vocab_size,
                 padding_idx,
                 pretrained_embeddings=None,
                 num_layers=1):
        super(RNNEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.GRU(
            emb_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True)

        self.init_weights(pretrained_embeddings)
        self.num_layers = num_layers

    def init_weights(self, pretrained_embeddings):
        if pretrained_embeddings is not None:
            self.embedding.weight.data = torch.from_numpy(
                pretrained_embeddings).float()

    def forward(self, x, lengths):
        batch_size, seq_len = x.size()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        lengths = list(lengths[idx_sort])
        x = x.index_select(0, idx_sort)

        hidden = self.init_hidden(batch_size)
        embed = self.embedding(x)
        embed = torch.nn.utils.rnn.pack_padded_sequence(
            embed, lengths, batch_first=True)
        output, hidden = self.rnn(embed, hidden)

        # sum representations from the both directions
        hidden = torch.sum(hidden, dim=0)
        hidden = hidden.index_select(0, idx_unsort)
        return hidden

    def init_hidden(self, batch_size):
        hidden = torch.randn(self.num_layers * 2, batch_size,
                             self.hidden_size).to(DEVICE)
        return hidden


class InferenceModel(nn.Module):
    def __init__(self, encoder, num_classes, linear_size, dropout=0.0):
        super(InferenceModel, self).__init__()
        self.encoder = encoder
        self.dense1 = nn.Linear(linear_size, linear_size)
        self.dense2 = nn.Linear(linear_size, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        encodings = torch.cat(
            [self.encoder(x['sentence1'], x['length1']),
             self.encoder(x['sentence2'], x['length2'])], dim=1)
        encodings = self.dense1(encodings)
        encodings = F.relu(encodings)
        encodings = self.dropout(encodings)
        logits = self.dense2(encodings)
        return logits
