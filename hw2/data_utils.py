from collections import Counter, defaultdict

import numpy as np
import torch

MAX_SENT_LENGTH = 70
MAX_VOCAB_SIZE = 30000
PAD_IDX = 0
UNK_IDX = 1
SEED = 24
LABELS_DICT = {'contradiction': 0, 'entailment': 1, 'neutral': 2}

if torch.cuda.is_available and torch.has_cudnn:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device("cpu")


def tokenize_dataset(dataset):
    # Split both sentences into tokens and remove punctuation
    all_tokens1 = []
    all_tokens2 = []

    for i, row in dataset.iterrows():
        tokens1 = row['sentence1'].lower().strip().replace(
            '[^\w\s]', '').split()
        all_tokens1.append(tokens1)
        tokens2 = row['sentence2'].lower().strip().replace(
            '[^\w\s]', '').split()
        all_tokens2.append(tokens2)

    return all_tokens1, all_tokens2


def build_vocab(all_tokens, embeddings):
    # Build vocabulary
    token_counter = Counter(
        token for sentence in all_tokens for token in sentence if token in embeddings)
    vocab, count = zip(*token_counter.most_common(MAX_VOCAB_SIZE))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2, 2 + len(vocab))))

    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX
    token2id['<unk>'] = UNK_IDX

    return token2id, id2token


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data1, data2, labels, token2id):
        self.data1 = data1
        self.data2 = data2
        self.labels = labels
        self.token2id = token2id
        assert (len(self.data1) == len(self.labels))
        assert (len(self.data2) == len(self.labels))
        self.token2id = token2id

    def sent2index(self, sentence):
        tokens_ids = [self.token2id.get(
            token, self.token2id['<unk>']) for token in sentence]
        return tokens_ids

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, idx):
        indices1 = self.sent2index(self.data1[idx])[:MAX_SENT_LENGTH]
        indices2 = self.sent2index(self.data2[idx])[:MAX_SENT_LENGTH]
        label = LABELS_DICT[self.labels[idx]]
        return (indices1, indices2), (len(indices1), len(indices2)), label


def text_collate_func(batch):
    data_dict = defaultdict(list)
    label_list = []
    length_dict = defaultdict(list)

    for datum in batch:
        label_list.append(datum[2])
        length_dict[0].append(datum[1][0])
        length_dict[1].append(datum[1][1])

    for datum in batch:
        for i, sentence in enumerate(datum[0]):
            padded_vec = np.pad(
                np.array(sentence),
                pad_width=((0, MAX_SENT_LENGTH - datum[1][i])),
                mode="constant",
                constant_values=0)
            data_dict[i].append(padded_vec)
    data_dict[0] = np.array(data_dict[0], dtype=int)
    data_dict[1] = np.array(data_dict[1], dtype=int)
    length_dict[0] = np.array(length_dict[0])
    length_dict[1] = np.array(length_dict[1])
    label_list = np.array(label_list)
    return {
        'sentence1': torch.from_numpy(data_dict[0]).to(DEVICE),
        'sentence2': torch.from_numpy(data_dict[1]).to(DEVICE),
        'length1': torch.LongTensor(length_dict[0]).to(DEVICE),
        'length2': torch.LongTensor(length_dict[1]).to(DEVICE),
        'label': torch.LongTensor(label_list).to(DEVICE)
    }
