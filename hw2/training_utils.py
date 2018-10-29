import numpy as np
import torch
import torch.nn.functional as F


def train_model(model, optimizer, train_loader, criterion):
    model.train()
    loss_train = 0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch['label'])
        loss.backward()
        # zero-out gradient for pretrained embeddings
        model.encoder.embedding.weight.grad[2:] = 0
        optimizer.step()
        loss_train += loss.item() * \
            len(batch['label']) / len(train_loader.dataset)
    return loss_train


def eval_model(model, val_loader, criterion):
    model.eval()
    loss_val = 0
    ys = []
    ys_hat = []
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch)
            loss = criterion(outputs, batch['label'])
            loss_val += loss.item() * \
                len(batch['label']) / len(val_loader.dataset)
            ys.append(batch['label'].cpu().numpy())
            ys_hat.append((F.softmax(outputs, dim=1).cpu().numpy()))
    return loss_val, np.concatenate(ys_hat), np.concatenate(ys)
