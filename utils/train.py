import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F


def train(model, data, args, verbose=True):
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    train_loss_values, train_acc_values = [], []
    val_loss_values, val_acc_values = [], []

    best = np.inf
    bad_counter = 0
    model_path = 'model.pth'
    for epoch in tqdm(range(args.train_epochs), desc='Training', leave=False):
        if epoch == 0:
            print('       |     Trainging     |     Validation     |')
            print('       |-------------------|--------------------|')
            print(' Epoch |  loss    accuracy |  loss    accuracy  |')
            print('-------|-------------------|--------------------|')

        train_loss, train_acc = train_on_epoch(model, optimizer, data)
        train_loss_values.append(train_loss.item())
        train_acc_values.append(train_acc.item())

        val_loss, val_acc = evaluate(model, data, data.val_mask)
        val_loss_values.append(val_loss.item())
        val_acc_values.append(val_acc.item())

        if val_loss_values[-1] < best:
            bad_counter = 0
            log = '  {:3d}  | {:.4f}    {:.4f}  | {:.4f}    {:.4f}   |'.format(epoch + 1,
                                                                               train_loss.item(),
                                                                               train_acc.item(),
                                                                               val_loss.item(),
                                                                               val_acc.item())
            
            torch.save(model.state_dict(), model_path)
            log += ' save model to {}'.format(model_path)
            
            if verbose:
                tqdm.write(log)

            best = val_loss_values[-1]
        else:
            bad_counter += 1

    print('-------------------------------------------------')

    model.load_state_dict(torch.load(model_path))


def train_on_epoch(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)

    train_loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    train_acc = accuracy(output[data.train_mask], data.y[data.train_mask])

    train_loss.backward()
    optimizer.step()

    return train_loss, train_acc


def evaluate(model, data, mask):
    model.eval()

    with torch.no_grad():
        output = model(data.x, data.edge_index)
        loss = F.nll_loss(output[mask], data.y[mask])
        acc = accuracy(output[mask], data.y[mask])

    return loss, acc


def accuracy(output, labels):
    _, pred = output.max(dim=1)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    
    return correct / len(labels)
