import torch
import numpy as np
from details import device, epochs, batchsize
from data import train_loader

def trainModel(model, optimizer, criterion):

    model.to(device)
    train_acc = []

    for epoch in range(epochs):

        train_epoch_acc = []

        for inputs, targets in train_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            model.train()

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                y_hat = torch.argmax(output, 1)
                score = torch.eq(y_hat, targets).sum()
                train_epoch_acc.append((score.item()/batchsize)*100)

        with torch.no_grad():
            train_acc.append(np.mean(np.array(train_epoch_acc)))
    
    return max(train_acc)
