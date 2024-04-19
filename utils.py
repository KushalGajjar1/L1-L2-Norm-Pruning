import torch
import numpy as np
import matplotlib.pyplot as plt
from data import test_loader
import torch.nn.functional as F
from details import device, batchsize

def get_bottom_indices(values, prune_limit):

    sorted_indices = sorted(range(len(values)), key = lambda k : values[k])
    non_pruned_indices = sorted_indices[prune_limit : ]
    # non_pruned_values = [values[i] for i in non_pruned_indices]

    return non_pruned_indices

def getGraph(valuesX, valuesY):

    x = np.array(valuesX)
    y = np.array(valuesY)
    
    plt.ylabel("Accuracy")
    plt.xlabel("Percent Pruned")
    plt.plot(x, y)
    plt.show()

def getTestAcc(model):

    test_acc = []
    model.eval()

    for inputs_test, targets_test in test_loader:

        inputs_test = inputs_test.to(device)
        targets_test = targets_test.to(device)

        output_test = model(inputs_test)

        y_hat_test = torch.argmax(output_test, 1)
        score_test = torch.eq(y_hat_test, targets_test).sum()
        test_acc.append((score_test.item()/batchsize)*100)

    return np.mean(np.array(test_acc))

def findEntropy(matrix):
    flattern_tensor = matrix.view(-1)
    probabilities = F.softmax(flattern_tensor, dim=0)
    entropy = -torch.sum(probabilities * torch.log2(probabilities))
    return entropy.item()