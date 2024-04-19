import torch.nn as nn
import torch.nn.functional as F
import torchvision
from details import output

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)

        self.fc1 = nn.Linear(50*4*4, 800)
        self.fc2 = nn.Linear(800, 500)
        self.fc3 = nn.Linear(500, 3)

    def forward(self, x):
        layer1 = F.max_pool2d(F.relu(self.conv1) , (2, 2))
        layer2 = F.max_pool2d(F.relu(layer1) , (2, 2))
        layer2_p = layer2.view(-1, int(layer2.nelement()/layer2.shape[0]))
        layer3 = F.relu(self.fc1(layer2_p))
        layer4 = F.relu(self.fc2(layer3))
        layer5 = self.fc3(layer4)
        return layer5
    
def getLeNetModel():
    model = LeNet()
    return model

def getModels(modelName):

    if modelName == 'vgg16':
        model = torchvision.models.vgg16()
        model.classifier[6] = nn.Linear(4096, output)
    
    elif modelName == 'vgg19':
        model = torchvision.models.vgg19()
        model.classifier[6] = nn.Linear(4096, output)

    elif modelName == 'alexnet':
        model = torchvision.models.alexnet()
        model.classifier[6] = nn.Linear(4096, output)

    return model