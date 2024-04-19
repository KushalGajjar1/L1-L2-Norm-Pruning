import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from details import batchsize


# data_potato = np.load('Potato_LHE/data_Potato_LHE.npy')
# label_potato = np.load('Potato_LHE/label_Potato_LHE.npy')

# fake_data_potato = np.load('Potato_LHE/Fake_data_Potato_LHE_300.npy')
# fake_label_potato = np.load('Potato_LHE/Fake_label_Potato_LHE_300.npy')

# merge_data_potato = np.concatenate((data_potato, fake_data_potato))
# merge_label_potato = np.concatenate((label_potato, fake_label_potato))

# data_tomato = np.load('Tomato_LHE/data_Tomato_LHE.npy')
# label_tomato = np.load('Tomato_LHE/label_Tomato_LHE.npy')

# merge_data_potato_norm = merge_data_potato / np.max(merge_data_potato)
# data_tomato_norm = data_tomato / np.max(data_tomato)

# data_potato_tensor_initial = torch.tensor(merge_data_potato_norm).float()
# data_potato_tensor = torch.transpose(data_potato_tensor_initial, 3, 1)

# label_potato_tensor = torch.tensor(merge_label_potato).long()


# data_tomato_tensor_initial = torch.tensor(data_tomato_norm).float()
# data_tomato_tensor = torch.transpose(data_tomato_tensor_initial, 3, 1)

# label_tomato_tensor = torch.tensor(label_tomato).long()


# train_data = data_tomato_tensor # data_potato_tensor
# train_label = label_tomato_tensor # label_potato_tensor

# test_data = data_potato_tensor # data_tomato_tensor
# test_label = label_potato_tensor #label_tomato_tensor

# train_data = TensorDataset(train_data,train_label)
# test_data = TensorDataset(test_data,test_label)

# train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
# test_loader = DataLoader(test_data, batch_size=batchsize)

data_source = np.load('Source/source_data.npy')
label_source = np.load('Source/source_label.npy')

data_target = np.load('Target/target_data.npy')
label_target = np.load('Target/target_label.npy')

data_source_norm = data_source / np.max(data_source)
data_target_norm = data_target / np.max(data_target)

data_source_tensor_initial = torch.tensor(data_source_norm).float()
data_source_tensor = torch.transpose(data_source_tensor_initial, 3, 1)

label_source_tensor = torch.tensor(label_source).long()

data_target_tensor_initial = torch.tensor(data_target_norm).float()
data_target_tensor = torch.transpose(data_target_tensor_initial, 3, 1)

label_target_tensor = torch.tensor(label_target).long()

train_data = data_source_tensor
train_label = label_source_tensor

test_data = data_target_tensor
test_label = label_target_tensor

train_data = TensorDataset(train_data,train_label)
test_data = TensorDataset(test_data,test_label)

train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batchsize)