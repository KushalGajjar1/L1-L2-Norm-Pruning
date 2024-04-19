import torch
import torch.nn as nn

def prune_filters(model, indices, dimension):

    conv_layer = 0

    for layer_name, layer_module in model.named_modules():

        if(isinstance(layer_module, nn.Conv2d)):

            if(conv_layer == 0):
                in_channels = [i for i in range(layer_module.weight.shape[1])]

            else:
                in_channels = indices[conv_layer - 1]

            out_channels = indices[conv_layer]
            layer_module.weight = nn.Parameter(torch.FloatTensor(torch.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])))

            if(layer_module.bias is not None):
                layer_module.bias = nn.Parameter(torch.FloatTensor(torch.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])))

            layer_module.weight = nn.Parameter(torch.FloatTensor(torch.from_numpy(layer_module.weight.data.cpu().numpy()[:, in_channels])))

            layer_module.in_channels = len(in_channels)
            layer_module.out_channels = len(out_channels)

            conv_layer += 1

        if(isinstance(layer_module, nn.BatchNorm2d)):

            out_channels = indices[conv_layer]

            layer_module.weight = nn.Parameter(torch.FloatTensor(torch.from_numpy(layer_module.weight.data.cpu().numpy()[out_channels])))
            layer_module.bias = nn.Parameter(torch.FloatTensor(torch.from_numpy(layer_module.bias.data.cpu().numpy()[out_channels])))

            layer_module.running_mean = torch.from_numpy(layer_module.running_mean.cpu().numpy()[out_channels])
            layer_module.running_var = torch.from_numpy(layer_module.running_var.cpu().numpy()[out_channels])

            layer_module.num_features = len(out_channels)

        if(isinstance(layer_module, nn.Linear)):

            conv_layer -= 1
            in_channels = indices[conv_layer]

            weight_linear = layer_module.weight.data.cpu().numpy()

            size = dimension * dimension
            expanded_in_channels = []
            for i in in_channels:
                for j in range(size):
                    expanded_in_channels.extend([i*size + j])

            layer_module.weight = nn.Parameter(torch.from_numpy(weight_linear[:, expanded_in_channels]))
            layer_module.in_features = len(expanded_in_channels)

            break
