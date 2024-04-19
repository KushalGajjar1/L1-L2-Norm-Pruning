import torch
import torch.nn as nn
from details import models_list, model_input, filters_to_prune, ffl_shape
from models import getModels
from train import trainModel
from utils import get_bottom_indices, getTestAcc
from prune_filters import prune_filters

from tqdm import tqdm
from fvcore.nn import parameter_count
from fvcore.nn import FlopCountAnalysis

import pandas as pd

if __name__ == '__main__':

    for modelName in models_list:

        model = getModels(modelName)
        print(model)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_acc_before_pruning = trainModel(model, optimizer, criterion)

        torch.save(model.state_dict(), f'{modelName}.pt')

        total_params_original = parameter_count(model)[""]

        input = (torch.randn(model_input[modelName]), )
        flops_original = FlopCountAnalysis(model.to('cpu'), input)
        total_original_flops = flops_original.total()

        percentageParameterPruned = []
        percentageFLOPSPruned = []
        parameters = []
        FLOPS = []
        testAccRetraining = []

        for prune_limits in tqdm(filters_to_prune[modelName]):

            model = getModels(modelName)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            model.load_state_dict(torch.load(f'{modelName}.pt'))

            l1Norm = []

            for layer_name, layer_module in model.named_modules():
                if(isinstance(layer_module, nn.Conv2d)):
                    temp = []
                    filter_weight = layer_module.weight.clone()
                    for k in range(filter_weight.size()[0]):
                        temp.append(float("{:.6f}".format((filter_weight[k, :, :, :]).norm(1).item())))
                    l1Norm.append(temp)

            layer_bound = l1Norm
            dec_indices = []

            for i in range(len(layer_bound)):
                temp = []
                temp = get_bottom_indices(layer_bound[i], prune_limits[i])
                dec_indices.append(temp)

            prune_filters(model, dec_indices, ffl_shape[modelName])

            total_params = parameter_count(model)[""]
            parameters.append(total_params)
            percentageParameterPruned.append(100-((total_params/total_params_original)*100))

            input = (torch.randn(model_input[modelName]), )
            flops = FlopCountAnalysis(model.to('cpu'), input)
            total_flops = flops.total()
            FLOPS.append(total_flops)
            percentageFLOPSPruned.append(100-((total_flops/total_original_flops)*100))

            train_acc_retraining = trainModel(model, optimizer, criterion)
            testAccRetraining.append(getTestAcc(model))

        data = {'Parameters' : parameters, 'Percentage Parameters Pruned' : percentageParameterPruned, 'FLOPS' : FLOPS, 'Percentage FLOPS Pruned' : percentageFLOPSPruned, 'Test Accuracy L1' : testAccRetraining}

        df = pd.DataFrame(data)
        excel_file_path = f'{modelName}_L1.xlsx'
        df.to_excel(excel_file_path, index=False)
        