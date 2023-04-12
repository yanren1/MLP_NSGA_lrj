import os.path

import torch
import torchvision
from dataloader.dataloader import SampleDataset
from torch.utils.data import DataLoader
import numpy as np
from loss.loss import MAPE
import pandas as pd

def pytorch_val(data_pth='data',model_pth='final.pt',output_name='result'):
    #seperate train and val set

    debug = True

    train_ratio = 0.95
    dataset = SampleDataset(root_dir = data_pth)
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # main model
    backbone = torchvision.ops.MLP(in_channels=27,
                                # hidden_channels=[28, 32, 64, 128, 256, 128, 64, 32, 16, 8, 3],
                                hidden_channels=[28, 64, 256, 64, 8, 3],
                                # norm_layer=nn.LayerNorm,
                                dropout= 0,inplace=False).cuda()
    # try read pre-train model
    weights_pth = model_pth
    try:
        backbone.load_state_dict(torch.load(weights_pth))

    except:
        print(f'No {weights_pth}')

    backbone = backbone.eval()
    val_loss_list1 = []
    with torch.no_grad():
        for val_batch in val_loader:
            val_sample, val_target = val_batch
            val_sample, val_target = val_sample.cuda(), val_target.cuda()
            output = backbone(val_sample)
            val_loss_list1.append(MAPE(output, val_target).item())

    result_dict = {'Pred_能耗': [], 'Pred_舒适时间': [], 'Pred_增量成本': [], 'Val_能耗': [], 'Val_舒适时间': [], 'Val_增量成本': []}
    # result_dict = {f'{tag[i]}_{target_names[j]}':[] for i in range(len(tag)) for j in range(len(target_names))}

    for bs in range(len(output)):
        result_dict['Pred_能耗'].append(output[bs][0].item())
        result_dict['Pred_舒适时间'].append(output[bs][1].item())
        result_dict['Pred_增量成本'].append(output[bs][2].item())

        result_dict['Val_能耗'].append(val_target[bs][0].item())
        result_dict['Val_舒适时间'].append(val_target[bs][1].item())
        result_dict['Val_增量成本'].append(val_target[bs][2].item())

    df=pd.DataFrame(result_dict)
    # if debug:
    #     # give some valing example, only for test
    #     print('############ sample val ##################')
    #     for i in range(100):
    #         print('Pred:', output[i])
    #         print('Target:', val_target[i])
    #         print()
    #     print('################################################')

    val_MAPE = np.mean(val_loss_list1)
    df.to_excel(output_name + f'_MAPE_{val_MAPE}.xlsx')

    print(f'val MAPE = {val_MAPE}    ')

if __name__ == '__main__':
    pytorch_val(data_pth='../data',model_pth='../final.pt',output_name='../result')