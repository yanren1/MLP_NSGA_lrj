import torch
import torchvision
from dataloader.dataloader import SampleDataset
from torch.utils.data import DataLoader
import numpy as np
from loss.loss import MAPE

def val():
    #seperate train and val set

    debug = True

    train_ratio = 0.95
    dataset = SampleDataset(root_dir = 'data')
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=True,
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
    weights_pth = 'final.pt'
    try:
        backbone.load_state_dict(torch.load(weights_pth))

    except:
        print(f'No {weights_pth}')

    val_loss_list1 = []
    with torch.no_grad():
        for val_batch in val_loader:
            val_sample, val_target = val_batch
            val_sample, val_target = val_sample.cuda(), val_target.cuda()
            output = backbone(val_sample)
            val_loss_list1.append(MAPE(output, val_target).item())


    if debug:
        # give some valing example, only for test
        print('############ sample val ##################')
        for i in range(5):
            print('Pred:', output[i])
            print('Target:', val_target[i])
            print()
        print('################################################')

    val_MAPE = np.mean(val_loss_list1)
    print(f'val MAPE = {val_MAPE}    ')
    print()

if __name__ == '__main__':
    val()