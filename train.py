import os
import torch
import torch.nn as nn
import torchvision
from dataloader.dataloader import SampleDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import numpy as np
from loss.loss import weighted_mse,MAPE
import time

# save model according to time
def save_model(model_save_pth,model, epoch,train_wmse,train_mape,val_wmse,val_mape):
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    filename = 'model_{}_epoch_{}_train_wmse_{:0.2e}_train_mape_{:0.2e}_val_wmse_{:0.2e}_val_mape_{:0.2e}.pt'.format(current_time,
                                                                                                                     epoch,
                                                                                                                     train_wmse,
                                                                                                                     train_mape,
                                                                                                                     val_wmse,
                                                                                                                     val_mape)

    filename = os.path.join(model_save_pth,filename)
    torch.save(model.state_dict(), filename)

def train():
    #seperate train and val set

    debug = True
    use_pretrain = False

    train_ratio = 0.8
    dataset = SampleDataset(root_dir = 'data')
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)


    # main model
    backbone = torchvision.ops.MLP(in_channels=27,
                                # hidden_channels=[28, 32, 64, 128, 256, 128, 64, 32, 16, 8, 3],
                                hidden_channels=[28, 64, 256, 64, 8, 3],

                                # norm_layer=nn.LayerNorm,
                                dropout= 0,inplace=False).cuda()
    # try read pre-train model
    if use_pretrain:
        weights_pth = 'final.pt'
        try:
            backbone.load_state_dict(torch.load(weights_pth))
        except:
            print(f'No {weights_pth}')

    # set lr,#epoch, optimizer and scheduler
    lr=1e-3
    optimizer = optim.Adam(
        backbone.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


    num_epoch = 200000
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=5e-6)
    mse_weight = [1,1/0.05,1/5]
    # mse_weight = [1,1,1]

    # set tensorboard dir
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    tf_pth = os.path.join('runs',current_time)
    model_save_pth = os.path.join('model_saved',current_time)
    os.mkdir(tf_pth)
    os.mkdir(model_save_pth)
    writer = SummaryWriter(tf_pth)

    #start training
    backbone.train()
    for epoch in range(num_epoch):
        loss_list = []
        train_mape_list = []
        for step,batch in enumerate(train_loader):

            backbone.zero_grad()
            sample, target = batch
            sample, target = sample.cuda(), target.cuda()

            output = backbone(sample)

            loss = weighted_mse(output, target,weights=torch.tensor(mse_weight).cuda())
            loss.backward()
            optimizer.step()
            # scheduler.step()

            loss_list.append(loss.item())
            with torch.no_grad():
                train_mape_list.append(MAPE(output, target).item())

        scheduler.step()

        # send message to tensorboard
        if epoch % 100 == 0:
        #     print(f'\r Epoch:{epoch} MSE loss = {np.mean(loss_list)} ,lr = {optimizer.param_groups[0]["lr"]}     ', end = ' ')
            writer.add_scalar('Training WMSE Loss', np.mean(loss_list), epoch)
            writer.add_scalar('Training MAPE', np.mean(train_mape_list), epoch)
            writer.add_scalar('Learning rate', optimizer.param_groups[0]["lr"], epoch)

        # valing and save
        if epoch % 1000==0:
            print('Valing.....')
            val_loss_list1 = []
            val_loss_list2 = []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_sample, val_target = val_batch
                    val_sample, val_target = val_sample.cuda(), val_target.cuda()

                    output = backbone(val_sample)
                    val_MAPE = MAPE(output, val_target)
                    val_wmse =weighted_mse(output, val_target,weights=torch.tensor(mse_weight).cuda())

                    val_loss_list1.append(val_wmse.item())
                    val_loss_list2.append(val_MAPE.item())
                    writer.add_scalar('Validation WMSE', np.mean(val_loss_list1), epoch + 1)
                    writer.add_scalar('Validation MAPE', np.mean(val_loss_list2), epoch + 1)

            if debug:
                # give some valing example, only for test
                print('############ sample val ##################')
                for i in range(5):
                    print('Pred:',output[i])
                    print('Target:',val_target[i])
                    print()
                print('################################################')

            Train_wmse = np.mean(loss_list)
            Train_MAPE = np.mean(train_mape_list)
            val_wmse = np.mean(val_loss_list1)
            val_MAPE = np.mean(val_loss_list2)
            print(f'VAL Epoch:{epoch} Train wmse = {Train_wmse}, '
                  f'Train MAPE = {Train_MAPE},'
                  f'val wmse = {val_wmse}, '
                  f'val MAPE = {val_MAPE}    ')
            print()
            save_model(model_save_pth,backbone, epoch, Train_wmse, Train_MAPE, val_wmse, val_MAPE)

    torch.save(backbone.state_dict(), os.path.join(model_save_pth,'final.pt'))
    writer.flush()
    writer.close()


if __name__ == '__main__':
    train()
