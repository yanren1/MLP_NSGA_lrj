import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class SampleDataset(Dataset):
    def __init__(self, root_dir, ):
        super(SampleDataset, self).__init__()

        self.root_dir = root_dir
        self.samples = self.__read_xlsx()

    def __getitem__(self, index):
        samples = self.samples[index]
        # sample, target = samples[:-3],samples[-3:]

        return samples[:-3], samples[-3:]

    def __len__(self):
        return len(self.samples)

    def __read_xlsx(self):
        f_pth = os.path.join(self.root_dir, 'data.xlsx')
        # f_pth = os.path.join(root_dir, 'data.xlsx')
        df = pd.read_excel(f_pth,)


        # window_type = ['5+12A+5+12A+5LOWE', '5+12A+5+12A+5LOWE*2', '5+12Ar+5+12Ar+5LOWE', '5+12Ar+5+12Ar+5LOWE*2',
        #  '5+12A+5LOWE+12A+5LOWE', '5+12A+5LOWE*2+12A+5LOWE*2', '5+12Ar+5LOWE+12Ar+5LOWE', '5LOWE+12A+5LOWE+12A+5LOWE',
        #  '5+12Ar+5LOWE*2+12Ar+5LOWE*2', '5LOWE*2+12A+5LOWE*2+12A+5LOWE*2', '5LOWE+12Ar+5LOWE+12Ar+5LOWE',
        #  '5LOWE*2+12Ar+5LOWE*2+12Ar+5LOWE*2']
        # window_type = [i.strip() for i in pd.read_excel(os.path.join('data', '1323.xlsx'))['window_type']]
        p_type = ['内廊式', '中庭式']

        # df['外窗类型'] = [window_type.index(i.strip()) for i in df['外窗类型']]
        df['平面形式'] = [p_type.index(i.strip()) for i in df['平面形式']]


        samples = torch.from_numpy(df.to_numpy()).float()

        return samples


if __name__ == '__main__':
    dataset = SampleDataset(root_dir='data')

    # train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    # for step,batch in enumerate(train_loader):


