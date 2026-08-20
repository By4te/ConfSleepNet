import torch
from torch.utils.data import Dataset, DataLoader

import glob
import os
import numpy as np


# two EEG channels: EEG Fpz-Cz; EEG Pz-Oz
class EdfDataset_2EEG(Dataset):
    def __init__(self, files, seq_len, shuffle_seed=42):
        super(EdfDataset_2EEG, self).__init__()


        X_data = []
        y = []

        for fi in files:
            data = np.load(fi)
            for seq_idx in range(len(data['EEG Fpz-Cz'])//seq_len):
                item_x1 = data['EEG Fpz-Cz'][seq_idx*seq_len:(seq_idx+1)*seq_len, ...]  # item_x1.shape: (20, 3000)
                item_x2 = data['EEG Pz-Oz'][seq_idx*seq_len:(seq_idx+1)*seq_len, ...]   # item_x2.shape: (20, 3000)
                item_y = data['y'][seq_idx*seq_len:(seq_idx+1)*seq_len, ...]            # item_y.shape:  (20, )

                item_x1 = torch.from_numpy(item_x1).unsqueeze(0)     # (1, 20, 3000)
                item_x2 = torch.from_numpy(item_x2).unsqueeze(0)
                item_x = torch.cat((item_x1, item_x2), 0)            # (2, 20, 3000)

                item_y = torch.from_numpy(item_y)
                assert item_x1.shape[1] == item_y.shape[0]
                X_data.append(item_x)
                y.append(item_y)
            # print('--len(X_data):', len(X_data))

        assert len(X_data) == len(y)

        self.X_data = X_data
        self.y = y

    def __getitem__(self, idx):
        return self.X_data[idx], self.y[idx]

    def __len__(self):
        return len(self.y)




# single EEG channel
class EdfDataset_1EEG(Dataset):
    def __init__(self, files, seq_len, shuffle_seed=42):
        super(EdfDataset_1EEG, self).__init__()
        X_data = []
        y = []
        print('len(files):', len(files))

        for fi in files:
            data = np.load(fi)
            for seq_idx in range(len(data['EEG Fpz-Cz'])//seq_len):
                item_x = data['EEG Fpz-Cz'][seq_idx*seq_len:(seq_idx+1)*seq_len, ...]  #item_x.shape: (20, 3000)
                item_y = data['y'][seq_idx*seq_len:(seq_idx+1)*seq_len, ...]           #item_y.shape: (20, )

                item_x = torch.tensor(item_x)
                item_y = torch.tensor(item_y)
                assert item_x.shape[0] == item_y.shape[0]
                X_data.append(item_x)
                y.append(item_y)
                del item_x, item_y

        assert len(X_data) == len(y)

        self.X_data = X_data
        self.y = y

    def __getitem__(self, idx):
        return self.X_data[idx], self.y[idx]

    def __len__(self):
        return len(self.y)





# single EOG channel
class EdfDataset_EOG_5Stages(Dataset):
    def __init__(self, files, seq_len, shuffle_seed=42):
        super(EdfDataset_EOG_5Stages, self).__init__()
        X_data = []
        y = []

        for fi in files:
            data = np.load(fi)
            for seq_idx in range(len(data['EOG'])//seq_len):
                item_x = data['EOG'][seq_idx*seq_len:(seq_idx+1)*seq_len, ...]  #item_x.shape: (20, 3000)
                item_y = data['y'][seq_idx*seq_len:(seq_idx+1)*seq_len, ...]           #item_y.shape: (20, )

                item_x = torch.tensor(item_x)
                item_y = torch.tensor(item_y)
                assert item_x.shape[0] == item_y.shape[0]
                X_data.append(item_x)
                y.append(item_y)
            #print('--len(X_data):', len(X_data))

        assert len(X_data) == len(y)

        #print('X_data.shape, y.shape:', np.array(X_data).shape, np.array(y).shape)

        self.X_data = X_data
        self.y = y

    def __getitem__(self, idx):
        return self.X_data[idx], self.y[idx]

    def __len__(self):
        return len(self.y)

# single EOG channel
class EdfDataset_EOG_3Stages(Dataset):
    def __init__(self, files, seq_len, shuffle_seed=42):
        super(EdfDataset_EOG_3Stages, self).__init__()
        X_data = []
        y = []

        for fi in files:
            data = np.load(fi)
            for seq_idx in range(len(data['EOG'])//seq_len):
                item_x = data['EOG'][seq_idx*seq_len:(seq_idx+1)*seq_len, ...]  #item_x.shape: (20, 3000)
                item_y = data['y'][seq_idx*seq_len:(seq_idx+1)*seq_len, ...]           #item_y.shape: (20, )

                #print('before   --item_y:', item_y)
                item_y = np.where(item_y == 2, 1, item_y)
                item_y = np.where(item_y == 3, 1, item_y)
                item_y = np.where(item_y == 4, 2, item_y)
                #print('after   --item_y:', item_y)

                item_x = torch.tensor(item_x)
                item_y = torch.tensor(item_y)
                assert item_x.shape[0] == item_y.shape[0]
                X_data.append(item_x)
                y.append(item_y)
            #print('--len(X_data):', len(X_data))

        assert len(X_data) == len(y)

        self.X_data = X_data
        self.y = y

    def __getitem__(self, idx):
        return self.X_data[idx], self.y[idx]

    def __len__(self):
        return len(self.y)


# two EEG channels: EEG Fpz-Cz; EEG Pz-Oz
class EdfDataset_2EEG_1EOG(Dataset):
    def __init__(self, files, seq_len, shuffle_seed=42):
        super(EdfDataset_2EEG_1EOG, self).__init__()
        X_data = []
        y = []

        for fi in files:
            data = np.load(fi)
            for seq_idx in range(len(data['EEG Fpz-Cz'])//seq_len):
                item_x1 = data['EEG Fpz-Cz'][seq_idx*seq_len:(seq_idx+1)*seq_len, ...]  # item_x1.shape: (20, 3000)
                item_x2 = data['EEG Pz-Oz'][seq_idx*seq_len:(seq_idx+1)*seq_len, ...]   # item_x2.shape: (20, 3000)
                item_x3 = data['EOG'][seq_idx * seq_len:(seq_idx + 1) * seq_len, ...]  # item_x.shape: (20, 3000)
                item_y = data['y'][seq_idx*seq_len:(seq_idx+1)*seq_len, ...]            # item_y.shape:  (20, )

                item_x1 = torch.tensor(item_x1).unsqueeze(0)     # (1, 20, 3000)  EEG Fpz-Cz
                item_x2 = torch.tensor(item_x2).unsqueeze(0)     # (1, 20, 3000)  EEG Pz-Oz
                item_x3 = torch.tensor(item_x3).unsqueeze(0)     # (1, 20, 3000)  EOG
                item_x_12 = torch.cat((item_x1, item_x2), 0)         # (2, 20, 3000)
                item_x = torch.cat((item_x_12, item_x3), 0)          # (3, 20, 3000)

                item_y_5stages = torch.tensor(item_y).unsqueeze(0)    # (1, 20)
                item_y = np.where(item_y == 2, 1, item_y)
                item_y = np.where(item_y == 3, 1, item_y)
                item_y = np.where(item_y == 4, 2, item_y)
                item_y_3stages = torch.tensor(item_y).unsqueeze(0)    # (1, 20)
                item_y = torch.cat((item_y_5stages, item_y_3stages), 0)   # (2, 20)

                assert item_x.shape[1] == item_y.shape[-1]
                X_data.append(item_x)
                y.append(item_y)
        assert len(X_data) == len(y)

        self.X_data = X_data
        self.y = y

    def __getitem__(self, idx):
        return self.X_data[idx], self.y[idx]

    def __len__(self):
        return len(self.y)



if __name__ == "__main__":
    data_path = "G:/Data/npz/"

    edf_dataset = EdfDataset(data_path, seq_len=20, is_train=True)
    edf_dataloader = DataLoader(edf_dataset, batch_size=15, shuffle=True)

    for X, y in edf_dataloader:
        print(X.shape, y.shape)

    # data_list = glob.glob(os.path.join(data_path,"*.npz"))

    # data_list.sort()

    # for fi in data_list:
    #     data = np.load(fi)
    #     print(data['x'].shape,data['y'].shape)

    # data = np.load(data_list[0])

    # print(data['x'].shape, data['y'].shape)