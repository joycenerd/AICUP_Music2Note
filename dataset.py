from torch.utils.data import Dataset,DataLoader
import torch
from torch.nn.utils import rnn
import numpy as np
from torch.utils.data import random_split
from options import opt


class NoteDataset(Dataset):
    def __init__(self, data_seq, label):
        self.x = data_seq
        self.y = label

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {
            'data': self.x[index],
            'label': self.y[index]
        }


def get_loader(data_set):
    total_sz=data_set.__len__()
    valid_sz=int(0.15*total_sz)
    train_set,valid_set=random_split(data_set,[total_sz-valid_sz,valid_sz])
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, num_workers=opt.num_workers, collate_fn=collate_func, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=opt.batch_size, num_workers=opt.num_workers, collate_fn=collate_func, shuffle=True)
    return train_loader, valid_loader


def collate_func(samples):
    batch = {}
    tmp = [torch.from_numpy(np.array(sample['data'], dtype=np.float32)) for sample in samples]
    padded_data = rnn.pad_sequence(tmp, batch_first=True, padding_value=0)
    batch['data'] = padded_data
    batch['label'] = [np.array(sample['label'], dtype=np.float32) for sample in samples]

    return batch
