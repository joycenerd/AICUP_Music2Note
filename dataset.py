from torch.utils.data import Dataset,DataLoader
import torch
from torch.nn.utils import rnn
import numpy as np


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


def gen_loader(dataset, batch_size, num_workers, collate_fn, shuffle=True):
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=collate_func)
    return data_loader


def collate_func(samples):
    batch = {}
    tmp = [torch.from_numpy(np.array(sample['data'], dtype=np.float32)) for sample in samples]
    padded_data = rnn.pad_sequence(tmp, batch_first=True, padding_value=0)
    batch['data'] = padded_data
    batch['label'] = [np.array(sample['label'], dtype=np.float32) for sample in samples]

    return batch
