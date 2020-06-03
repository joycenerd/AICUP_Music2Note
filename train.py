from options import opt
import os
from pathlib import Path
import json
import numpy as np
from dataset import NoteDataset, get_loader
import torch
from model import Rnn, BiRNN, NeuralNet, NeuralNetWithRNN
import torch.nn as nn
import copy
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from visual import visualization


def preprocess(data_seq, label):
    new_label=[]

    for i in range(len(label)):
        label_of_one_song = []
        cur_note = 0
        cur_note_onset = label[i][cur_note][0]
        cur_note_offset = label[i][cur_note][1]
        cur_note_pitch = label[i][cur_note][2]

        for j in range(len(data_seq[i])):
            cur_time = j * 0.032 + 0.016

            if abs(cur_time - cur_note_onset) < 0.017:
                label_of_one_song.append(np.array([1, 0, cur_note_pitch]))
            elif cur_time < cur_note_onset or cur_note >= len(label[i]):
                label_of_one_song.append(np.array([0, 0, 0.0]))
            elif abs(cur_time - cur_note_offset) < 0.017:
                label_of_one_song.append(np.array([0, 1, cur_note_pitch]))
                cur_note = cur_note + 1
                if cur_note < len(label[i]):
                    cur_note_onset = label[i][cur_note][0]
                    cur_note_offset = label[i][cur_note][1]
                    cur_note_pitch = label[i][cur_note][2]
            else:
                label_of_one_song.append(np.array([0, 0, cur_note_pitch]))

        new_label.append(label_of_one_song)

    return new_label


def train():
    data_set = NoteDataset(data_seq, label)
    train_loader, valid_loader = get_loader(data_set)
    # model = Rnn(opt.input_dim, opt.hidden_size)
    # model = BiRNN(opt.input_dim, opt.hidden_size, opt.num_layers)
    # model=NeuralNet(opt.input_dim,[34,51,34,17])
    model=NeuralNetWithRNN(opt.input_dim,[34,51,34,17])
    model = model.cuda(opt.cuda_devices)

    best_model_params = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    training_loss_list = []
    valid_loss_list = []

    criterion_onset = nn.BCEWithLogitsLoss()
    criterion_pitch = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=1e-8)
    
    record = open('record.txt', 'w')

    for epoch in range(opt.epochs):
        print(f'Epoch: {epoch + 1}/{opt.epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{opt.epochs}'))

        training_loss = 0.0
        valid_loss = 0.0
        total_length=0.0
        
        model.train()

        for i, sample in enumerate(tqdm(train_loader)):
            inputs = sample['data']
            inputs = torch.FloatTensor(inputs)
            inputs = inputs.permute(1, 0, 2)
            inputs = inputs.cuda(opt.cuda_devices)

            target = sample['label']
            target = torch.FloatTensor(target)
            target = target.permute(1, 0, 2)
            target = target.cuda(opt.cuda_devices)

            inputs_length = list(inputs.shape)[0]

            optimizer.zero_grad()

            output1, output2 = model(inputs)
            onset_loss = criterion_onset(output1, torch.narrow(target, dim=2, start=0, length=2))
            pitch_loss = criterion_pitch(output2, torch.narrow(target, dim=2, start=2, length=1))
            total_loss = onset_loss + pitch_loss
            training_loss = training_loss + total_loss.item()
            total_length += 1

            total_loss.backward()
            optimizer.step()

        training_loss /= total_length
        training_loss_list.append(training_loss)
        print(f'training_loss: {training_loss:.4f}')
        
        model.eval()
        
        total_length = 0
        
        for i, sample in enumerate(tqdm(valid_loader)):
            inputs = sample['data']
            inputs = torch.FloatTensor(inputs)
            inputs = inputs.permute(1, 0, 2)
            inputs = inputs.cuda(opt.cuda_devices)

            target = sample['label']
            target = torch.FloatTensor(target)
            target = target.permute(1, 0, 2)
            target = target.cuda(opt.cuda_devices)

            inputs_length = list(inputs.shape)[0]

            optimizer.zero_grad()

            output1, output2 = model(inputs)
            onset_loss = criterion_onset(output1, torch.narrow(target, dim=2, start=0, length=2))
            pitch_loss = criterion_pitch(output2, torch.narrow(target, dim=2, start=2, length=1))
            total_loss = onset_loss + pitch_loss
            valid_loss = valid_loss + total_loss.item()
            total_length += 1
        
        
        valid_loss /= total_length
        valid_loss_list.append(valid_loss)
        print(f'valid_loss: {valid_loss:.4f}\n')
        
        scheduler.step(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_training_loss = training_loss
            best_model_params = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 50 == 0:
            model.load_state_dict(best_model_params)
            weight_path = Path(opt.checkpoint_dir).joinpath(
                f'model-{epoch + 1}epoch-{best_loss:.02f}-best_valid_loss.pth')
            torch.save(model, str(weight_path))
            record.write(f'{epoch + 1}\n')
            record.write(f'Best training loss: {best_training_loss:.4f}\n')
            record.write(f'Best valid loss: {best_loss:.4f}\n')

    print(f'Best training loss: {best_training_loss:.4f}')
    print(f'Best valid loss: {best_loss:.4f}')

    model.load_state_dict(best_model_params)
    weight_path = Path(opt.checkpoint_dir).joinpath(f'model-{best_loss:.02f}-best_valid_loss.pth')
    torch.save(model, str(weight_path))
    visualization(training_loss_list, valid_loss_list)

    return model


if __name__ == '__main__':
    THE_FOLDER = opt.data_root

    data_seq = []
    label = []

    for the_dir in os.listdir(THE_FOLDER):
        json_path = Path(THE_FOLDER).joinpath(the_dir).joinpath(the_dir+'_feature.json')
        gt_path = Path(THE_FOLDER).joinpath(the_dir).joinpath(the_dir+'_groundtruth.txt')
        youtube_link_path = Path(THE_FOLDER).joinpath(the_dir).joinpath(the_dir+'_link.txt')

        with open(json_path,'r') as json_file:
            temp = json.loads(json_file.read())

        data = []
        for key, value in temp.items():
            data.append(value)
            #print(key)

        data = np.array(data).T
        data_seq.append(data)

        gtdata = np.loadtxt(gt_path)
        label.append(gtdata)

    label=preprocess(data_seq,label)

    model = train()
