from options import opt
from pathlib import Path
import torch
import os
import json
import numpy as np


def post_processing(output1, pitch):
    pitch = pitch.view(-1).cpu().detach().numpy()
    pitch = np.rint(pitch)
    pitch =  pitch.astype(int)
    notes=[]
    prev = -1
    onset = None
    offset =  None
    flag = 1
    for i in range(len(pitch)):
        this_pitch = pitch[i]
        this_onset = i*0.032
        this_offset = this_onset+0.032 - 0.001
        if this_pitch > 1 and prev==-1:
            onset = this_onset
            offset = this_offset
            prev =  this_pitch
        elif this_pitch > 1 and prev != -1 and this_pitch==prev:
            offset =  this_offset
        elif this_pitch <= 1 and prev != -1:
            notes.append([onset, offset, prev])
            onset = None
            ofset = None
            prev = -1
    if prev != -1:
        notes.append([onset, offset, prev])
            
    return notes
                
            
def testing(model, data_seq):
    model.eval()
    
    inputs = data_seq
    inputs = torch.FloatTensor(inputs)
    inputs = inputs.permute(1, 0, 2)
    inputs = inputs.cuda(opt.cuda_devices)

    data_length= list(inputs.shape)[0]

    output1, output2 = model(inputs)
    # print(output1)
    answer = post_processing(output1, output2)
    return answer


if __name__ == '__main__':
    THE_FOLDER = opt.test_path
    
    weight_path = Path(opt.checkpoint_dir).joinpath(opt.weight)
    model = torch.load(str(weight_path))
    model =  model.cuda(opt.cuda_devices)

    data_seq = []

    for the_dir in os.listdir(THE_FOLDER):
        # the music id i
        the_key = the_dir
        json_path = Path(THE_FOLDER).joinpath(the_dir).joinpath(the_dir+'_feature.json')
        youtube_link_path = Path(THE_FOLDER).joinpath(the_dir).joinpath(the_dir+'_link.txt')
        answer_list=[]
        print(json_path)

        with open(json_path,'r') as json_file:
            temp = json.loads(json_file.read())

        data = []
        for key, value in temp.items():
            data.append(value)
            #print(key)

        data = np.array(data).T
        data_seq.append(data)
        
        # answer is all the [onset, offset, pitch]
        answer = testing(model, data_seq)
        