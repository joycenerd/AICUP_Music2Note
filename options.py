import argparse
from pathlib import Path


ROOTPATH = "/mnt/md0/new-home/joycenerd/AICUP_Music2Note"

parser=argparse.ArgumentParser()
parser.add_argument('--data_root',type=str,default=Path(ROOTPATH).joinpath('Data/MIR'),help='Your dataset root directory')
parser.add_argument('--cuda_devices', type=int, default=1, help='gpu device')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
parser.add_argument('--input_dim', type=int, default=23, help='input dimension')
parser.add_argument('--hidden_size', type=int, default=50, help='size of hidden layer')
parser.add_argument('--checkpoint_dir', type=str, default=Path(ROOTPATH).joinpath('checkpoint'), help='Directory that save the checkpoints')
opt = parser.parse_args()
