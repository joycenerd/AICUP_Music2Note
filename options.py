import argparse
from pathlib import Path


ROOTPATH = "/home/dl406410013/music/AICUP_Music2Note"

parser=argparse.ArgumentParser()
parser.add_argument('--data_root',type=str,default=Path(ROOTPATH).joinpath('Data/MIR'),help='Your dataset root directory')
parser.add_argument('--cuda_devices', type=int, default=1, help='gpu device')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
parser.add_argument('--input_dim', type=int, default=23, help='input dimension')
parser.add_argument('--hidden_size', type=int, default=128, help='size of hidden layer')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
parser.add_argument('--lr', type=float, default=0.003, help='learning rate')
parser.add_argument('--checkpoint_dir', type=str, default=Path(ROOTPATH).joinpath('checkpoint'), help='Directory that save the checkpoints')
parser.add_argument('--weight', type=str, help='weight for testing')
parser.add_argument('--test_path', type=str, default=Path(ROOTPATH).joinpath('AIcup_testset_ok'), help='path of test set')
opt = parser.parse_args()
