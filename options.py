import argparse
from pathlib import Path


ROOTPATH="/home/joycenerd/AICUP_Music2Note"

parser=argparse.ArgumentParser()
parser.add_argument('--data_root',type=str,default=Path(ROOTPATH).joinpath('MIR-ST500/MIR'),help='Your dataset root directory')
opt=parser.parse_args()