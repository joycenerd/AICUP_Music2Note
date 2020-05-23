from options import opt
import os


if __name__ == '__main__':
    THE_FOLDER = opt.data_root

    data_seq = []
    label = []

    for the_dir in os.listdir(THE_FOLDER):
