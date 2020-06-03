import matplotlib.pyplot as plt
from options import opt
from pathlib import Path
import numpy as np
import matplotlib as mpl


def visualization(training_loss, valid_loss):
    mpl.style.use('default')
    plt.figure(figsize=[8,6])
    plt.plot(training_loss,linewidth=2.0)
    plt.plot(valid_loss,linewidth=2.0)
    plt.legend(['Training Loss', 'Valid Loss'],fontsize=18)
    plt.xlabel('Epochs',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.savefig('./figure/loss_curve.jpg')