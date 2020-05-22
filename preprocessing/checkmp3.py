from pathlib import Path


ROOTPATH="/home/joycenerd/AICUP_Music2Note/Data/MIR"

for folders in Path(ROOTPATH).glob('*'):
    for files in folders.glob('*'):
        have_mp3=False
        if str(files)[-4:]=='.mp3':
            have_mp3=True
    if have_mp3==False:
        print(folders)