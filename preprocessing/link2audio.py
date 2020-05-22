from pytube import YouTube
from pathlib import Path
import shutil


ROOTPATH="/home/joycenerd/AICUP_Music2Note/Data"

f=open('./lost.txt', 'r')

for _dir in f:
    folder_num = _dir[42:-1]
    print(folder_num)
    link = open(Path(_dir[:-1]).joinpath(folder_num+'_link.txt'),'r')
    youtube_link = link.read()
    y = YouTube(youtube_link)
    t = y.streams.filter(only_audio=True).all()
    t[0].download(output_path=Path(_dir[:-1]))
    for mp4_file in Path(_dir[:-1]).glob('*.mp4'):
        print(str(mp4_file))
    mp3_file=str(Path(_dir[:-1]).joinpath(folder_num+'.mp3'))
    shutil.move(mp4_file,mp3_file)

