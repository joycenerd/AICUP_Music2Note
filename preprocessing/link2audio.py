from pytube import YouTube
from pathlib import Path


ROOTPATH="/home/joycenerd/AICUP_Music2Note"

f=open('./lost.txt', 'r')

for _dir in f:
    folder_num = _dir[43:-1]
    print(folder_num)
    link = open(Path(_dir).joinpath(folder_num).joinpath('_link.txt'),'r')
    youtube_link = link.read()
    y = YouTube(youtube_link)
    t = y.streams.filter(only_audio=True).all()
    t[0].download(output_path=folder_num)
    break

