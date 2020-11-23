from moviepy.editor import *
from mutagen.mp3 import MP3


video = VideoFileClip('6.mp4')
audio = video.audio
audio.write_audiofile('test.mp3')

audio = MP3("test.mp3")
print(audio.info.length)
