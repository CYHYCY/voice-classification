import time
import winsound
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import pyaudio

pyaudio.PyAudio()
num_cpu = multiprocessing.cpu_count()  # cpu逻辑核心数
print(num_cpu)
threadPool = ThreadPoolExecutor(max_workers=2 * num_cpu)


def record_audio(record_second):
    """
    读取record_secondS语音信息
    """
    print(1)
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    data = []
    print("begin")
    for i in range(0, int(RATE / CHUNK * record_second)):
        data.append(stream.read(CHUNK))
    print(len(data), "end")

    stream.stop_stream()
    stream.close()
    p.terminate()
    return data


def deal(data):
    '''

    :return: True or False
    '''
    return True


def start_process(second):
    data = record_audio(second)
    flag = deal(data)
    if flag:
        winsound.Beep(1000, 1000)


if __name__ == "__main__":
    while True:
        threadPool.submit(start_process, 3)
        time.sleep(1)
