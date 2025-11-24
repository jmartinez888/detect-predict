import pygame
import os
import time

try:
    pygame.mixer.init()
    sound_file = 'audios/danger_alarm_80s_2seconds.mp3'
    if not os.path.exists(sound_file):
        print(f"Error: File {sound_file} not found.")
    else:
        print(f"Playing {sound_file}...")
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)
        print("Done.")
except Exception as e:
    print(f"Error: {e}")
