import threading
from playsound import playsound
import vlc

def play_sound():
    playsound('audios/danger_alarm_80s_2seconds.mp3')
    

# Crear y empezar un hilo
thread = threading.Thread(target=play_sound)
thread.start()

# El programa principal sigue ejecutándose mientras el hilo está reproduciendo el sonido
print('playing sound using playsound')