import mido
import numpy as np

file = mido.MidiFile("output.mid") 

tps = 4		#4 Ticks pro Sekunde
state = np.zeros([128])
array = np.zeros([int(file.length*tps), 128])
index = 0



#msg Object structure: {'type': 'note_on', 'time': 0, 'channel': 0, 'note': 1, 'velocity': 100}
for msg in file.play():
	print(msg)
	#calc how many ticks have passed since the last message
	if msg.time == 0:
		array[index][msg.note] = 1
	else:
		for step in range(round(msg.time * tps)):
			array[index] = state
			index += 1
	state[msg.note] = not state[msg.note]
array[-1] = state
np.save("output_array.npy", array)
