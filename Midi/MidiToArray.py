"""
Zuerst werden alle Eventzeiten auf die samplerate gerundet.(Tempochanges werden ignoriert)
"""
import mido
import numpy as np

#10 Samples/Second
samplerate = 100

file = mido.MidiFile("Files/example.mid") 
print("Tempo: {}, ticks per beat: {}".format(file.__dict__, file.ticks_per_beat))

state = np.zeros([128])
array = np.zeros([int(file.length*samplerate), 128])
index = 0

def roundEventTime(time):
	return round(time*samplerate)/samplerate

count = 0

#msg Object structure: {'type': 'note_on', 'time': 0, 'channel': 0, 'note': 1, 'velocity': 100}
#'time' attribute is given in seconds
for msg in file:
	#ignore all non-note changes
	if msg.type == "note_on" or msg.type == "note_off":
		#Round the msg time to the nearest sample
		msg.time = roundEventTime(msg.time)

		#if the event is at a new timestep, flush the old state n times to the array
		if msg.time != 0:
			for _ in range(int(msg.time*samplerate)):
				array[count] = state
				count += 1


		#Note events with velocity 0 are considered to be note off events
		if msg.type == "note_off" or msg.velocity == 0:
			state[msg.note] = 0
		else:
			state[msg.note] = 1
file.close()
np.save("Files/output_array", array)