import mido
import numpy as np

def getNoteLength(start_idx, note):
		end_idx = start_idx
		while array[end_idx][note] and end_idx != array.shape[0]:
			if end_idx < array.shape[0]-1:
				end_idx += 1
			else:
				break
		return (end_idx-start_idx)/tps


def roundEventTime(time, samplerate):
	return round(time*samplerate)/samplerate

def toArray(inp_file, samplerate=100):

	file = mido.MidiFile(inp_file) 

	state = np.zeros([80])
	array = np.zeros([int(file.length*samplerate), 80])
	index = 0
	count = 0

	#msg Object structure: {'type': 'note_on', 'time': 0, 'channel': 0, 'note': 1, 'velocity': 100}
	#'time' attribute is given in seconds
	for msg in file:
		#ignore all non-note changes
		if msg.type == "note_on" or msg.type == "note_off":
			if msg.note < 100 and msg.note > 20:
				#Round the msg time to the nearest sample
				msg.time = roundEventTime(msg.time, samplerate)

				#if the event is at a new timestep, flush the old state n times to the array
				if msg.time != 0:
					for _ in range(int(msg.time*samplerate)-1):
						array[count] = state
						count += 1


				#Note events with velocity 0 are considered to be note off events
				if msg.type == "note_off" or msg.velocity == 0:
					state[msg.note-20] = 0
				else:
					state[msg.note-20] = 1

	return array[:count]

#toArray("example.mid")


def toMidi(array, samplerate = 100):
	mid = mido.MidiFile(type = 0)
	track = mido.MidiTrack()
	mid.tracks.append(track)

	mid.ticks_per_beat = 96
	beats_per_minute = 120
	tempo = mido.bpm2tempo(beats_per_minute)

	track.append(mido.Message('program_change', program=12, time=0))

	playing = np.zeros(80)

	#prevtimeindex merkt sich den zeitpunkt der letzten note, wichtig für dtime
	prev_time_index = 0

	for time_index, state in enumerate(array):
		for note_index, note in enumerate(state):
			if note and not playing[note_index]:
				track.append(mido.Message('note_on', note=note_index+20, velocity=64, time=int(mido.second2tick((time_index-prev_time_index)/samplerate, mid.ticks_per_beat, tempo))))
				playing[note_index] = True
				prev_time_index = time_index

			elif not note and playing[note_index]:
				track.append(mido.Message('note_off', note=note_index+20, velocity=64, time=int(mido.second2tick((time_index-prev_time_index)/samplerate, mid.ticks_per_beat, tempo))))
				playing[note_index] = False
				prev_time_index = time_index

	return mid

def playMidi(file):
	#define a port to stream the messages to
	port = mido.open_output()

	for msg in file.play():
		print(msg)
		port.send(msg)


