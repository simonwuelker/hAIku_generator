import mido
import numpy as np

array = np.load("output_array.npy")

def getNoteLength(start_idx, note):
	end_idx = start_idx
	while array[end_idx][note] and end_idx != array.shape[0]:
		if end_idx < array.shape[0]-1:
			end_idx += 1
		else:
			break
	return (end_idx-start_idx)/tps



mid = mido.MidiFile(type = 0)
track = mido.MidiTrack()
mid.tracks.append(track)
samplerate = 100

mid.ticks_per_beat = 96
beats_per_minute = 120
tempo = mido.bpm2tempo(beats_per_minute)

track.append(mido.Message('program_change', program=12, time=0))

playing = np.zeros(128)
#prevtimeindex merkt sich den zeitpunkt der letzten note, wichtig für dtime(sehr gute fehlerquelle)
prev_time_index = 0

for time_index, state in enumerate(array):
	for note_index, note in enumerate(state):
		if note and not playing[note_index]:
			track.append(mido.Message('note_on', note=note_index, velocity=64, time=int(mido.second2tick((time_index-prev_time_index)/samplerate, mid.ticks_per_beat, tempo))))
			playing[note_index] = True
			prev_time_index = time_index

		elif not note and playing[note_index]:
			track.append(mido.Message('note_off', note=note_index, velocity=64, time=int(mido.second2tick((time_index-prev_time_index)/samplerate, mid.ticks_per_beat, tempo))))
			playing[note_index] = False
			prev_time_index = time_index



mid.save("from_array.mid")

