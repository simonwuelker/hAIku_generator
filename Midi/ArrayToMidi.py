#Documentation for MIDIUtils:https://midiutil.readthedocs.io/en/latest/
#from midiutil.MidiFile import MIDIFile
import mido
import numpy as np

array = np.load("Files/output_array.npy")

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
			#print("Dtime = {}".format(mido.second2tick((time_index-prev_time_index)/10, mid.ticks_per_beat, tempo)))
			#print("Dtime = {}".format((time_index-prev_time_index)/10))
			track.append(mido.Message('note_on', note=note_index, velocity=64, time=int(mido.second2tick((time_index-prev_time_index)/samplerate, mid.ticks_per_beat, tempo))))
			playing[note_index] = True
			prev_time_index = time_index

		elif not note and playing[note_index]:
			track.append(mido.Message('note_off', note=note_index, velocity=64, time=int(mido.second2tick((time_index-prev_time_index)/samplerate, mid.ticks_per_beat, tempo))))
			playing[note_index] = False
			prev_time_index = time_index



mid.save("Files/from_array.mid")


"""
# create your MIDI object
mf = MIDIFile(1)
track = 0
tps = 4	#4 ticks per second

time = 0    # start at the beginning
mf.addTrackName(track, time, "Music read from .npy file")
mf.addTempo(track, time, 120)

channel = 0
volume = 100

playing = np.zeros(128)

for tick_idx in range(array.shape[0]):
	for note_idx in range(array.shape[1]):
		if array[tick_idx, note_idx] == 1 and not playing[note_idx]:
				duration = getNoteLength(tick_idx, note_idx)*2
				mf.addNote(track, channel, note_idx, tick_idx*0.5, duration, volume)
				playing[note_idx] = 1
		elif array[tick_idx, note_idx] == 0 and playing[note_idx]:
			playing[note_idx] = 0



# write it to disk
with open("from_array.mid", 'wb') as outf:
    mf.writeFile(outf)
"""
