import mido
import sys

file = mido.MidiFile(sys.argv[1]) 

#msg Object structure: {'type': 'note_on', 'time': 0, 'channel': 0, 'note': 1, 'velocity': 100}
for msg in file.play():
	print(msg)
