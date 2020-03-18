import mido

tempo = None

def second2tick(seconds):
	#0 scond events are usually meta, tempo might not be initialized yet
	if seconds != 0:
		#make sure the tempo is known
		if tempo is None:
			raise ValueError

		ticks = round(ticks_per_second*seconds)
		return ticks
	else:
		return 0

#Load the original midi file
file = mido.MidiFile("example.mid")
ticks_per_beat = file.ticks_per_beat


#create new midi file(This is our type 0 file)
new = mido.MidiFile(type = 0, ticks_per_beat = file.ticks_per_beat) 
track = mido.MidiTrack()
new.tracks.append(track)

#time is given in seconds and must be converted to ticks using second2tick()
for msg in file:
	#'set_tempo' event is special bc the second2tick() method depends on the tempo
	if msg.type == "set_tempo":
		tempo = msg.tempo
		beats_per_minute = mido.tempo2bpm(tempo)
		ticks_per_second = (ticks_per_beat*beats_per_minute)/60

	track.append(msg.copy(time = second2tick(msg.time)))

new.save("example.mid")
