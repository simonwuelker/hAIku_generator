#Documentation for MIDIUtils:https://midiutil.readthedocs.io/en/latest/

from midiutil.MidiFile import MIDIFile

# create your MIDI object
mf = MIDIFile(1)#1 track, all time in quarternotes, 120 ticks per bpm
track = 0
time = 0
bpm = 120
mf.addTrackName(track, time, "Artificially generated music")
mf.addTempo(track, time, bpm)

# add some notes
channel = 0
volume = 100

pitch = 1
time = 0
duration = 2#duration is given in 1/2 seconds
mf.addNote(track, channel, pitch, time, duration, volume)


pitch = 32
time = 2
duration = 4
mf.addNote(track, channel, pitch, time, duration, volume)

# write it to disk
with open("output.mid", 'wb') as outf:
    mf.writeFile(outf)