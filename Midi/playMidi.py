import mido
import sys

#define a port to stream the messages to
port = mido.open_output()

#open the .mid file
file = mido.MidiFile(sys.argv[1])

for msg in file.play():
    port.send(msg)
