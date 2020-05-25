import mido 
import numpy as np
import random
import os
import torch

with open("C:/Users/Wuelle/Documents/KI-Bundeswettbewerb-2020/BW-KI-2020/vocab.txt", "r") as vocab:
	words = vocab.read().split(",")
	vocab_size = len(words)
	ix_to_msg = {ix:msg for ix, msg in enumerate(words)}
	msg_to_ix = {msg:ix for ix, msg in enumerate(words)}

def fetch_sample(length, dataset_path, encode = True):
	while True:
		filename = dataset_path + random.choice(os.listdir(dataset_path))
		print(f"Now sampling from {filename}")
		sample = np.load(filename)
		for i in range(int(len(sample)/length)):
			part = sample[i*length:(i+1)*length]
			if encode:
				yield OneHotEncode(torch.from_numpy(part)).unsqueeze(1).float()
			else:
				yield part

def OneHotEncode(sequence):
	result = torch.zeros(sequence.shape[0], vocab_size)
	for index, element in enumerate(sequence):
		result[index, element.int().item()] = 1
	return result


def roundEventTime(time, samplerate):
	return round(time*samplerate)/samplerate

def getVocabSize():
	return len(words)

def encode(path, samplerate, lower_bound = 20, upper_bound = 81):
	file = mido.MidiFile(path)

	messages = []
	for msg in file:
		d_time = roundEventTime(msg.time, samplerate)

		#account for delay by waiting 1/samplerate of a second n times
		for _ in range(int(d_time*samplerate)):
			messages.append(msg_to_ix["wait12"])


		if msg.type == "note_on":
			if msg.note > lower_bound and msg.note < upper_bound:
				if msg.velocity != 0:
					messages.append(msg_to_ix["note_on:{}".format(msg.note)])
				else:
					messages.append(msg_to_ix["note_off:{}".format(msg.note)])

		elif msg.type == "note_off":
			if msg.note > lower_bound and msg.note < upper_bound:
				messages.append(msg_to_ix["note_off:{}".format(msg.note)])

	#remove all delays at the start and the end of the track(why do these exist in the first place????)
	for i in range(len(messages)):
		if messages[i] != msg_to_ix["wait12"]:
			break
			
	for j in range(len(messages)-1, -1, -1):
		if messages[j] != msg_to_ix["wait12"]:
			break
	messages = messages[i:j+1]

	return np.array(messages)

def decode(messages):
	mid = mido.MidiFile(type = 0)
	track = mido.MidiTrack()
	mid.tracks.append(track)

	mid.ticks_per_beat = 96
	beats_per_minute = 120
	tempo = mido.bpm2tempo(beats_per_minute)
	msg_time = 0

	track.append(mido.Message('program_change', program=11, time=0))
	for msg in messages:
		msg_txt = ix_to_msg[msg]
		if msg_txt.find("note_on") != -1:
			track.append(mido.Message('note_on', note=int(msg_txt[msg_txt.index(":")+1:])+20, velocity=64, time=msg_time))
			msg_time = 0
		elif msg_txt.find("note_off") != -1:
			track.append(mido.Message('note_off', note=int(msg_txt[msg_txt.index(":")+1:])+20, velocity=64, time=msg_time))
			msg_time = 0

		elif msg_txt.find("wait12") != -1:
			msg_time += int(mido.second2tick(1/12, mid.ticks_per_beat, tempo))
	track.append(mido.MetaMessage('end_of_track', time=msg_time))
	return mid

def playMidi(file):
	#define a port to stream the messages to
	port = mido.open_output()

	for msg in file.play():
		print(msg)
		port.send(msg)