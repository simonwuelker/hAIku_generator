import os
import sys
import numpy as np
import Tools
count = 0

def convert(path, store_in):
	global count
	for filename in os.listdir(path):
		file_path = f"{path}/{filename}"
		print(file_path)
		if os.path.isfile(file_path):
			print(f"File Nr.{count}: {filename}")
			count += 1
			try:
				messages = Tools.encode(file_path, 12)
				np.save(f"{store_in}/{filename}.npy", messages)
			except:
				os.remove(file_path)

		#recursively go through subdirectories
		elif os.path.isdir(file_path):
			convert(file_path, store_in)

convert("/home/kepler/Desktop/midi_files", "/home/kepler/Desktop/notewise")
