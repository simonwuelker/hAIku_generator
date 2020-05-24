import os
import sys
import numpy as np
import MidiTools

path = "C:/Users/Wuelle/Documents/KI-Bundeswettbewerb-2020/Datensatz/lmd_full"
store_in = "notewise"#"C:/Users/Wuelle/Documents/KI-Bundeswettbewerb-2020/Datensatz/notewise"

for index, filename in enumerate(os.listdir(path)):
	file_path = f"{path}/{filename}"
	if os.path.isfile(file_path):
		print(f"File Nr.{index}: {filename}")
		try:
			messages = MidiTools.encode(file_path, 12)
			print(f"{store_in}/{filename}")
			np.save(f"{store_in}/{filename}.npy", messages)
		except:
			os.remove(file_path)