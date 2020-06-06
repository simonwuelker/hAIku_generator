import string

with open("haikus.csv", "r", encoding="utf8", errors="ignore") as infile:
	data = infile.readlines()

result = []
tags = ["tempslibres", "sballas", "img2poems", "twaiku", "haikuzao"]

alphabet = string.ascii_lowercase + "," + " " + "\n"

for line in data:
	new = ''.join(c for c in line if c.lower() in alphabet)
	for i in range(100):
		new = new.replace("  ", " ")
		new = new.replace(", ", ",")
		new = new.replace(" ,", ",")

	for tag in tags:
		index = new.find(tag)
		if index != -1:
			new = new[:index-1] + "\n"

	result.append(new)

with open("dataset.txt", "w", errors="ignore") as outfile:
	outfile.writelines(result)

print(set("".join(result)))
