input_notes = "F A B F A B F A B E D B C B G E D E G E F A B F A \
B F A B E D B C E B G B G D E F G A B C D E F G \
G F A G B A C B D C E D F E B C A B"
#these are the 3 octave vals
notes_to_val = {"A":36,
				"B":32,
				"C":60,
				"D":53,
				"E":47,
				"F":45,
				"G":40}
output = ""
for i in input_notes:
	if i != " ":
		print i
		print type(i)
		output += str(notes_to_val[i])
		output += ", "

print output

