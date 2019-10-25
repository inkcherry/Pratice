

import  os

filepath="/home/1.txt"
print(os.path.basename(filepath))

filepath2="234.txt"
print(os.path.splitext(filepath2)[0])



# midi_name = os.path.splitext(os.path.basename(filepath))[0]
# print("234")
# print(midi_name)