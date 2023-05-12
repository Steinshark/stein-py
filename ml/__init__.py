#   Author:     Everett Stenbeg 
#   Github:     Steinshark
import os 
#Ensure path exists 
root = "C:/" if not "posix" in os.name else "/home/steinshark/"

for path in ["data","chess"]:
    root += path  

    if not os.path.exists(root):
        os.mkdir(root)

for path in ["models","experiences"]:
    if not os.path.exists(root+"/"+path):
        os.mkdir(root+"/"+path)

if not os.path.exists(root+"/experiences/gen1"):
    os.mkdir(root+"/experiences/gen1")
