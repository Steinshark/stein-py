a = open("out2.txt","r").read().split("\n")
b = a[a.index('begin')+1:]

elp = [] 
for item in b[1:]:
  c = item.split(" ")
  c = [l for l in c if not l == ""]
  c[1] = float(c[1])
  if c[1] > 1:
    elp.append((c[1],"".join(c[5:])))
import pprint 
pprint.pp(elp)