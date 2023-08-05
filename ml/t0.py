a = open("out2.txt","r").read().split("\n")

elp = [] 
for item in a[1:]:
	c = item.split(" ")
	c = [l for l in c if not l == ""]
	try:
		c[1] = float(c[1])
		if c[1] > .1:
			elp.append((c[1],"".join(c[5:])))
	except ValueError:
		pass
	except IndexError:
		pass
import pprint 
pprint.pp(sorted(elp,reverse=True))