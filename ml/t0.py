a = open("out2.txt","r").read().split("\n")
total 	= 0 
elp = [] 
for item in a[1:]:
	c = item.split(" ")
	c = [l for l in c if not l == ""]
	try:
		c[1] = float(c[1])
		total += c[1]
		if c[1] > .1:
			elp.append((c[1],"".join(c[5:])))
	except ValueError:
		pass
	except IndexError:
		pass
import pprint 
print(f"Total: {total}")
pprint.pp(sorted(elp,reverse=True))