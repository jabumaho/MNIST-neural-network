from matplotlib import pyplot as plt

path = "C:/Temp/mnisterrors/chunk" + str(input("chunk: ")) + ".txt"

with open(path, "r") as f:
	errorhistory = [float(line.rstrip('\n')) for line in f]

plt.plot(errorhistory)
plt.show()
