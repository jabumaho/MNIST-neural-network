import load_mnist
from network import *
from random import randint


class NetworkController:
	def __init__(self, path):
		self.dataloader = load_mnist.MnistLoader(path=path)
		self.images, self.labels = self.dataloader.load_training()
		self.timages, self.tlabels = self.dataloader.load_testing()
		self.maxerror = 1e-2
		self.maxIterations = 200000
		self.errorhistory = []
		self.nudge_tolerance = 0.00001
		self.nudge_scale = 0.001
		self.nudge_window = 500

	def setup_network(self):
		self.network = NeuralNetwork((784, 50, 10))

	def run_network(self, images):
		return self.network.run(images)

	def get_training_chunk(self, size):
		ichunk = []
		lchunk = []
		for i in xrange(size):
			r = randint(0, len(self.images)-1)
			ichunk.append(self.images[r])
			lchunk.append(self.labels[r])
		return np.array(ichunk), np.array(lchunk)

	def get_testing_chunk(self, size):
		ichunk = []
		lchunk = []
		for i in xrange(size):
			r = randint(0, len(self.timages)-1)
			ichunk.append(self.timages[r])
			lchunk.append(self.tlabels[r])
		return np.array(ichunk), np.array(lchunk)

	def train_network(self, limit=6000, chunksize=10):
		for i in xrange(limit):
			im, la = self.get_training_chunk(chunksize)
			self.train_chunk(i, im, la)
			self.save_errorhistory(self.errorhistory, i)

		print "\nTraining complete, running test data..."

		avg = self.get_avg_error()

		print "Testing complete, default error: {0}".format(avg)

	def train_chunk(self, it, trainchunk, labelchunk):
		print "Training on data chunk {0}".format(it)
		finished = False
		self.errorhistory = []
		for i in range(self.maxIterations + 1):
			self.error = self.network.trainEpoch(trainchunk, labelchunk, trainingRate=0.3, momentum=0.3)
			self.errorhistory.append(self.error)
			if self.error <= self.maxerror:
				print "Finished training on datachunk {0} on iteration {1} with error {2}".format(it, i, self.error)
				finished = True
				break
			if i % self.nudge_window == 0:
				self.checknudge(i)
		if not finished:
			print "Maximum number of itertions reached on chunk {0} with error {1}".format(it, self.error)

	def save(self):
		path = input("Path to location: ")
		path += input("Name of network: ") + ".nn"
		#name = input("File Name for saved network: ")
		#path = "C:/Temp/" + name + ".nn"
		with open(path, "w+") as file:
			for i in self.network.shape:
				file.write(str(i))
				file.write(" ")
			file.write("\n")
			for layer in self.network.weights:
				for node in layer:
					for weight in node:
						file.write(str(weight))
						file.write(" ")
					file.write("\n")
				file.write("\n")

	def load(self):
		path = input("Path to file: ")
		#name = input("File Name: ")
		#path = "C:/Temp/" + name + ".nn"
		with open(path, "r") as file:
			f = file.read()
		strlen = len(f)
		shape = ()
		weights = []
		z = 0
		dump = ""
		while not f[z] == "\n":
			if not f[z] == " ":
				dump += f[z]
			else:
				shape += (int(dump),)
				dump = ""
			z += 1
		while z < strlen:
			ldump = []
			while not f[z] == "\n" and not f[z+1] == "\n":
				ndump = []
				while not f[z] == "\n":
					if not f[z] == " ":
						dump += f[z]
					else:
						ndump.append(float(dump))
						dump = ""
					z += 1
				ldump.append(ndump)
				z += 1
			weights.append(np.array(ldump))
			z += 1
		weights.pop(0)

		self.network = NeuralNetwork(shape)
		self.network.weights = weights

	def checknudge(self, iteration):
		if iteration < 2 * self.nudge_window:
			return
		oldavg, newavg = 0, 0
		l = len(self.errorhistory)

		for i in xrange(self.nudge_window):
			oldavg += self.errorhistory[l - 2 * self.nudge_window +i]
			newavg += self.errorhistory[l - self.nudge_window + i]
		oldavg /= self.nudge_window
		newavg /= self.nudge_window

		if float(abs(newavg - oldavg)) / self.nudge_window < self.nudge_tolerance:
			self.network.nudge(self.nudge_scale)

	@staticmethod
	def save_errorhistory(history, chunk):
		path = input("Path to file: ")
		#path = "C:/Temp/mnisterrors/chunk" + str(chunk) + ".txt"
		with open(path, "w+") as file:
			for herror in history:
				file.write(str(herror))
				file.write("\n")

	def test_image(self, i):
		print self.dataloader.display(self.timages[i])
		o = self.run_network(np.array([self.timages[i]]))

		print "Output:\t\t\tDesired Output:\n"
		z = 0
		for n in o[0]:
			print "{0}: {1:.4f}\t\t{0}: {2}".format(z, n, self.tlabels[i][z])
			z += 1

	def get_avg_error(self):
		avgerror = 0.0
		for i in xrange(len(self.timages)):
			o = self.run_network(np.array([self.timages[i]]))
			w = self.tlabels[i]
			err = []
			for i in xrange(len(o[0])):
				err.append(abs(o[0][i] - w[i]))
			avg = sum(err)/len(err)
			avgerror += avg
		avgerror /= len(self.timages)
		return avgerror
