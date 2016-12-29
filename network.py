import numpy as np


def sgm(x, derivative=False):
	if not derivative:
		return 1/(1+np.exp(-x))
	else:
		return sgm(x) * (1 - sgm(x))


def linear(x, derivative=False):
	if not derivative:
		return x
	else:
		return 1


class NeuralNetwork:
	layerCount = 0
	shape = None
	weights = []
	layerTransferFunc = []

	def __init__(self, layerSize, layerTransferFunc=None):
		self.layerCount = len(layerSize) - 1
		self.shape = layerSize

		self._layerInput = []
		self._layerOutput = []
		self._previousWeightDelta = []

		for (l1, l2) in zip(layerSize[:-1], layerSize[1:]):
			self.weights.append(np.random.normal(scale=0.1, size=(l2, l1 + 1)))
			self._previousWeightDelta.append(np.zeros(shape=(l2, l1 + 1)))

		if layerTransferFunc is None:
			layerTransferFunc = []
			for i in range(self.layerCount):
				if i == self.layerCount - 1:
					layerTransferFunc.append(sgm)
				else:
					layerTransferFunc.append(sgm)
		else:
			if len(layerTransferFunc) != len(layerSize):
				raise ValueError("Incompatible no of transfer functions.")
			elif layerTransferFunc[0] is not None:
				raise ValueError("no transfer functions for input layer.")
			else:
				layerTransferFunc = layerTransferFunc[1:]

		self.layerTransferFunc = layerTransferFunc

	def run(self, inputr):

		lnCases = inputr.shape[0]

		self._layerInput = []
		self._layerOutput = []

		for i in range(self.layerCount):
			if i == 0:
				layerInput = self.weights[0].dot(np.vstack([inputr.T, np.ones([1, lnCases])]))
			else:
				layerInput = self.weights[i].dot(np.vstack([self._layerOutput[-1], np.ones([1, lnCases])]))

			self._layerInput.append(layerInput)
			self._layerOutput.append(self.layerTransferFunc[i](layerInput))

		return self._layerOutput[-1].T

	def trainEpoch(self, inputt, target, trainingRate=0.5, momentum=0.5):
		delta = []
		lnCases = inputt.shape[0]

		self.run(inputt)

		for i in reversed(range(self.layerCount)):

			if i == self.layerCount - 1:
				output_delta = self._layerOutput[i] - target.T
				error = 0.5 * np.sum(output_delta**2)
				delta.append(output_delta * self.layerTransferFunc[i](self._layerInput[i], True))
			else:
				deltaPullback = self.weights[i + 1].T.dot(delta[-1])
				delta.append(deltaPullback[:-1, :] * self.layerTransferFunc[i](self._layerInput[i], True))
		for i in range(self.layerCount):
			deltaIndex = self.layerCount - 1 - i
			if i == 0:
				layerOutput = np.vstack([inputt.T, np.ones([1, lnCases])])
			else:
				layerOutput = np.vstack([self._layerOutput[i - 1], np.ones([1, self._layerOutput[i - 1].shape[1]])])

			currentweightDelta = np.sum(layerOutput[None, :, :].transpose(2, 0, 1) * delta[deltaIndex][None, :, :].transpose(2, 1, 0), axis=0)

			weightDelta = trainingRate * currentweightDelta + momentum * self._previousWeightDelta[i]

			self.weights[i] -= weightDelta
			self._previousWeightDelta[i] = weightDelta
		return error

	def test_network(self, inputtest, target):
		self.run(inputtest)

		output_delta = self._layerOutput[self.layerCount - 1] - target.T
		return 0.5 * np.sum(output_delta**2)

	def nudge(self, scale):
		for i in xrange(len(self.weights)):
			for j in xrange(len(self.weights[i])):
				for k in xrange(len(self.weights[i][j])):
					w = self.weights[i][j][k]
					w *= scale
					u = np.random.normal(scale=abs(w))
					self.weights[i][j][k] += u
