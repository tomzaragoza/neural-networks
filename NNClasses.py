import math

class Neuron:
	def __init__(self, weight=None, delta=None, net_value=None, output=None, bias=None):
		self.input_neurons = {} # neuron's id -> Neuron Object
		self.neurons_connected_to = {} # the neurons that this neuron is connected/directed to
		self.weight = weight
		self.delta = delta
		self.net_value = net_value
		self.output = output
		self.bias = bias
		self.error_factor = None

	def UpdateFreeParameters(self, learning_rate=None):
		""" Update the free parameters. Free parameters include the bias and weight values""" 
		self.bias += learning_rate * 1 * self.delta

		for neuron in self.input_neurons.iteritems():
			neuron.weight += learning_rate * 1 * neuron.output * self.delta

	def UpdateOutput(self):
		""" Calculate the input value through the summation unit. 
			Then, pass the value from the summation unit to the
			transfer function to determine the output of the neuron. """
		SummationUnit()
		TransferUnit(summation_unit_value=self.net_value)

	def SummationUnit(self):
		""" Calculate the net value of the neuron based on
		the weights of the input neurons. """

		if self.bias:
			self.net_value += self.bias

		for neuron in self.input_neurons.iteritems():
			self.net_value += neuron.weight * neuron.output

	def TransferUnit(self, summation_unit_value):
		""" Calculate the output value using a sigmoid function """
		self.output = 1 / (1 + math.exp(-1 * summation_net_value)) #should be exponent of negative net value

class Layer:
	def __init__(self):
		self.neurons = []

class HiddenLayer:
	def __init__(self):
		self.layers = []


class NeuralNetwork:
	def __init__(self, learning_rate=None, expected_output=0, num_of_inputs=None, num_of_hidden_layers=None, num_of_outputs=None):
	#def __init__(self, num_neurons_input_layer=None, num_neurons_hidden_layer=None, num_neuron_output_layer=None):
		self.learning_rate = learning_rate
		self.input_layer = Layer()
		self.hidden_layers = []  # sequentially, the order of the layers
		self.output_layer = Layer()
		self.expected_output = expected_output

		# run this initially to find all neurons that are connected to what neurons
		GetNeuronsConnectedTo()

	def GetNeuronsConnectedTo(self):
		""" Find all neurons that a particular neuron is connected to. 
			Start with neurons connected to output layer all the way up to
			neurons in the first hidden layer. """

		for neuron in self.output_layer:
			for input_neuron in neuron.input_neurons.iteritems():
				input_neuron.neurons_connected_to[neuron.id] = neuron

		for layer in self.hidden_layers:
			for neuron in layer:
				for input_neuron in neuron.input_neurons.iteritems():
					input_neuron.neurons_connected_to[neuron.id] = neuron

	def AdjustFreeWeightsBias(self):
		""" Adjust the net values and weights of all neurons
			in the output layer and hidden layers. This is
			called when feed forward propagation has finished. """
		for layer in self.hidden_layers:
			for neuron in layer.neurons:
				neuron.UpdateFreeParameters()

		for neuron in self.output_layer.neurons:
			neuron.UpdateFreeParameters()

	def DeltaErrorfactorOutputLayerNeurons(self):
		""" Calculate the deltas of all outputlayer neurons. 
			This must be done before calculating the deltas of all
			the previous hidden layer neurons for backpropagation to work. """
		for output_layer_neuron in self.output_layer:
			# first calculate ALL error factors of output layer neurons
			output_layer_neuron.error_factor = self.expected_output - output_layer_neuron.output
		# Then, after caluclating all error factors of output layer neurons, you can directly get the delta
		output_layer_neuron.delta = output_layer_neuron.output * (1 - output_layer_neuron.output) * output_layer_neuron.error_factor

	def DeltaErrorFactorHiddenLayerNeuron(self):
		""" Calculate the error factor of a neuron in a hidden layer. """
		for hidden_layer in self.hidden_layers:
			for hidden_layer_neuron in hidden_layer:
				for connected_neuron in hidden_layer_neuron.input_neurons.iteritems():
					# First calculate all error factors of the current neuron using the neurons connected to it
					hidden_layer_neuron.error_factor += connected_neuron.delta * connected_neuron.weight 
				hidden_layer_neuron.delta += hidden_layer_neuron.output * (1 - hidden_layer_neuron.output) * hidden_layer_neuron.error_factor


if __name__ == "__main__":
	# Using a NN for AND logic gate as an example

	NN = NeuralNetwork()

	# Create two input neurons for the input layer
	n1 = Neuron()
	n2 = Neuron()

	NN.input_layer.extend([n1, n2])

	# Create two hidden layer neurons for the 1 hidden layer in the AND logic gate
	# then create and append the neurons to the layer
	hidden_layer_1 = Layer()
	n3 = Neuron()
	n4 = Neuron()
	hidden_layer_1.neurons.extend([n3, n4])
	NN.hidden_layers.append(hidden_layer_1)

	#create one output layer and one neuron for that output layer
	output_layer = Layer()
	n5 = Neuron
	output_layer.neurons.append(n5)

	# Find the neurons that each neuron is connected to
	# This is done after building the NN
	NN.GetNeuronsConnectedTo()




