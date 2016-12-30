from network_controller import *

controller = NetworkController("./data")

while True:
	a = input("To train a new network press 1, to load an existing network press 2 or press 99 to exit:")
	if a == 1:
		controller.setup_network()
		controller.train_network(limit=int(input("Number of training chunks: ")), chunksize=int(input("Number of images per chunk: ")))
		controller.save()
	elif a == 2:
		controller.load()
		print "Average error: {0}".format(controller.get_avg_error())
		while True:
			i = input("Index of image for testing (0 - 9999), anything else to exit: ")
			if 0 <= i <= 9999:
				controller.test_image(i)
			else:
				break
	elif a == 99:
		break
	else:
		print "Please chose a valid option, you person with a melting popsicle as a brain"
