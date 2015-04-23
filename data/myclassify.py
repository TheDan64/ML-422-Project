#! /usr/bin/env python2

import caffe
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import csv
import os
import argparse

# Globals and Paths
MODEL = "models/deploy.prototxt"
PRETRAINED = "snapshots/snapshot_iter_200.caffemodel"
IMG_DIR = "sample/"

caffe.set_mode_cpu()

if os.path.isfile(MODEL):
	print("Model confirmed")
if os.path.isfile(PRETRAINED):
	print("Pretrained confirmed")

net = caffe.Classifier(MODEL, PRETRAINED, #mean=np.load("/home/dkolsoi/repos/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy").mean(1).mean(1),
	channel_swap=(2, 1, 0), raw_scale=255, image_dims=(256, 256))

# print("Started pathing images.")
# print("Started loading images.")

def get_predictions(dir):
	for img in os.listdir(dir):
		if img[-5:] == ".jpeg":
			yield (img[-5:], net.predict([caffe.io.load_image(dir + img)]))

# print("Finished loading images.")
# predictions = (net.predict([img]) for img in image_list)
# print("Finished making predictions.")

with open("submission.csv", "w") as csvfile:
	print("Writing predicitions.")

	# names = []
	# with open("test.txt", 'rb') as f:
	# 	for row in csv.reader(f, delimiter=' '):
	# 		names.append(row[0])

	writer = csv.writer(csvfile, delimiter=',')

	writer.writerow(["image", "level"])

	count = 1

	for img in os.listdir(IMG_DIR):
		start = time.time()
		if img[-5:] == ".jpeg":
			predictions = net.predict([caffe.io.load_image(IMG_DIR + img)])
			writer.writerow([img[:-5], predictions.argmax()])
			sys.stdout.write("Iter {} took {:.2f}s and predicted {}    \r".format(count, time.time() - start, predictions))
			sys.stdout.flush()
			count += 1

	# for (name, pred) in get_predictions("test/"):
	# 	print(str(count) + "th image processed")
	# 	output_buf += name + ',' + str(pred.argmax()) + '\n'

	# 	if not count % 50:
	# 		print("Writing to buffer")
	# 		f.write(output_buf)
	# 		output_buf = ""

	# 	count += 1

print("\nMade it to the end!")
