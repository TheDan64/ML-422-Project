#! /usr/bin/env python2

import caffe
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import csv
import os
import argparse

parser = argparse.ArgumentParser(description="Classify eye images in a path")
parser.add_argument("-ip", "--img_path", help="Directory of images")
parser.add_argument("-d", "--deploy", help="Path to deploy.prototxt model")
parser.add_argument("-cm", "--caffemodel", help="Path to pretrained .caffemodel")
parser.add_argument("-m", "--mean", help="Path to .binaryproto mean file")
args = parser.parse_args()

caffe.set_mode_gpu()

np_mean = None

if args.mean:
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.ParseFromString(open(args.mean, "rb").read())
        np_mean = np.asarray(blob.data).reshape(1, 256, 256)

print("Loading:")
print("Images directory: " + args.img_path)
print("Deploy: " + args.deploy)
print("Pretrained mode: " + args.caffemodel)
print("Images mean: " + str(args.mean))
raw_input("Press Enter to start: ")

net = caffe.Classifier(args.deploy, args.caffemodel, mean=np_mean,
	#channel_swap=(2, 1, 0), 
        raw_scale=255,
        image_dims=(256, 256))

with open("submission.csv", "w") as csvfile:
	print("Writing predicitions.")

	writer = csv.writer(csvfile, delimiter=',')

	writer.writerow(["image", "level"])

	count = 1

	for img in os.listdir(args.img_path):
		start = time.time()
		if img[-5:] == ".jpeg":
#                        np_img = map(lambda x: map(lambda y: [y[0]], x), caffe.io.load_image(args.img_path + img))
			predictions = net.predict([caffe.io.load_image(args.img_path + img, False)])#, oversample=False) # I think oversample does it a bunch of times and takes the average
			writer.writerow([img[:-5], predictions.argmax()])
			sys.stdout.write("{}% took {:.2f}s       \r".format(count/535.76, time.time() - start)) # should use /file len * 100
			sys.stdout.flush()
			count += 1

print("\nMade it to the end!")
