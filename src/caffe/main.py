#! /usr/bin/env python2

import caffe
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--buildmodel")

args = parser.parse_args()

# Globals and Paths
MODEL_DEFINITION = ""
PRETRAINED_MODEL = "models/model.caffemodel"
IMG_TRAIN_PATH = "../../data/sample" # "../../data/train"
IMG_TEST_PATH = "../../data/test"
LABELS = "../../data/trainLabels.csv"

caffe.set_mode_cpu()

if args.buildmodel:
	if os.path.isfile(args.buildmodel):
		print("Building model to " + PRETRAINED_MODEL)
		solver = caffe.SGDSolver(args.buildmodel)

	else:
		print(args.buildmodel + " is not a valid file!")

	exit()



if not os.path.isfile(PRETRAINED_MODEL):
	print("No pretrained model found!")

	exit()

#solver = caffe.SGDSolver("lenet_solver.prototxt")
print("Hello, world!")
