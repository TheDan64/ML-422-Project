#!/usr/bin/env sh
# Takes an lmdb image database and computes the mean

compute_image_mean $1 models/train_mean.binaryproto

echo "Done."
