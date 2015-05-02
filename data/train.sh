caffe train -solver models/solver.prototxt $1 $2 2>&1 | grep -i 'iter\|snapshot\|accuracy\|loss\|trace'
