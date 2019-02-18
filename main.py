
#########################################
#    Main File for MNIST Prediction     #
#                                       #
#########################################

###Required installations: wget

####Packages to import
import os
import wget

url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
wget.download(url)
